import os
from argparse import ArgumentParser
from functools import partial

import h5py
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import ImageFilter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataloader import EstimationFileDataset, EstimationHdf5DatasetMyDataset
from model import GazeEstiamationModel_resent18, GazeEstimationModel_vgg16, GazeEstimationPreactResnet
from GazeAngleAccuracy import GazeAngleAccuracy

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback

# Custom model checkpoint callback class
class CustomModelCheckpoint(pl.Callback):
    def __init__(self, dirpath):
        self.dirpath = dirpath

        # Ensure that the directory exists, create if it doesn't
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def on_epoch_end(self, trainer, pl_module):
        # Check if 'val_loss' is present in trainer.callback_metrics
        if not hasattr(trainer, 'callback_metrics') or 'val_loss' not in trainer.callback_metrics:
            print("Warning: 'val_loss' not found in trainer.callback_metrics. Skipping model checkpoint.")
            return

        # Get the current validation loss from trainer.callback_metrics
        val_loss = trainer.callback_metrics['val_loss']

        # Save the model with a simple name
        filename = os.path.join(self.dirpath, f"model_{trainer.current_epoch}.pth")
        torch.save(pl_module.state_dict(), filename)

# Training model class
class TrainEstimationModel(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, test_subjects):
        super(TrainEstimationModel, self).__init__()

        # Dictionary to store loss functions
        _loss_fn = {
            "mse": torch.nn.MSELoss
        }

        # Dictionary to store the number of output parameters
        _param_num = {
            "mse": 2
        }

        # Dictionary to map model names to their corresponding classes
        _models = {
            "vgg16": partial(GazeEstimationModel_vgg16, num_out=_param_num.get(hparams['loss_fn'])),
            "resnet18": partial(GazeEstiamationModel_resent18, num_out=_param_num.get(hparams['loss_fn'])),
            "preactresnet": partial(GazeEstimationPreactResnet, num_out=_param_num.get(hparams['loss_fn']))
        }

        # Initialize the selected model
        self._model = _models.get(hparams['model_base'])()

        # Initialize the selected loss function
        self._criterion = _loss_fn.get(hparams['loss_fn'])()

        # Initialize the angle accuracy metric
        self._angle_acc = GazeAngleAccuracy()

        # Store the training, validation, and test subjects
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._test_subjects = test_subjects

        # Store the hyperparameters
        self._hparams = hparams

        # Dictionary to store validation outputs
        self.val_outputs = {'val_loss': [], 'angle_acc': []}

        # Dictionary to store test outputs
        self.test_outputs = {'angle_acc': []}

    def forward(self, left_patch, right_patch, head_pose):
        # Forward pass through the model
        return self._model(left_patch, right_patch, head_pose)

    def training_step(self, batch, batch_idx):
        # Extract data from the batch
        _left_patch, _right_patch, _headpose_label, _gaze_labels, _landmark_labels = batch

        # Forward pass through the model
        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)

        # Calculate the loss
        loss = self._criterion(angular_out, _gaze_labels)

        # Log the training loss
        self.log("train_loss", loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Set the model to evaluation mode (disable dropout)
        self.eval()

        # Extract data from the batch
        _left_patch, _right_patch, _headpose_label, _gaze_labels, _landmark_labels = batch

        # Forward pass through the model
        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)

        # Calculate the loss
        loss = self._criterion(angular_out, _gaze_labels)

        # Calculate angle accuracy
        angle_acc = self._angle_acc(angular_out[:, :2], _gaze_labels)

        # Store the validation loss and angle accuracy
        self.val_outputs['val_loss'].append(loss)
        self.val_outputs['angle_acc'].append(angle_acc)

        return self.val_outputs

    def on_validation_epoch_end(self):
        # Stack the validation losses and angle accuracies
        _losses = torch.stack([x for x in self.val_outputs['val_loss']])
        _angles = np.array([x for x in self.val_outputs['angle_acc']])

        # Compute the mean validation loss and angle accuracy
        self.log('val_loss', _losses.mean(), prog_bar=True)
        self.log('val_acc', _angles.mean(), prog_bar=True)

        # Clear the validation outputs
        self.val_outputs.clear()
        self.val_outputs = {'val_loss': [], 'angle_acc': []}

    def test_step(self, batch, batch_idx):
        # Extract data from the batch
        _left_patch, _right_patch, _headpose_label, _gaze_labels, _landmark_labels = batch

        # Forward pass through the model
        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)

        # Calculate angle accuracy
        angle_acc = self._angle_acc(angular_out[:, :2], _gaze_labels)

        # Store the test angle accuracy
        self.test_outputs['angle_acc'].append(angle_acc)

        return self.test_outputs

    def on_test_epoch_end(self):
        # Stack the test angle accuracies
        _angles = np.array([x for x in self.test_outputs['angle_acc']])

        # Log the mean and standard deviation of test angle accuracy
        self.log("test_angle_mean", _angles.mean())
        self.log("test_angle_std", _angles.std())

        # Clear the test outputs
        self.test_outputs.clear()
        self.test_outputs = {'angle_acc': []}

    def configure_optimizers(self):
        # Get the model parameters that require gradients
        _params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                _params_to_update.append(param)

        # Initialize the optimizer with the selected learning rate
        _learning_rate = self._hparams['learning_rate']
        _optimizer = torch.optim.Adam(_params_to_update, lr=_learning_rate)

        # Initialize the learning rate scheduler
        _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=30, gamma=0.1)

        return [_optimizer], [_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # Add arguments for data augmentation, loss function, batch size, etc.
        parser.add_argument('--augment', action="store_true", dest="augment")
        parser.add_argument('--loss_fn', choices=["mse"], default="mse")
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--batch_norm', default=True, type=bool)
        parser.add_argument('--learning_rate', type=float, default=0.0003)
        parser.add_argument('--model_base', choices=["vgg16", "resnet18", "preactresnet"], default="preactresnet")

        return parser

    def train_dataloader(self):
        _train_transforms = None

        if self._hparams['augment']:
            # Data augmentation transforms
            _train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=(36, 60), scale=(0.5, 1.3)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5, saturation=0.5),
                lambda x: x if np.random.random_sample() <= 0.1 else x.filter(ImageFilter.GaussianBlur(radius=3)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Initialize the training dataset
        _data_train = EstimationHdf5DatasetMyDataset(
            h5_file=h5py.File(self._hparams['hdf5_file'], mode="r"),
            subject_list=self._train_subjects,
            transform=_train_transforms
        )

        # Initialize the training data loader
        return DataLoader(_data_train, batch_size=self._hparams['batch_size'], shuffle=True,
                          num_workers=self._hparams['num_io_workers'], pin_memory=True)

    def val_dataloader(self):
        # Initialize the validation dataset
        _data_validate = EstimationHdf5DatasetMyDataset(
            h5_file=h5py.File(self._hparams['hdf5_file'], mode="r"),
            subject_list=self._validate_subjects
        )

        # Initialize the validation data loader
        return DataLoader(_data_validate, batch_size=self._hparams['batch_size'], shuffle=False,
                          num_workers=self._hparams['num_io_workers'], pin_memory=True)

    def test_dataloader(self):
        # Initialize the test dataset
        _data_test = EstimationHdf5DatasetMyDataset(
            h5_file=h5py.File(self._hparams['hdf5_file'], mode="r"),
            subject_list=self._test_subjects
        )

        # Initialize the test data loader
        return DataLoader(_data_test, batch_size=self._hparams['batch_size'], shuffle=False,
                          num_workers=self._hparams['num_io_workers'], pin_memory=True)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    # Get the root directory where this script is located
    root_dir = os.path.dirname(os.path.realpath(__file__))

    # Define command-line arguments
    _root_parser = ArgumentParser(add_help=False)
    _root_parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='gpu',
                              help='gpu to use, can be repeated for multiple gpus i.e. --gpu 1 --gpu 2')
    _root_parser.add_argument('--hdf5_file', type=str,
                              default="dataset/hh/KaAI_dataset_1.hdf5")  # Specify the HDF5 dataset file path
    _root_parser.add_argument('--dataset', type=str, choices=["KaAI", "other"], default="KaAI")
    _root_parser.add_argument('--save_dir', type=str, default='gaze_estimation/checkpoints/fold08')  # Specify the directory to save checkpoints
    _root_parser.add_argument('--benchmark', action='store_true', dest="benchmark")
    _root_parser.add_argument('--no_benchmark', action='store_false', dest="benchmark")
    _root_parser.add_argument('--num_io_workers', default=2, type=int)
    _root_parser.add_argument('--k_fold_validation', default=True, type=bool)
    _root_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    _root_parser.add_argument('--seed', type=int, default=0)
    _root_parser.add_argument('--min_epochs', type=int, default=1, help="Number of Epochs to perform at a minimum")
    _root_parser.add_argument('--max_epochs', type=int, default=29,
                              help="Maximum number of epochs to perform; the trainer will exit after.")
    _root_parser.add_argument('--checkpoint', type=list, default=[])
    _root_parser.set_defaults(benchmark=False)
    _root_parser.set_defaults(augment=True)

    # Parse the command-line arguments
    _model_parser = TrainEstimationModel.add_model_specific_args(_root_parser)
    _hyperparams = _model_parser.parse_args()
    _hyperparams = vars(_hyperparams)

    # Set a random seed for reproducibility
    pl.seed_everything(_hyperparams['seed'])

    # Define the list of training, validation, and test subjects based on the dataset choice
    _train_subjects = []
    _valid_subjects = []
    _test_subjects = []
    if _hyperparams['dataset'] == "KaAI":
        if _hyperparams['k_fold_validation']:
            # Define k-fold cross-validation splits for KaAI dataset
            _train_subjects.append([1, 2, 8, 10, 3, 4, 7, 9])
            _train_subjects.append([1, 2, 8, 10, 5, 6, 11, 12, 13])
            _train_subjects.append([3, 4, 7, 9, 5, 6, 11, 12, 13])

            # Validation set is always subjects 14, 15, and 16
            _valid_subjects.append([0, 14, 15, 16])
            _valid_subjects.append([0, 14, 15, 16])
            _valid_subjects.append([0, 14, 15, 16])

            # Test subjects
            _test_subjects.append([5, 6, 11, 12, 13])
            _test_subjects.append([3, 4, 7, 9])
            _test_subjects.append([1, 2, 8, 10])
        else:
            # Define non-cross-validated splits for KaAI dataset
            _train_subjects.append([1, 2, 5, 6, 7, 8, 10, 12, 13, 14, 16])
            _train_subjects.append([1, 3, 4, 7, 10])
            _valid_subjects.append([7, 9, 12, 15])
            _test_subjects.append([4, 14])
    else:
        file = h5py.File(_hyperparams['hdf5_file'], mode="r")
        keys = [int(subject[1:]) for subject in list(file.keys())]
        file.close()
        if _hyperparams['k_fold_validation']:
            all_subjects = range(len(keys))
            for leave_one_out_idx in all_subjects:
                _train_subjects.append(all_subjects[:leave_one_out_idx] + all_subjects[leave_one_out_idx + 1:])
                _valid_subjects.append([leave_one_out_idx])
                _test_subjects.append([leave_one_out_idx])
        else:
            _train_subjects.append(keys[1:])
            _valid_subjects.append([keys[0]])
            _test_subjects.append([keys[0]])

    # Iterate through the folds and train the model
    for fold, (train_s, valid_s, test_s) in enumerate(zip(_train_subjects, _valid_subjects, _test_subjects)):
        # Define the complete path to save checkpoints for this fold
        complete_path = os.path.abspath(os.path.join(_hyperparams['save_dir'], f"fold_{fold}/"))

        # Initialize the training model
        _model = TrainEstimationModel(hparams=_hyperparams,
                                      train_subjects=train_s,
                                      validate_subjects=valid_s,
                                      test_subjects=test_s)

        # Define the custom model checkpoint callback
        checkpoint_callback = CustomModelCheckpoint(dirpath=complete_path)

        # Define the TensorBoard logger
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir='gaze_estimation/logs/0823', name='gaze_direction_estimation_logs')

        # Define the PyTorch Lightning Trainer
        trainer = Trainer(accelerator=_hyperparams['accelerator'],
                          precision=32,
                          callbacks=[checkpoint_callback],
                          min_epochs=_hyperparams['min_epochs'],
                          max_epochs=_hyperparams['max_epochs'],
                          accumulate_grad_batches=_hyperparams['accumulate_grad_batches'],
                          benchmark=_hyperparams['benchmark'],
                          logger=logger)

        # Start training
        trainer.fit(_model)

        # Run testing
        trainer.test()
