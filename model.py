import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GazeEstimationModel_fc(nn.Module):
    """A gaze estimation model using fully connected layers."""
    def __init__(self):
        super(GazeEstimationModel_fc, self).__init__()
        self.flatten = nn.Flatten(1)  # Flatten the input for fully connected layers

    @staticmethod
    def _fc_layers(in_features, out_features):
        # Define fully connected layers for left and right eye inputs
        x_l = nn.Sequential(
            nn.Linear(in_features, 1024),  # Fully connected layer with 1024 output units
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )

        x_r = nn.Sequential(
            nn.Linear(in_features, 1024),  # Fully connected layer with 1024 output units
            nn.BatchNorm1d(1024, momentum=0.99, eps=1e-3),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )

        # Concatenate the left and right eye features
        concat = nn.Sequential(
            nn.Linear(2048, 512),  # Fully connected layer with 512 output units
            nn.BatchNorm1d(512, momentum=0.99, eps=1e-3),  # Batch normalization
            nn.ReLU(inplace=True)  # ReLU activation
        )

        # Final fully connected layers for gaze estimation
        fc = nn.Sequential(
            nn.Linear(514, 256),  # Fully connected layer with 256 output units
            nn.ReLU(inplace=True),  # ReLU activation
            nn.Linear(256, out_features)  # Fully connected layer with 'out_features' output units
        )

        return x_l, x_r, concat, fc

    def forward(self, left_eye, right_eye, head_pose):
        left_x = self.left_features(left_eye)  # Extract features from left eye
        left_x = self.flatten(left_x)  # Flatten the features
        left_x = self.xl(left_x)  # Apply the fully connected layers for the left eye

        right_x = self.right_features(right_eye)  # Extract features from right eye
        right_x = self.flatten(right_x)  # Flatten the features
        right_x = self.xr(right_x)  # Apply the fully connected layers for the right eye

        eyes_x = torch.cat((left_x, right_x), dim=1)  # Concatenate left and right eye features
        eyes_x = self.concat(eyes_x)  # Apply the fully connected layers for the concatenated features

        eyes_headpose = torch.cat((eyes_x, head_pose), dim=1)  # Concatenate with head pose information

        fc_output = self.fc(eyes_headpose)  # Final gaze estimation
        return fc_output

    @staticmethod
    def _init_weights(modules):
        for md in modules:
            if isinstance(md, nn.Linear):
                nn.init.kaiming_uniform_(md.weight, mode="fan_in", nonlinearity="relu")  # Initialize weights
                nn.init.zeros_(md.bias)  # Initialize bias

class GazeEstimationModel_vgg16(GazeEstimationModel_fc):
    """A gaze estimation model using VGG16 as the base architecture."""
    def __init__(self, num_out=2):
        super(GazeEstimationModel_vgg16, self).__init__()
        _left_eye_model = models.vgg16(pretrained=True)  # Load pre-trained VGG16 model
        _right_eye_model = models.vgg16(pretrained=True)  # Load another pre-trained VGG16 model

        # Extract features from VGG16 for left and right eyes
        _left_modules = [module for module in _left_eye_model.features]
        _left_modules.append(_left_eye_model.avgpool)
        self.left_features = nn.Sequential(*_left_modules)

        _right_modules = [module for module in _right_eye_model.features]
        _right_modules.append(_right_eye_model.avgpool)
        self.right_features = nn.Sequential(*_right_modules)

        # Allow gradients for feature extraction layers
        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        # Define fully connected layers for gaze estimation
        self.xl, self.xr, self.concat, self.fc = GazeEstimationModel_fc._fc_layers(
            in_features=_left_eye_model.classifier[0].in_features,
            out_features=num_out
        )
        GazeEstimationModel_fc._init_weights(self.modules())  # Initialize weights

class GazeEstimationModel_resnet18(GazeEstimationModel_fc):
    """A gaze estimation model using ResNet18 as the base architecture."""
    def __init__(self, num_out=2):
        super(GazeEstimationModel_resnet18, self).__init__()
        _left_eye_model = models.resnet18(pretrained=True)  # Load pre-trained ResNet18 model
        _right_eye_model = models.resnet18(pretrained=True)  # Load another pre-trained ResNet18 model

        # Extract features from ResNet18 for left and right eyes
        self.left_features = nn.Sequential(
            _left_eye_model.conv1,
            _left_eye_model.bn1,
            _left_eye_model.relu,
            _left_eye_model.maxpool,
            _left_eye_model.layer1,
            _left_eye_model.layer2,
            _left_eye_model.layer3,
            _left_eye_model.layer4,
            _left_eye_model.avgpool
        )

        self.right_features = nn.Sequential(
            _right_eye_model.conv1,
            _right_eye_model.bn1,
            _right_eye_model.relu,
            _right_eye_model.maxpool,
            _right_eye_model.layer1,
            _right_eye_model.layer2,
            _right_eye_model.layer3,
            _right_eye_model.layer4,
            _right_eye_model.avgpool
        )

        # Allow gradients for feature extraction layers
        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        # Define fully connected layers for gaze estimation
        self.xl, self.xr, self.concat, self.fc = GazeEstimationModel_fc._fc_layers(
            in_features=_left_eye_model.fc.in_features,
            out_features=num_out
        )
        GazeEstimationModel_fc._init_weights(self.modules())  # Initialize weights

class GazeEstimationPreactResnet(GazeEstimationModel_fc):
    """A gaze estimation model using a pre-activation ResNet."""
    class PreactResnet(nn.Module):
        """A pre-activation ResNet architecture."""
        class BasicBlock(nn.Module):
            """A basic block for the pre-activation ResNet."""
            def __init__(self, in_channels, out_channels, stride):
                super().__init__()

                self.bn1 = nn.BatchNorm2d(in_channels)
                self.conv1 = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
                self.shortcut = nn.Sequential()
                if in_channels != out_channels:
                    self.shortcut.add_module(
                        'conv',
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  bias=False)
                    )

            def forward(self, x):
                x = F.relu(self.bn1(x), inplace=True)
                y = self.conv1(x)
                y = F.relu(self.bn2(y), inplace=True)
                y = self.conv2(y)
                y += self.shortcut(x)
                return y

        def __init__(self, depth=30, base_channels=16, input_shape=(1, 3, 224, 224)):
            super().__init__()

            n_blocks_per_stage = (depth - 2) // 6
            n_channels = [base_channels, base_channels * 2, base_channels * 4]

            self.conv = nn.Conv2d(input_shape[1],
                                  n_channels[0],
                                  kernel_size=(3, 3),
                                  stride=1,
                                  padding=1,
                                  bias=False)

            self.stage1 = self._make_stage(n_channels[0],
                                           n_channels[0],
                                           n_blocks_per_stage,
                                           GazeEstimationPreactResnet.PreactResnet.BasicBlock,
                                           stride=1)
            self.stage2 = self._make_stage(n_channels[0],
                                           n_channels[1],
                                           n_blocks_per_stage,
                                           GazeEstimationPreactResnet.PreactResnet.BasicBlock,
                                           stride=2)
            self.stage3 = self._make_stage(n_channels[1],
                                           n_channels[2],
                                           n_blocks_per_stage,
                                           GazeEstimationPreactResnet.PreactResnet.BasicBlock,
                                           stride=2)
            self.bn = nn.BatchNorm2d(n_channels[2])

            self._init_weights(self.modules())

        @staticmethod
        def _init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')  # Initialize weights
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        @staticmethod
        def _make_stage(in_channels, out_channels, n_blocks, block, stride):
            stage = nn.Sequential()
            for index in range(n_blocks):
                block_name = "block{}".format(index + 1)
                if index == 0:
                    stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
                else:
                    stage.add_module(block_name, block(out_channels, out_channels, stride=1))
            return stage

        def forward(self, x):
            x = self.conv(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = F.relu(self.bn(x), inplace=True)
            x = F.adaptive_avg_pool2d(x, output_size=1)
            return x

    def __init__(self, num_out=2):
        super(GazeEstimationPreactResnet, self).__init__()

        # Create pre-activation ResNet models for left and right eyes
        self.left_features = GazeEstimationPreactResnet.PreactResnet()
        self.right_features = GazeEstimationPreactResnet.PreactResnet()

        # Allow gradients for feature extraction layers
        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        # Define fully connected layers for gaze estimation
        self.xl, self.xr, self.concat, self.fc = GazeEstimationModel_fc._fc_layers(
            in_features=64,  # Assuming output channels from pre-activation ResNet
            out_features=num_out
        )
        GazeEstimationModel_fc._init_weights(self.modules())  # Initialize weights
