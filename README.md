# Driver-Monitoring-Study
A Study on the Estimation and Monitoring of Driver's Eyes for the Realization of Self-Driving Technology

**
(1) Cloud  programming world cup on the topic of developing a distracted driving accident prevention system through driver gaze estimation using deep learning. 
**


We use "pupil" to track the driver's gaze. And we use azure to track the direction of the driver's head. Through this, the driver's gaze and head direction estimate situations in which the driver closes his eyes for a long time or the driver is unable to concentrate on driving. Based on this, the system first alerts the driver. Furthermore, the information is passed over to the driver around them via communication and visualized on the GUI.
![Untitled (9)](https://github.com/leehyunseo921/Driver-Monitoring-Study/assets/153660740/2f372e20-bd08-44ef-8b87-89b6eeed3ba0)



Attached is a description of the PPT technology below.

[CPWC2022-0024_Program_Concept_Future_tasks_and_ideas.pptx](https://github.com/leehyunseo921/Driver-Monitoring-Study/files/13651865/CPWC2022-0024_Program_Concept_Future_tasks_and_ideas.pptx)


**
(2) Capstone project on the topic of drowsy driving and forward warning system through driver face recognition using webcam
**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bd63cb73-2b18-4ae6-a39a-61825b0b238e/Untitled.png)

1. Development Motivation:
Cars, which are essential means of transportation for modern people, carry the risk of traffic accidents along with convenience. In particular, traffic accidents frequently occur due to drowsy driving and lack of attention ahead. As a result, we have developed an economical and accessible system using webcams to prevent accidents caused by carelessness.

2. System Configuration:

Our system uses Kosi's QHD imaging camera to photograph the driver's face head-on.
After face detection, use the shape_predictor_68_face_landmarks model to locate the driver's eyes and calculate the aspect ratio, which detects the closing of the eyes.
The direction of the driver's head is a landmark-based angle calculation that determines the top, bottom, left, and right sides, and ensures that they remain staring straight ahead.

3. Criteria for judging drowsiness and carelessness:

If the blink duration is more than 1.5 seconds, it is determined as drowsy by measuring the time you close your eyes.
If you look in a different direction for more than 5 seconds based on the front gaze through a change in head direction, it is considered careless.

4. Experiment and Utilization:

The experiment used Kosi's QHD image camera, and the experiment was conducted under the assumption of various scenarios while driving.
Image data from the webcam can be extracted in the form of csv files and used for future accident analysis and research.

5. Results and future developments:

This system can prevent inadvertent traffic accidents and increase safety by linking with V2X communications that can report emergencies.
In the future, we plan to provide a safer driving environment by building a more sophisticated vehicle-driver interaction system using a deep learning model.

* See headpose_and_eye.py

**(3) Acquisition of autonomous driving datasets through Senso Fusion in autonomous driving artificial intelligence academic clubs **


