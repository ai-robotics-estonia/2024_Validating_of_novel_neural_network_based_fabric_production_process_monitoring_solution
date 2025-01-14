# Reducing Waste by Detecting Defects on Fast-Moving Fabric Knitting Lines with Object Detection

## Summary
| Company Name | [Vikan Estonia AS](https://www.vikan.com/int)|
| :--- | :--- |
| Development Team Lead Name | [Kädi Veeroja](https://www.linkedin.com/in/kädi-veeroja/) |
| Development Team Lead E-mail | kadi.veeroja@taltech.ee |
| Duration of the Demonstration Project | February 2024 - December 2024 |

# Description

## Objectives of the Demonstration Project

The primary objective of the project was to enhance the quality control process by automating defect detection in the knitting process. Specifically, the goal was to detect yarn breaks or non-binding of yarns as early as possible, eliminating the need for operators to constantly monitor machines. This objective was divided into two sub-goals:

### Environmental Goal
To reduce textile waste generated during production by implementing automated quality control. Scrap products cannot be reused or sold, making them a direct waste with significant environmental impact. Since the materials required to produce mops are primarily synthetic or a blend of organic and synthetic materials, recycling them is either highly challenging or economically unfeasible in the long term.

### Economic Goal
To minimize the economic impact of waste materials on the company by addressing multiple factors:

- **Lost work time**: Time spent producing defective products incurs direct costs and delays subsequent production stages.
- **Lost material**: The cost of materials wasted during production.
- **Disposal costs**: Scrap fabric must be disposed of, with disposal fees increasing as waste volumes grow.
- **Competitiveness**: Scrap fabric increases the product's cost price, reducing market competitiveness.


To achieve these goals, a machine vision solution was tested and validated to notify operators of yarn breaks or non-binding of yarns in real time. By shifting defect detection from human operators to cameras, the process became more accurate and efficient. Human operators are prone to errors due to distractions or fatigue, especially when tasked with identifying defects such as missing or knotted threads, some as fine as 0.2 mm in diameter—an almost impossible task for the human eye.

The use of cameras allowed a single operator to oversee multiple knitting machines simultaneously, reducing the workload and labor costs at subsequent production stages. This improvement not only increased the company's competitiveness but also reduced the overall cost of production.

## Activities and Results of the Demonstration Project

### Challenge
The challenge was automating the quality control of a specific knitting line to reduce waste material.  During the planning stage the project solution changed and after consulting with the team, we decided to use the YOLO machine learning algorithm to detect the knitting line defects.  YOLO must be trained with images of the correct mop, then it takes a real-time picture of the knitting line on the machine and compares the result with the images of the correct mop.  

Such a solution was suitable for a project, where the goal is to automate quality control and gave more time to train the algorithm and experiment with different camera setups and LED panel lighting and camera settings. The demo project proved that a quality control solution can be built in this form, but there is no analysable result yet, because the efficiency cannot be calculated now, but further development is already in planning.

### Activities Implemented and Results Achieved
First, data research and examination of equipment and software began. For this, it was necessary to go to the factory to better understand, which quality control solution would be suitable for Vikan, and for the installation of the hardware, measurements were also taken from the knitting machine.

Preliminary work was done for a long time, to select which hardware and software to use in the solution. The project mentors were consulted, who confirmed the hardware options, and after that a list of equipment for the procurement was put together.  
  
Researching software and talking to software mentors led to the YOLO algorithm, which seemed like a practical choice instead of writing a new machine learning algorithm from scratch. Using YOLO gave more time to modify the solution according to Vikan's wishes.

Before software development started, the initial task was formulated, and a block scheme of the solution was made.

Xavier Jetson, a machine learning computer, was the first hardware piece to set up. Ubuntu version 22 was uploaded on Xavier and a 2 TB SSD card was added to the computer board, for better calculation for capacity.  
  
FLIR BlackFly S USB3 Mono cameras were used for detection, and since the software of the camera itself was not compatible with the computer, the Spinnaker SDK was used to better change the camera settings, and it also gave better access to change the camera parameters.  Colour camera was also tested, but it gave better results only on the prototype so in the solution mono cameras are used.  
  
On the hardware side, a prototype was built to simulate the factory conditions, and for this an aluminium stand had to be made, in addition the same 3D printed camera attachment as on the factory machine, had to be made for the prototype. The mount consists of a camera socket and an aluminium frame mount that are bolted together.

Since the mop needs to be lit, it was necessary to make a special LED lighting panel. The panel has an RGB led strip, and the panel is also attached to the frame with 3D printed connectors. All 3D models are made with Autodesk Fusion 360 software.

Next, pictures of the knitting machine were taken in the factory, and with the help of photo geometry, a 3D model of the knitting machine was made, and according to the model, an aluminum frame was modelled, which would fit on top of the factory machine. Using the photo geometry model, the mounts to bolt together the touch display to the electronics box and all of that to the other side of the knitting machine are also 3D printed.

Technologically four codes were written. One code, to create a trackbar to change the parameters of the cameras for changing the picture quality (exposure, gain, gamma). The values are stored in a database. The second code takes pictures of the knitting line, takes the values from the database and creates an image according to the parameters in the first code. A new picture is taken every 3 seconds, a total of 3 pictures are taken, it reduces memory usage and load, so it was a useful choice for us. The third code is the YOLO v8 algorithm to detect the defects and the fourth one controls the LED panel (light intensity and colour).   
 
Detection is done by the Ultralytics Yolo v8 algorithm model, which was trained on a database of mop images with bounding boxes around the defects. At first, the algorithm was tested on a prototype and a new code had to be made in the factory from the beginning and a new model trained, because the environmental conditions were too different for the prototype and the solution in the factory.
  
The trained model was exported on Xavier as a TensorRT model, and this model detects the images that the camera takes. The camera saves 3 images in a folder and the algorithm goes through the images, sorts the images into two folders, a correct and a defect mop. According to this, the blue LED on the side of the electronics box is lit (if the mop is correct) and the red LED and sound signal (if the mop is faulty).   
  
When the frame going to the factory was ready, the new 3D printed camera housing and LED panel were attached to the frame, and after that it was possible to go to Vikan to test the primary solution. 
 
Once the solution was tested, it was necessary to pack all the electronics into one specially modeled 3D box. The box includes a power supply unit (provides appropriate power to all electronic components (computer, LED, screen), Xavier Jetson machine learning computer (which handles detection), ESP32 (to control the LED panel/intermediate link between the LED panel and the computer), two fans (for cooling), a signal module (to indicate with light and sound whether the mop is correct or faulty). The back panel of the box has connectors so that it is possible to connect the wires conveniently.

In addition, the touch screen mounts, and screen jig are also 3D printed, as is the stand for the entire electronics housing box, which is also attached to the knitting machine.

It was necessary to go to the factory several times to train and test the model, and it was also necessary to test the electronics box and other attachments, but even in this respect, the solution was completed on time.

### Data Sources
- [Hugging Face](https://huggingface.co/)
- [Kaggle](https://www.kaggle.com/)
- [Blackfly S USB3](https://www.flir.eu/products/blackfly-s-usb3/?model=BFS-U3-50S5M-C&vertical=machine+vision&segment=iis)
- [ULTRA COMPACT Lens](https://www.kowa-lenses.com/en/lm8jc5mc-5mp-industrial-lens-c-mount-)
- [Machine Vision Lens Calculator](https://www.teledynevisionsolutions.com/en-150/support/support-center/technical-guidance/lens-calculator/)
- [Jetson Xavier NX Developer Kit - Get Started](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit)
- [JETSON-XAVIER-NX-DEV-KIT](https://www.waveshare.com/wiki/JETSON-XAVIER-NX-DEV-KIT)
- [Spinnaker SDK C++ 4.2.0.21](http://softwareservices.flir.com/Spinnaker/latest/index.html)
- [NVIDIA cuSPARSELt](https://docs.nvidia.com/cuda/cusparselt/)
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

### AI Technologies
"YOLO" (You Only Look Once) is a real-time object detection algorithm designed to identify specific objects in videos, live feeds, or images. It leverages a deep convolutional neural network to learn features and accurately detect objects within an image. This functionality can be achieved using libraries like Keras, OpenCV's deep learning module, or open-source Python libraries such as Ultralytics, which specialize in state-of-the-art object detection implementations."

"YOLO" divides the input image into a grid of cells and predicts bounding box and class probabilities for each cell. Each bounding box predicts the location of an object and the probability of its corresponding class. The network then calculates a confidence score for each predicted bounding box, indicating that the object is within the bounding box.

Uses a single neural network for detection and classification. It differs from other object recognition systems in that YOLO uses a single neural network. This approach allows the system good accuracy and speed. This in turn reduces computational requirements and facilitates the integration of our system.

This project used YOLOv8n.

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [Train Yolov8 object detection on a custom dataset](https://www.youtube.com/watch?v=m9fH9OWn8YM)

### Technological Architecture and Results
The solution planning began with configuring the camera to display its view on a screen. Once this was set up, the camera, along with its 3D-printed case, was installed onto the prototype frame. Displaying live images enabled adjustments to optimize exposure, gain, and gamma settings, which were fine-tuned using a trackbar.
 
First testing was on the prototype with red lights. Testing database was made with pictures of the defect mop and correct mop and the code needed to compare all the new pictures to the training dataset. Training was successful as the defect was clearly visible as a red dot on the mop and the training model detected it. Afterwards, different lights were tested, but red LED lights worked best, so the solutions were ready to be field tested for the first time in the factory on the real knitting machine.

Due to the factory's own lights, the red lights on the LED panel could not detect defects properly so the light was switched to white. This fitted with the environment and gave proper pictures to test the solution with. After the training data (new pictures of defects and correct mops) was collected from the knitting machine, a new detection model was trained which gave remarkable results.

There was a lot of back and forth testing on the knitting machine with different camera settings and different lights, but the most optimal solutions were to use FLIR MONO camera and white lights. Those gave better results than color camera and red lights. At first the model only decided if the mop was correct or defective - Image Classification.

Going further, a new training set was made that had different defects in different folders, so the model could display which flaw was being detected. Next step was making bounding boxes on the defect areas for better understanding where the defect was on the mop and displaying the accuracy with which the model made its decisions - Object Detection. This allowed us to modify the model to only respond to a certain accuracy rate.

To determine the optimal settings for the decision model, extensive training data was analyzed. For instance, selecting an optimal recall value involved examining the Recall-Confidence and Precision-Confidence curves to ensure they supported the chosen recall. Currently, the optimal recall is set at 0.6, as both curves validate this choice. The model operating in the factory currently achieves an optimal accuracy rate of 0.6.

The camera was configured to capture three images with optimal settings, save them to a folder for comparison, and then run the model's detection code to identify defects.

### User Interface 
Currently, there is no integration with the knitting machine. However, when the main camera code is executed, it retrieves the desired settings from the database and begins capturing images of the mop. These images are processed by the main algorithm, which detects objects and displays the results on the screen. If a fault is detected, an audible signal and a red light on the electronics box are activated. When the mop is in good condition, a blue light remains illuminated.  
  
An electronics box is attached to the knitting machine. Inside the box, there are indicator lights, a buzzer, a power supply unit, a Xavier Jetson (the object-detection computer), an ESP32, two cooling fans, and a screen that displays the captured images. A small keyboard and mouse are also included for interacting with the computer.

### Future Potential of the Technical Solution
A similar solution can be integrated into other Vikan knitting machines, and with hardware modifications, the quality control system can be adapted for different production lines. The algorithm can be tailored to specific tasks and locations without requiring changes to the camera mounting system, though the hardware frame and mounting solution should be customized for each company and machine.

Currently, the final solution does not operate independently within the company. While there is a database of mop defects, the system cannot yet learn new mop types on its own. Further development from Vikan is needed to create a fully autonomous solution. Ideally, the final version would be remotely controllable, capable of starting and stopping, learning new mop types automatically, saving them to the mop database, and stopping the knitting machine when defects are detected.

This solution can also be applied to other knitting machines, including those at Vikan's parent company, and can be modified to detect errors on smaller production lines as well. Its full potential will be realized with these future enhancements.

### Lessons Learned

The solution addressed part of the challenge. However, due to limited validation time, it has not been actively operational in the factory for long enough to gather sufficient data to determine its environmental and financial benefits for the company. The solution works and detects defects, there are still a few false detections - where the model is classifying a correct mop as flawed, but overall, the solution (software and hardware) is working well, and it can be concluded that optimising quality control by using this kind of machine learning is doable and probably reduces material costs.