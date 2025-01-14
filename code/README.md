# Object Detection

## Project Structure
- The `camera` folder contains the BFS-U3 mono camera code. This script operates two cameras and saves three images to a specific folder.

- The `detection` folder contains the object detection code. You will need a trained PyTorch model tailored to your specific detection needs.

## Usage
It is recommended to use Python's Virtual Environment for isolated dependency management.

1. **Install PyTorch**  
   Requirements depend on your operating system and CUDA compatibility. Follow the official instructions to install PyTorch from [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally).

2. **Install the `ultralytics` package from PyPI**  
   Run the following command:
   ```bash
   pip install ultralytics

3. **Install `Spinnaker` software for the camera**
   Select the appropriate version for your machine. You can download the software from [https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK).