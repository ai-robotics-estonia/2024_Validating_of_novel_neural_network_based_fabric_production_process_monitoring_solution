# Object Detection

## Project Structure

- The **camera** folder contains the code for the BFS-U3 mono camera. This script operates two cameras and manages settings such as `gamma`, `gain`, `exposure`, `pixel format`, `black level clamping`, `sensor shutter mode`,`width`, `height`, and `offset-X/Y`. It saves three images to a specific folder, which are then overwritten during subsequent captures.

- The **detection** folder contains the object detection code for identifying broken products. To improve accuracy, the code compares the X-coordinates of detected defects. If a defect consistently falls within a specific ± range across multiple frames, the product is confirmed as broken. Each machine logs data to a CSV file.  
  **Note:** A trained PyTorch model tailored to your specific detection needs is required.


## Usage
It is recommended to use Python's Virtual Environment for isolated dependency management.

1. **Install `PyTorch`**  
   Follow the official instructions to install [PyTorch](https://pytorch.org/get-started/locally), requirements depend on your operating system and CUDA compatibility.

2. **Install the `ultralytics` package from PyPI**  
   Run the following command:
   ` pip install ultralytics`

3. **Install `Spinnaker` software for the camera**  
   Download the [Teledyne Spinnaker SDK](https://www.teledynevisionsolutions.com/support/support-center/software-firmware-downloads/iis/spinnaker-sdk-download/spinnaker-sdk--download-files/?pn=Spinnaker+SDK&vn=Spinnaker+SDK) and select the appropriate version for your machine.