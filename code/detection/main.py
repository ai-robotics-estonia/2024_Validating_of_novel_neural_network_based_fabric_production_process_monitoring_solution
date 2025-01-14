# Description: Main script to run the object detection
# Author: Gregor Kokk
# Date: 06.01.2025

from object_detection import ObjectDetection

def main():
    input_folder = "/path/to/your/images"  # Path to the folder containing the images
    output_foler = "/path/to/save/detected/images"  # Path to the folder where the output images will be saved

    # Initialize the detector
    detector = ObjectDetection(input_folder, output_foler)
    detector()  # Start the detection process


if __name__ == "__main__":
    main()