# Description: Object detection class to process images and detect broken products.
# Author: Gregor Kokk
# Date: 06.01.2025

import cv2
from time import time, sleep
import csv
import os
from datetime import datetime
import threading
import signal
import torch

# LED and buzzer import - user can will be able to get feedback about the status of the system
from gpio_utils import GPIOHandler

# Object detection import
from ultralytics import YOLO

# GPIO pin numbers for LEDs and buzzer
blue_led_pin = 16
red_led_w_buzzer_pin = 18

class ObjectDetection:

    def __init__(self, input_folder, output_folder):
        '''
        Initialize the object detection class

        Args:
            input_folder (str): Path to the folder containing the images
            output_folder (str): Path to the folder where the output images will be saved
        '''

        self.input_folder = input_folder
        
        # Create a timestamped output folder
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_folder = os.path.join(output_folder, self.timestamp)
        self.output_folder_broken = os.path.join(self.output_folder, "broken")
        self.output_folder_good = os.path.join(self.output_folder, "good")

        # Create output folders if they don't exist
        os.makedirs(self.output_folder_broken, exist_ok=True)
        os.makedirs(self.output_folder_good, exist_ok=True)
        print(f"Output folders created at {self.output_folder}")

        self.gpio_handler = GPIOHandler(blue_led_pin, red_led_w_buzzer_pin)
       
        self.no_detection_count = 0     # Initialize the counter for frames with no detections

        self.exit_flag = True           # Flag to control the main loop

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        '''Load the model (TensorRT or PyTorch based on device availability)'''
        try:
            # Load the model
            self.model = self.load_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the object detection model: {str(e)}")

        # Access class names from model
        self.name_dict = {0: "broken_mop"}

        # Initialize detection history
        self.history = {}
        self.consecutive_frame_detections = 3   # Number of consecutive detections to confirm an object
        self.x_threshold = 60                  # Threshold for x-axis movement
        self.min_confidence = 0.63              # Minimum confidence for detection

        # Create lists to store X-coordinates for each product line - serial /camera number/ image name / timestamp
        self.product_lines = {
            "Camera_0": [],
            "Camera_1": [],
            "Camera_2": [],
            "Camera_3": []
        }
        
        # Track CSV file paths for each product line
        self.product_line_csv_files = {}

        # Define a mapping from product line keys to machine names
        self.machine_mapping = {
            "Camera_0": "Machine 1",
            "Camera_1": "Machine 2",
            "Camera_2": "Machine 3",
            "Camera_3": "Machine 4"
        }

        print("INITIALIZATION COMPLETE!\n")
        
    def get_product_line_key(self, filename):
        """
        Extract the product line key from the filename.

        Args:
            filename (str): The filename of the image
        
        Returns:
            str: The product line key extracted from the filename
        """

        parts = os.path.splitext(filename)[0].split('_')    # Remove the file extension and split by underscore
        if len(parts) >= 5:
            return f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
        return "Unknown"
    
    def initialize_csv_for_product_line(self, product_line_key):
        """
        Initialize a CSV file for each product line.

        Args:
            product_line_key (str): Key for the product line
        
        Returns:
            None
        """

        if product_line_key not in self.product_line_csv_files:
            csv_path = os.path.join(self.output_folder, f"{product_line_key}_log.csv")
            self.product_line_csv_files[product_line_key] = csv_path
            os.makedirs(self.output_folder, exist_ok=True)  # Ensure the output folder exists
            print(f"Initializing CSV at: {csv_path}")
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Image/Product/Camera Name", "Prediction Time", "Class", "Confidence", "Timestamp"])
            print(f"CSV initialized for product line: {product_line_key}")

    def write_to_product_line_csv(self, product_line_key, row):
        """
        Write a detection log to the CSV file for the given product line

        Args:
            product_line_key (str): Key for the product line
            row (list): List of values to write to the CSV

        Returns:
            None
        """

        csv_path = self.product_line_csv_files.get(product_line_key)
        if csv_path:
            print(f"Writing to CSV: {csv_path}")
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)
                print(f"Row written: {row}")
        else:
            print(f"CSV file not found for product line: {product_line_key}")
    
    def buzzer_timer(self, duration):
        self.gpio_handler.red_led_buzzer_on()  # Turn on Red LED and buzzer
        sleep(duration)  # Wait for the duration
        self.gpio_handler.red_led_buzzer_off()  # Turn off Red LED and buzzer


    def load_model(self):
        """
        Load a YOLO model for TensorRT inference or PyTorch.
        
        Returns:
            model: Loaded YOLO model object.
        """

        try:
            if self.device == "cuda":
                print("Loading TensorRT model.")
                model = YOLO("/path/to/tensor/model", task="detect")  # .engine TensorRT model
            if self.device == 'cpu':
                print("CUDA is not available or using PyTorch model. Loading PyTorch model.")
                model = YOLO("/path/to/pytorch/model", task="detect")  # .pt PyTorch model
            
            print("{model} loaded successfully.")
            return model
        
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError("Failed to load the model. Check the file path and compatibility.") from e


    def predict(self, image_path):
        """
        Predict objects in an image using the TensorRT model.

        Args:
            image_path (str): Path to the input image.

        Returns:
            results (object): Prediction results from the model.
            image (ndarray): The loaded image (BGR format).
            prediction_time (float): Time taken for prediction in seconds.
        """

        start_time = time()  # Start timer

        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")

            print(f"Image loaded successfully from {image_path}")

            # Run prediction -> these settings depend on the model
            results = self.model.predict(
                source=image,
                conf=0.60,              # Confidence threshold
                iou=0.4,                # Lower IoU to allow overlapping detections
                imgsz=(352, 1216),      # Image size for inference
                half=True,              # Use FP16, False for FP16
                device=self.device,     # Device to use for inference
                max_det=100,            # Maximum number of detections
                augment=False,          # Disable augmentation for consistent results
                classes=[0],            # Only detect the class with index 0 (broken mop)
                retina_masks=True,      # Use RetinaNet masks for better detection
            )

            # Calculate prediction time
            prediction_time = time() - start_time
            prediction_time_rounded = round(prediction_time, 2) # Round to 2 decimal places, numeric value

            return results, image, prediction_time_rounded

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None, None


    def plot_bboxes(self, results, frame, image_path, prediction_time_rounded):
        """
        Plot bounding boxes on the image and handle detection results.

        Args:
            results (object): Detection results from the model.
            frame (ndarray): Image frame to plot bounding boxes.
            image_path (str): Path to the input image.
            prediction_time_rounded (float): Rounded prediction time in seconds.

        Returns:
            frame (ndarray): Image frame with bounding boxes.
        """

        try:
            print(f"Processing image: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"Image path does not exist: {image_path}")
                return frame

            # Turn off all LEDs and buzzer
            self.gpio_handler.deinitialize_gpio()

            product_line_key = self.get_product_line_key(os.path.basename(image_path))
            print(f"Extracted product line key: {product_line_key}")
            print(f"Known product lines: {list(self.product_lines.keys())}")

            if product_line_key not in self.product_lines:
                print(f"Unknown product line: {product_line_key}. Skipping.")
                return frame


            self.initialize_csv_for_product_line(product_line_key)
            detection_found = False
            current_detections = []  # Store current detections for this product line

            # Process detection results
            if results[0].boxes:
                for result in results[0].boxes:
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                    confidence = float(result.conf[0])
                    detection_class = self.name_dict[int(result.cls[0])]
                    print(f"Detections found for {product_line_key}, processing...")

                    if detection_class == "broken_mop" and confidence >= self.min_confidence:
                        detection_found = True
                        x_center = (x1 + x2) // 2
                        current_detections.append((x1, y1, x2, y2, x_center, confidence))
                        print("Current detections:", current_detections)
                
            if not detection_found:
                self.handle_no_detection(frame, image_path, product_line_key, prediction_time_rounded)
                return frame

            if detection_found:
                # Update product line history with current detections
                self.update_history(product_line_key, current_detections)

            # Check if we can confirm a broken product
            if self.is_broken_mop_confirmed(product_line_key):
                self.trigger_alert(frame, image_path, product_line_key, prediction_time_rounded)
        
        except Exception as e:
            print(f"Error during classification: {str(e)}")

            return frame


    def update_history(self, product_line_key, detections):
        """
        Update the detection history for a specific product line.

        Args:
            product_line_key (str): Key for the product line.
            detections (list): List of detections for the current frame.
        
        Returns:
            None
        """

        if product_line_key in self.product_lines:
            self.product_lines[product_line_key].extend(detections)
            if len(self.product_lines[product_line_key]) > self.consecutive_frame_detections:
                self.product_lines[product_line_key] = self.product_lines[product_line_key][-self.consecutive_frame_detections:]


    def is_broken_mop_confirmed(self, product_line_key):
        """
        Check if the same broken mop is detected consistently for a product line.

        Args:
            product_line_key (str): Key for the product line.
        
        Returns:
            bool: True if the broken mop is confirmed, False otherwise.
        """
        history = self.product_lines.get(product_line_key, [])
        print(f"Current detection history for {product_line_key}: {history}")

        if len(history) < self.consecutive_frame_detections:
            print(f"Not enough detections for {product_line_key} to confirm consistency.")
            return False

        consistent_detections = []
        for i in range(len(history) - 1):
            box1 = history[i]
            box2 = history[i + 1]

            if abs(box1[4] - box2[4]) <= self.x_threshold:
                consistent_detections.extend([box1, box2])
            else:
                print(f"X-axis mismatch for {product_line_key}. Breaking consistency check.")
                break

        if len(consistent_detections) >= 2:  # At least two consistent detections
            self.last_confirmed_boxes = consistent_detections[:2]
            return True

        print(f"No consistent detections for {product_line_key}. Retaining detection history.")
        return False
    
    def resize_frame_to_screen(self, frame, screen_width, screen_height):
        """
        Resize the frame to fit within the given screen dimensions.

        Args:
            frame (ndarray): Input image frame.
            screen_width (int): Width of the screen.
            screen_height (int): Height of the screen.
        
        Returns:
            frame (ndarray): Resized image frame.
        """

        frame_height, frame_width = frame.shape[:2]
        scaling_factor = min(screen_width / frame_width, screen_height / frame_height)
        new_width = int(frame_width * scaling_factor)
        new_height = int(frame_height * scaling_factor)
        return cv2.resize(frame, (new_width, new_height))


    def trigger_alert(self, frame, image_path, product_line_key, prediction_time_rounded):
        """
        Display confirmed broken product with bounding boxes for 3 seconds.

        Args:
            frame (ndarray): Input image frame.
            image_path (str): Path to the input image.
            product_line_key (str): Key for the product line.
            prediction_time_rounded (float): Rounded prediction time in seconds.
        
        Returns:
            None
        """
        
        # Get the machine name from the mapping
        machine_name = self.machine_mapping.get(product_line_key, "Unknown Machine")
        print(f"Broken mop detected for {machine_name}.")

        # Turn off blue LED
        self.gpio_handler.blue_led_off()

        # Start a thread to trigger the buzzer for 3 seconds
        threading.Thread(target=self.buzzer_timer, args=(1,), daemon=True).start()

        # Draw bounding boxes and labels for detections
        for box in self.last_confirmed_boxes:
            x1, y1, x2, y2, _, confidence = box
            color = (0, 0, 255)  # Red color for broken product
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"broken_mop: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add the machine name to the top of the image
        cv2.putText(
            frame,
            machine_name,
            (10, 30),  # Position at the top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # Font size
            (0, 0, 255),  # Also Red color for text
            2  # Thickness of the text
        )

        # Generate a unique filename for the saved image
        timestamp = datetime.now().strftime("%H-%M-%S")
        save_filename = f"{machine_name.replace(' ', '').lower()}_{timestamp}.jpg"
        save_path = os.path.join(self.output_folder_broken, save_filename)

        # Save the image
        cv2.imwrite(save_path, frame)
        print(f"Broken image saved as {save_path}")

        # Resize the frame to fit the screen
        resized_frame = self.resize_frame_to_screen(frame, screen_width=800, screen_height=600)

        # Display the resized frame
        cv2.imshow("Broken Product Found", resized_frame)
        cv2.waitKey(3000)  # Display for 3 seconds
        cv2.destroyAllWindows()

        # Log data to CSV
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for box in self.last_confirmed_boxes:
            x1, y1, x2, y2, _, confidence = box
            self.initialize_csv_for_product_line(product_line_key)  # Ensure CSV is initialized
            self.write_to_product_line_csv(product_line_key, [
                os.path.basename(image_path),
                prediction_time_rounded,
                "broken_mop",
                confidence,
                current_time
            ])

    def handle_no_detection(self, frame, image_path, product_line_key, prediction_time_rounded):
        """
        Handle non-detection cases for the product line.

        Args:
            frame (ndarray): Input image frame.
            image_path (str): Path to the input image.
            product_line_key (str): Key for the product line.
            prediction_time_rounded (float): Rounded prediction time in seconds.

        Returns:
            None
        """

        # Show alert for no detection
        self.gpio_handler.red_led_buzzer_off() # Turn off red LED and buzzer
        self.gpio_handler.blue_led_on() # Turn on blue LED
    
        # Get the machine name from the mapping
        machine_name = self.machine_mapping.get(product_line_key, "Unknown Machine")

        self.no_detection_count += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.no_detection_count % 10 == 0: # Save an image every 10 frames with no detection
            timestamp = datetime.now().strftime("%H-%M-%S")
            save_filename = f"{machine_name.replace(' ', '').lower()}_{timestamp}.jpg"
            save_path = os.path.join(self.output_folder_good, save_filename)
            cv2.imwrite(save_path, frame)
            print(f"Saved image with no detection as {save_path}")

        # Write non-detection info to CSV        
        self.initialize_csv_for_product_line(product_line_key)  # Ensure CSV is initialized
        self.write_to_product_line_csv(product_line_key, [
            os.path.basename(image_path),
            prediction_time_rounded,
            "no_detection",
            "N/A",
            current_time,
            "N/A",
        ])

    def signal_handler(self, sig, frame):
        """
        Signal handler to handle keyboard interrupts.

        Args:
            sig (int): Signal number
            frame (object): Frame object
        Returns:
            None
        """
        
        print("Signal handler called with signal")
        
        self.exit_flag = False

    def process_images(self):
        """
        Process images in the input folder and detect broken products.

        Args:
            None

        Returns:
            None
        """
        
        try:
            # Dictionary to track processed files and their modification times
            processed_files = {}

            signal.signal(signal.SIGINT, self.signal_handler)   # Bind the signal handler

            while self.exit_flag:
                # Get a list of all images in the folder
                image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith('.jpg')]

                for image in image_files:
                    image_path = os.path.join(self.input_folder, image)
                    mod_time = os.path.getmtime(image_path)  # Get last modification time

                    # Check if the image is new or has been updated
                    if image_path in processed_files and processed_files[image_path] == mod_time:
                        print (f"Skipping already processed file: {image_path}")
                        continue  # Skip already processed files

                    # Process the new or updated image
                    results, frame, prediction_time_rounded = self.predict(image_path)
                    if frame is not None:
                        print(f"Calling plot_bboxes for {image_path}")
                        frame = self.plot_bboxes(results, frame, image_path, prediction_time_rounded)

                    # Mark the image as processed by storing its modification time
                    processed_files[image_path] = mod_time
                    print(f"Processed file: {image_path}, modification time: {mod_time}")
                
                    print(f"Current product line coordinates: {self.product_lines}")
                
                # Wait for a short time before checking for new images again
                sleep(0.3)

        except KeyboardInterrupt:
            print("KeyboardInterrupt: Exiting program.")
        finally:
            cv2.destroyAllWindows()
            self.gpio_handler.delete_gpio()  # Clean up GPIO resources
            print("Program terminated and resources released.")

    def __call__(self):
        """
        Start the object detection process.
        """

        print("Starting object detection...")
        self.process_images()