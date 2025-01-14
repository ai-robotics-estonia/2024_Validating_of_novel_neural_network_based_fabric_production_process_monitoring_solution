# Description: Scrip to control the DRV8833 motor driver
# Author: Gregor Kokk
# Date: 06.01.2025

import Jetson.GPIO as GPIO

class DRV8833:
    """
    Helper class to manipulate single-channel outputs like LEDs and buzzer using DRV8833 control structure.
    """

    def __init__(self, a_out: int, b_out: int) -> None:
        """Initialize the DRV8833 class with the GPIO pin numbers."""
        self.led_a = a_out
        self.led_b = b_out
        GPIO.setmode(GPIO.BOARD)  # Use physical pin numbering
        GPIO.setup(self.led_a, GPIO.OUT)  # Initialize LED A as output
        GPIO.setup(self.led_b, GPIO.OUT)  # Initialize LED B as output

    def set_led_a(self, state: bool):
        """Turn LED A on or off."""
        GPIO.output(self.led_a, GPIO.HIGH if state else GPIO.LOW)  # Set pin HIGH for ON, LOW for OFF

    def set_led_b(self, state: bool):
        """Turn LED B on or off."""
        GPIO.output(self.led_b, GPIO.HIGH if state else GPIO.LOW)  # Set pin HIGH for ON, LOW for OFF

    def deinit(self):
        """Deinitialize GPIO outputs for LEDs."""
        GPIO.output(self.led_a, GPIO.LOW)  # Turn off LED A
        GPIO.output(self.led_b, GPIO.LOW)  # Turn off LED B
        GPIO.cleanup() # Cleanup all GPIO settings