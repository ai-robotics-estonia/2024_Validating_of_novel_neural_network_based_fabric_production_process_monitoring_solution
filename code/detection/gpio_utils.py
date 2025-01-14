# Description: Scrip to control the GPIO pins, LEDs and buzzer using the DRV8833 motor driver so that the user can get feedback about the status of the system.
# Note: On/off states are represented as 1/0, might depend on the configuration of the GPIO pins.
# Author: Gregor Kokk
# Date: 06.01.2025

import Jetson.GPIO as GPIO
from drv8833 import DRV8833

class GPIOHandler:
    def __init__(self, blue_pin, red_pin):
        self.blue_pin = blue_pin
        self.red_pin = red_pin
        self.controller = DRV8833(blue_pin, red_pin)
        self.gpio_initialized = False
        self.initialize_gpio()

    def initialize_gpio(self):
        try:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.blue_pin, GPIO.OUT)
            GPIO.setup(self.red_pin, GPIO.OUT)
            self.gpio_initialized = True
        except Exception as e:
            print(f"Error initializing GPIO: {e}")
            self.gpio_initialized = False

    def blue_led_on(self):
        if self.gpio_initialized:
            self.controller.set_led_b(0)  # Turn on
        else:
            print("GPIO not initialized. Skipping blue LED operation.")

    def blue_led_off(self):
        if self.gpio_initialized:
            self.controller.set_led_b(1)  # Turn off
        else:
            print("GPIO not initialized. Skipping blue LED operation.")

    def red_led_buzzer_on(self):
        if self.gpio_initialized:
            self.controller.set_led_a(0)  # Turn on
        else:
            print("GPIO not initialized. Skipping red LED operation.")

    def red_led_buzzer_off(self):
        if self.gpio_initialized:
            self.controller.set_led_a(1)  # Turn off
        else:
            print("GPIO not initialized. Skipping red LED operation.")

    def deinitialize_gpio(self):
        if self.gpio_initialized:
            self.red_led_buzzer_off()
            self.blue_led_off()
            print("GPIO deinitialized -> red LED, buzzer and blue LED off.")
    
    def delete_gpio(self):
        self.controller.deinit()
        print("GPIO cleanup complete.")
