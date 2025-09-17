import adafruit_blinka
from adafruit_pca9685 import PCA9685
import time

# Initialize the PCA9685 (replace 0x40 with the I2C address if different)
#i2c = adafruit_blinka.I2C(0)  # Use the correct bus number for your Pi
pwm = PCA9685(0x40)

# Set the PWM frequency (e.g., 50Hz)
pwm.set_pwm_freq(50)

# Example: Move a servo on channel 0
pwm.set_channel(0, 0, 500)  # Set to a low position (e.g., 0 degrees)
time.sleep(2)  # Wait 2 seconds
pwm.set_channel(0, 0, 2500)  # Set to a high position (e.g., 180 degrees)
time.sleep(2)  # Wait 2 seconds
