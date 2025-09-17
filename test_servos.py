import time
from adafruit_servokit import ServoKit
'''
import smbus2

bus = smbus2.SMBus(1)  # Using bus 1 (the default)

# Replace 0x40 with the actual I2C address of your PCA9685
i2c_address = 0x40

# Example: Reading the PWM frequency register
try:
    value = bus.read_byte_data(i2c_address, 0x00)  # Read register 0x00 (PWM frequency register)
    print(f"PWM frequency register value: {value}")
except IOError as e:
    print(f"Error reading from PCA9685: {e}")
'''    


# Set up the kit for 16-channel PCA9685
kit = ServoKit(channels=16, address=0x40)

# Initialize servo angles (you can adjust to center positions)
pan_channel = 0
tilt_channel = 1
pan_angle = 10
tilt_angle = 10
kit.servo[pan_channel].angle = pan_angle
kit.servo[tilt_channel].angle = tilt_angle
time.sleep(1.0);
pan_angle = 180
tilt_angle = 180
kit.servo[pan_channel].angle = pan_angle
kit.servo[tilt_channel].angle = tilt_angle
time.sleep(1.0);

while True:

	pan_angle = 10
	tilt_angle = 10
	kit.servo[pan_channel].angle = pan_angle
	kit.servo[tilt_channel].angle = tilt_angle
	print("angle 10")
	time.sleep(1.0);
	pan_angle = 180
	tilt_angle = 180
	kit.servo[pan_channel].angle = pan_angle
	kit.servo[tilt_channel].angle = tilt_angle
	print("angle 180")
	time.sleep(1.0);





