"""Simple script to test the installation"""

from telekinesis.medulla.cameras import webcam

camera = webcam.Webcam(name="my_webcam", camera_id=0)
camera.connect()
image = camera.capture_single_color_frame()
print(image)
camera.disconnect()
