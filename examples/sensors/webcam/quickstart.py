"""Simple script to test the installation"""

from telekinesis.medulla.cameras import webcam

camera = webcam.Webcam(name="my_webcam")
camera.connect(camera_id=0)
image = camera.capture_image()
print(image)
camera.disconnect()
