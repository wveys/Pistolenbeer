import cv2
import time
import numpy as np

# Load the reference image (glass image)
template = cv2.imread('segmented_glass.png')
# Ensure the image is in BGR format
if template.shape[2] == 4:  # Image with alpha channel (4 channels)
    template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)

if template is None:
    print("Error: Could not load template image.")
    exit()

# Open the default camera (usually the first camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame + safe it's time to have the start time
    ret, frame = cap.read()
    curr_time = time.time()

    # If the frame is read correctly, ret is True
    if not ret:
        #message of camera disabled
        break

    # Get the dimensions of the template
    w, h = template.shape[:2]

    # Perform template matching
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)

    # TODO: adjust (trial and error) -> how related does the object has to be with the glass
    threshold = 0.5

    # Find locations in the result map where the match is above the threshold
    locations = np.where(result >= threshold)

    # Draw rectangles around the matched regions
    for pt in zip(*locations[::-1]):
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # Show the result
    cv2.imshow('Detected Glasses', frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()