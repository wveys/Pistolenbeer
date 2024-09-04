import cv2
import numpy as np

# Global variables for the rectangle
rect_start = None
rect_end = None
drawing = False


def mouse_callback(event, x, y, flags, param):
    global rect_start, rect_end, drawing, image_with_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        # When the left mouse button is clicked, record the starting point
        rect_start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # Update the rectangle size while the mouse is moving
        if drawing:
            image_with_rect = image.copy()  # Copy the image to draw on
            rect_end = (x, y)
            cv2.rectangle(image_with_rect, rect_start, rect_end, (0, 255, 0), 2)
            cv2.imshow('Draw Rectangle', image_with_rect)

    elif event == cv2.EVENT_LBUTTONUP:
        # When the left mouse button is released, finalize the rectangle
        rect_end = (x, y)
        drawing = False
        cv2.rectangle(image_with_rect, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow('Draw Rectangle', image_with_rect)


# Load the image
image = cv2.imread('glass.jpg')
if image is None:
    print("Error: Could not load image.")
    exit()

# Make a copy of the image for drawing
image_with_rect = image.copy()

# Create a window and set the mouse callback function
cv2.namedWindow('Draw Rectangle')
cv2.setMouseCallback('Draw Rectangle', mouse_callback)

# Display the image and wait for user input
cv2.imshow('Draw Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The final rectangle coordinates
if rect_start and rect_end:
    x1, y1 = rect_start
    x2, y2 = rect_end
    rect = (x1, y1, x2 - x1, y2 - y1)
    print(f"Rectangle coordinates: {rect}")

    # Optionally, use the rectangle for further processing (e.g., GrabCut)
    # Initialize the mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Temporary arrays used by the GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask so that 0 and 2 pixels are converted to background (0),
    # and 1 and 3 pixels are converted to foreground (1)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Segment the image using the mask
    segmented_image = image * mask2[:, :, np.newaxis]

    # Show the segmented image
    cv2.imshow('Segmented Glass', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No rectangle was drawn.")
