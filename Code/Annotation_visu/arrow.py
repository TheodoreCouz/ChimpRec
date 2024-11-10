import cv2
import numpy as np
import matplotlib.pyplot as plt

def paste_transparent_object(background_path, object_path, position=(0, 0), downsampling_factor=1):
    # Load the background and the transparent object image
    background = cv2.imread(background_path)
    transparent_obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)

    # Split the object image into RGB and alpha channels
    obj_rgb = transparent_obj[:, :, :3]
    alpha = transparent_obj[:, :, 3]

    # Apply downsampling factor
    if downsampling_factor > 1:
        new_size = (int(obj_rgb.shape[1] / downsampling_factor), int(obj_rgb.shape[0] / downsampling_factor))
        obj_rgb = cv2.resize(obj_rgb, new_size, interpolation=cv2.INTER_LANCZOS4)
        alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_LANCZOS4)

    # Calculate the region where the object will be pasted on the background
    x, y = position
    h, w = obj_rgb.shape[:2]

    # Adjust dimensions if the overlay goes beyond the background's boundaries
    bg_h, bg_w = background.shape[:2]
    if y + h > bg_h:
        h = bg_h - y
        obj_rgb = obj_rgb[:h, :]
        alpha = alpha[:h, :]
    if x + w > bg_w:
        w = bg_w - x
        obj_rgb = obj_rgb[:, :w]
        alpha = alpha[:, :w]

    # Define the region of interest (ROI) on the background
    roi = background[y:y+h, x:x+w]

    # Normalize alpha channel to range [0, 1]
    alpha = alpha / 255.0

    # Use alpha blending to paste the object onto the ROI of the background
    for c in range(3):
        roi[:, :, c] = (obj_rgb[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)).astype(np.uint8)

    # Place the blended ROI back onto the background
    background[y:y+h, x:x+w] = roi

    return background


def imshow(image, *args, **kwargs):
    if len(image.shape) == 3:
        # Assume BGR, convert to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Convert grayscale to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    plt.imshow(image, *args, **kwargs)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Example usage
    background_path = "3.png"
    object_path = "Untitled.png"  # The transparent object image
    result = paste_transparent_object(background_path, object_path, position=(100, 50), downsampling_factor=10)

    imshow(result)
