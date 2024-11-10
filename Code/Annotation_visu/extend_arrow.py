import cv2
import numpy as np
from arrow import imshow

def adapt_arrow(arrow_path, d, H, W):
    # Load the arrow image with transparency
    arrow = cv2.imread(arrow_path, cv2.IMREAD_UNCHANGED)
    
    # Separate RGB and alpha channels
    arrow_rgb = arrow[:, :, :3]
    arrow_alpha = arrow[:, :, 3]

    # Determine the height and width of the original arrow
    h, w = arrow_rgb.shape[:2]

    # Create a new width for the arrow image with extended horizontal line
    new_width = w + d + W  # W is the width of the new horizontal line segment

    # Create new blank images for the extended arrow with a transparent background
    new_arrow_rgb = np.zeros((h, new_width, 3), dtype=np.uint8)
    new_arrow_alpha = np.zeros((h, new_width), dtype=np.uint8)

    # Place the original arrow RGB and alpha channels in the new images
    new_arrow_rgb[:, :w] = arrow_rgb
    new_arrow_alpha[:, :w] = arrow_alpha

    # Extend the horizontal line in the new arrow
    # Locate the row where the horizontal line of the arrow is located based on alpha
    line_row = np.argmax(np.any(arrow_alpha > 0, axis=1))
    
    # Extend the line to the right with specified height H and width W
    line_start_col = w + d
    line_end_col = line_start_col + W
    line_end_row = min(line_row + H, h)  # Ensure the line height stays within the arrow image height

    # Draw the extended line in the RGB and alpha channels
    new_arrow_rgb[line_row:line_end_row, line_start_col:line_end_col] = arrow_rgb[line_row, w-1]  # Use the same color as the arrow
    new_arrow_alpha[line_row:line_end_row, line_start_col:line_end_col] = arrow_alpha[line_row, w-1]  # Same transparency

    # Combine RGB and alpha channels
    extended_arrow = np.dstack([new_arrow_rgb, new_arrow_alpha])

    return extended_arrow

def paste_transparent_object(background_path, transparent_obj, position=(0, 0), downsampling_factor=1):
    # Load the background and the transparent object image
    background = cv2.imread(background_path)
    # transparent_obj = cv2.imread(object_path, cv2.IMREAD_UNCHANGED)

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

# Example usage
arrow_path = "Untitled.png"  # Path to the arrow image
extended_arrow = adapt_arrow(arrow_path, 500, 12, 12)
imshow(extended_arrow)

result = paste_transparent_object("white.png", extended_arrow, position=(100, 50), downsampling_factor=10)

imshow(result)