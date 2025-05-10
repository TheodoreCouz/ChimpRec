import cv2
import os

def downscale_image_cv2(input_path, scale_factor=0.5, output_path=None):
    """
    Downscale an image using OpenCV by a given scale factor.

    Args:
        input_path (str): Path to the input image.
        scale_factor (float): Scale factor to reduce the resolution (e.g., 0.5 = 50%).
        output_path (str, optional): Path to save the downscaled image. 
                                     If None, saves with '_lowres' suffix.

    Returns:
        str: Path to the saved low-resolution image.
    """
    # Read the image
    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")

    # Calculate new size
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    # Resize image
    lowres_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_lowres{ext}"

    # Save the image
    cv2.imwrite(output_path, lowres_image)
    return output_path

# Example usage:
path = downscale_image_cv2("C:/Users/Theo/Pictures/Screenshots/Screenshot 2025-05-09 114738.png", scale_factor=0.2, output_path="C:/Users/Theo/Pictures/Screenshots/low_quality_image.png")
print(f"Low-res image saved to: {path}")
