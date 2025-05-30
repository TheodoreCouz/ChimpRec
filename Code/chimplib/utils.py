import sys
sys.path.append("PATH TO /Code")

from chimplib.imports import *

# @inputs:
# image: image to be drawn on
# color: color of the bounding boxe (in rgb format)
# bbox: coordinates of the bounding box to be drawn (in format (x1, y1, x2, y2))
#   with:
#   (x1, y1): top left corner coordinates
#   (x2, y2): bottom right corner coordinates
# label: text to be written on the bounding box (typically the individual's name)
# @output:
# image: the input image annotated with the bounding box
def draw_bbox(image, color, bbox, label):
    x1, y1, x2, y2 = map(lambda v: int(float(v)), bbox)
    factor = 0.65 if label == "Face" else 0.3
    font_scale = max(0.65, ((x2 - x1 + y2 - y1) / 300) * factor)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    label_text = f"{label}"#: {score:.2f}"
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)

    overlay = image.copy()
    cv2.rectangle(overlay, (x2 - w - 10, y2 - h - 10), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    cv2.putText(image, label_text, (x2 - w - 5, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), 1)
    return image

# @inputs:
# image: image to be drawn on
# color: color of the label (in rgb format)
# bbox: coordinates of the bounding box (indicating the coordinates of the label) to be drawn (in format (x1, y1, x2, y2))
#   with:
#   (x1, y1): top left corner coordinates
#   (x2, y2): bottom right corner coordinates
# label: text to be written on the label (typically the individual's name)
# @output:
# image: the input image annotated with the label
def draw_triangle(image, color, bbox, label):

    x1, y1, x2, y2 = map(lambda v: int(float(v)), bbox)    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.8, 1)

    # Triangle dimensions
    triangle_height = 20
    triangle_width = 20

    # Tip of triangle at the middle of the top edge of the bounding box
    tip_x = (x1 + x2) // 2
    tip_y = max(y1, text_height + triangle_height + 5)

    # Base is ABOVE the tip (Y decreases upward)
    base_y = tip_y - triangle_height
    half_base = triangle_width // 2

    pt_tip = (tip_x, tip_y)
    pt_left = (tip_x - half_base, base_y)
    pt_right = (tip_x + half_base, base_y)

    triangle = np.array([pt_tip, pt_left, pt_right], dtype=np.int32).reshape((-1, 1, 2))

    # Draw filled triangle
    cv2.fillPoly(image, [triangle], color)

    # Text position: centered horizontally, inside triangle
    text_x = tip_x - text_width // 2
    text_y = base_y + (triangle_height // 2) - triangle_height

    # Draw transparent rectangle using overlay
    overlay = image.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), color, -1)
    alpha = 0.6  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1-alpha, 0, image)

    # Draw label text on top
    cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)

    return image

# @inputs:
# coord: bounding box coordinates (in format (x1, y1, x2, y2))
#   with:
#   (x1, y1): top left corner coordinates
#   (x2, y2): bottom right corner coordinates
# img_width
# img_height
# @output:
# image: bounding box coordinates (in YOLO format (cx, cy, w, h))
def yolo_to_relative_coord(bbox, img_dim):
    """
    Compute the coordinates of the bbox in pixels term.

    Args:
        bbox(tuple): Bbox in YOLO format.
        img_dim (tuple): Dimension of the image.

    Returns:
        tuple: The coordinate of the top left an the bottom right corner of the bbox in pixel term.
    """ 
    x_center, y_center, width, height = bbox
    img_w, img_h = img_dim
    
    x_min = (x_center - width / 2) * img_w
    y_min = (y_center - height / 2) * img_h
    x_max = (x_center + width / 2) * img_w
    y_max = (y_center + height / 2) * img_h
    
    return [x_min, y_min, x_max, y_max]

def convert_to_yolo(bbox, img_dim=(1080, 1920)):
    """
    Compute the coordinates of the bbox in pixels term.

    Args:
        bbox(tuple): Cordinate of the top left an the bottom right corner of the bbox in pixel term..
        img_dim (tuple): Dimension of the image.

    Returns:
        tuple: The coordinate of the bbox in YOLO format ([x_center, y_center, width, height]).
    """
    x1, y1, x2, y2 = bbox
    img_width, img_height = img_dim

    # Compute center coordinates
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0

    # Compute width and height
    width = x2 - x1
    height = y2 - y1

    # Normalize values by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return [x_center, y_center, width, height]