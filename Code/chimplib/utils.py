import sys
sys.path.append("C:\\Users\\Theo\\Documents\\Unif\\ChimpRec\\Code")

from imports import *

def draw_bbox(image, color, bbox, label):
    x1, y1, x2, y2, score = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    factor = 0.65 if label == "Face" else 0.3
    font_scale = max(0.4, (x2 - x1 + y2 - y1) / 300) * factor

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    label_text = f"{label}"#: {score:.2f}"
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, 1)

    overlay = image.copy()
    cv2.rectangle(overlay, (x2 - w - 10, y2 - h - 10), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    cv2.putText(image, label_text, (x2 - w - 5, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255,255,255), 1)
    return image