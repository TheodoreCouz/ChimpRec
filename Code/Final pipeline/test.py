from ultralytics import YOLO
import cv2
from tqdm import tqdm

model_1 = YOLO("/home/theo/Documents/Unif/Master/ChimpRec/Code/Body_detection/YOLO_small/runs/detect/train9/weights/best.pt")
model_2 = YOLO("/home/theo/Documents/Unif/Master/ChimpRec/Code/Face_detection/runs/detect/train3/weights/best.pt")

video_path = "/home/theo/Documents/Unif/Master/Chimprec - Extra/videos/20241023 - 09h28.MP4"
n = 3  # Process one frame every n frames
max_frames = 120  # Limit frame processing for efficiency

cap = cv2.VideoCapture(video_path)
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

def predict_1(image, t_confidence=0.4):
    return tuple(
        (x1, y1, x2, y2, score)
        for x1, y1, x2, y2, score, _ in model_1.predict(image, verbose=False)[0].boxes.data.tolist()
        if score >= t_confidence
    )

def predict_2(image, t_confidence=0.6):
    results = model_2.predict(image, verbose=False)
    return max(
        ((int(x1), int(y1), int(x2), int(y2), score) for result in results for x1, y1, x2, y2, score, _ in result.boxes.data.tolist()),
        default=None, key=lambda x: x[-1] if x[-1] >= t_confidence else float('-inf')
    )

def crop(image, bbox):
    x1, y1, x2, y2, _ = map(int, bbox)
    return image[max(y1, 0):y2, max(x1, 0):x2]

def face_to_src(body_bbox, face_bbox):
    bx1, by1, _, _, _ = body_bbox
    fx1, fy1, fx2, fy2, score = face_bbox
    return (bx1 + fx1, by1 + fy1, bx1 + fx2, by1 + fy2, score)

def predict_frame(image):
    body_bboxes = predict_1(image)
    face_bboxes = tuple(
        face_to_src(body_bbox, face_bbox)
        for body_bbox in body_bboxes
        if (face_bbox := predict_2(crop(image, body_bbox)))
    )
    return body_bboxes, face_bboxes

bboxes = []
with tqdm(total=max_frames, desc="Processing frames") as pbar:
    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        bboxes.append(predict_frame(frame) if frame_idx % n == 0 else bboxes[-1])
        pbar.update(1)

cap.release()
cv2.destroyAllWindows()