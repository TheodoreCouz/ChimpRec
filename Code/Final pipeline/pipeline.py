from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# returns a list of bounding boxes
def predict_1(frame_path, t_confidence=0.4):
    model_1_path = "/home/theo/Documents/Unif/Master/ChimpRec/Code/Body_detection/YOLO_small/runs/detect/train9/weights/best.pt"
    model_1 = YOLO(model_1_path)
    results = model_1(frame_path, verbose=False)[0]
    predicted_bboxes = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score < t_confidence : continue
        predicted_bboxes.append((x1, y1, x2, y2, score))
    return predicted_bboxes

#return one image cropped according to the bbox
def crop(image, bbox):
    img_height, img_width, _ = image.shape
    x1, y1, x2, y2, score = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_width, int(x2)), min(img_height, int(y2))

    cropped = image[y1:y2, x1:x2]
    return cropped  # Keep in BGR format for OpenCV


def save_image(image, output_folder, bbox):
    os.makedirs(output_folder, exist_ok=True)
    output_path = f"{output_folder}/{abs(hash(bbox))}.png"
    cv2.imwrite(output_path, image)


# prediction for 1 frame (1 body)
def predict_2(folder_path, bbox, t_confidence=0.4):
    model_2_path = "/home/theo/Documents/Unif/Master/ChimpRec/Code/Face_detection/runs/detect/train3/weights/best.pt"
    model_2 = YOLO(model_2_path)
    file_path = os.path.join(folder_path, f"{abs(hash(bbox))}.png")
    image = cv2.imread(file_path)
    results = model_2.predict(image, verbose=False)

    best_score = 0
    best_match = None

    # Process detections
    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box[:6]
            if (score > best_score): 
                best_match = map(int, (x1, y1, x2, y2))
                best_score = score
    if best_match == None or best_score < t_confidence: return None
    res = list(best_match)
    res.append(score)
    return tuple(res)
            
def face_to_src(body_bbox, face_bbox):
    body_x1, body_y1, _, _, _ = body_bbox  # Top-left corner of the body bbox in src_img
    face_x1, face_y1, face_x2, face_y2, score = face_bbox  # Face bbox relative to the body bbox

    # Convert to src_img coordinates by adding the body bbox top-left offset
    new_x1 = body_x1 + face_x1
    new_y1 = body_y1 + face_y1
    new_x2 = body_x1 + face_x2
    new_y2 = body_y1 + face_y2

    return (int(new_x1), int(new_y1), int(new_x2), int(new_y2), score)

def draw_bbox(image, color, bbox, label):

    factor = 0
    if label == "Face": factor = 0.65
    if label == "Body": factor = 0.3
    

    white = (255, 255, 255)
    x1, y1, x2, y2, score = bbox
    overlay = image.copy()

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    # Adjust thickness and font size dynamically
    thickness = 4
    font_thickness = 2
    font_scale = max(0.4, min(bbox_width, bbox_height) / 250)*factor

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Prepare label
    label_text = f"{label}: {score:.2f}"
    (label_width, label_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX, font_scale, font_thickness)

    # Background for the label
    overlay = image.copy()
    cv2.rectangle(overlay, (x2 - label_width - 10, y2 - label_height - 10), (x2, y2), color, -1)
    alpha = 0.8  # transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Draw rectangle around the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label text
    cv2.putText(image, label_text, (x2 - label_width - 5, y2 - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                font_scale, white, font_thickness)

def display_annotated_image(image_path, body_bboxes, face_bboxes):

    body_color = (51, 122, 254)
    face_color = (95, 226, 255)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_COMPLEX

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert BGR to RGB for Matplotlib display
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for bbox in face_bboxes:
        draw_bbox(image, face_color, bbox, "Face")

    for bbox in body_bboxes:
        draw_bbox(image, body_color, bbox, "Body")

    # Display the image with bounding boxes
    # plt.figure(figsize=(10, 6))
    # plt.imshow(image)
    # plt.axis("off")  # Hide axes for better visualization
    # plt.show()
    return image

def clear_folder(path):
    """
    Removes all files and subfolders within the specified folder,
    but does not remove the folder itself.

    :param path: Path to the target folder.
    """
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def predict_frame(frame_path):
    image = cv2.imread(frame_path)
    cropped_images_folder = "./body_cropped"
    
    # all the body detections within the initial image
    body_bboxes = predict_1(frame_path)

    body_cropped_images = []

    for bbox in body_bboxes:
        body_cropped_images.append(crop(image, bbox))

    for i in range(len(body_cropped_images)):
        save_image(body_cropped_images[i], "body_cropped", body_bboxes[i])

    face_bboxes = []

    for bbox in body_bboxes:
        res = predict_2("body_cropped", bbox)
        face_bboxes.append(res)

    ####SAVE THE CROPPED IMAGES####

    face_cropped_images = []

    for i, bbox in enumerate(face_bboxes):
        if bbox == None: 
            face_cropped_images.append(None)
            continue

        path = f"{cropped_images_folder}/{abs(hash(body_bboxes[i]))}.png"
        img = cv2.imread(path)
        face_cropped_images.append(crop(img, bbox))

    for i in range(len(body_cropped_images)):
        bbox = face_cropped_images[i]
        try:
            save_image(face_cropped_images[i], "face_cropped", body_bboxes[i])
        except: pass

    body_bbox_converted = []
    for bbox in body_bboxes:
        x1, y1, x2, y2, score = bbox
        body_bbox_converted.append((int(x1), int(y1), int(x2), int(y2), score))

    face_bbox_converted = []
    for i in range(len(body_bboxes)):
        try:
            face_bbox_converted.append(face_to_src(body_bboxes[i], face_bboxes[i]))
        except: pass
    
    clear_folder(cropped_images_folder)
    clear_folder("./face_cropped")

    return display_annotated_image(frame_path, body_bbox_converted, face_bbox_converted)

if __name__ == "__main__":

    video_path = "/home/theo/Documents/Unif/Master/Chimprec - Extra/videos/20241023 - 09h28.MP4"
    output_path = "output.mp4"
    frames_folder = "frames_from_video"

    cap = cv2.VideoCapture(video_path)

    # Get video properties for the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame_count > 30*5:
            break

        frame_name = f"frame_{frame_count:05d}.png"
        frame_filename = os.path.join(frames_folder, frame_name)
        cv2.imwrite(frame_filename, frame)

        frame_path = f"{frames_folder}/{frame_name}"
        annotated_frame = predict_frame(frame_path)

        out.write(annotated_frame)
        clear_folder("frames_from_video")

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
