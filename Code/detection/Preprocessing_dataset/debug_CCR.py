import cv2
import pandas as pd

def preprocess_coordinates(coord, img_W, img_H):
    x, y, w, h = coord
    res = (
        int(x*img_W), 
        int(y*img_W), 
        int(x*img_W+w*img_W), 
        int((y*img_W)+h*img_H)
        )
    return res

def to_yolo_format(coord, img_width, img_height):

    x, y, w, h = coord
    
    x1 = int(x*img_width)
    y1 = int(y*img_width)
    x2 = int(x*img_width+w*img_width)
    y2 = int((y*img_width)+h*img_height)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    width = x2 - x1
    height = y2 - y1

    cx_norm = cx / img_width
    cy_norm = cy / img_height

    width_norm = width / img_width
    height_norm = height / img_height

    return [cx_norm, cy_norm, width_norm, height_norm]


def draw_boxes(image, boxes):
    """
    Draw the bounding boxes on the frame.
    :param frame: The image frame.
    :param boxes: A list of bounding boxes. Each bounding box can be either (x, y, w, h) or (x, y, size).
    :return: The frame with the boxes drawn on it.
    """
    boxed_image = image
    for box in boxes:


        # Ensure coordinates are integers
        x1, y1, x2, y2 = box

        # Draw the rectangle on the frame
        cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    return boxed_image

def array_to_video(images, output_path, fps, frame_size):
    # Initialize the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img in images:
        out.write(img)

    out.release()  # Release the video writer

def process_image(frame_count, img_W, img_H, video_name="15.mp4"):
    df = pd.read_csv("/home/theo/Documents/Unif/Master/ChimpRec/ChimpRec-Dataset/CCR/metadata/annotations/body_data.csv")
    df = df.loc[df["video"] == video_name].loc[df["frame"] == frame_count].loc[df["label"] != "NOTCHIMP"]
    boxes = []

    for index, row in df.iterrows():
        x, y, w, h = row['x'], row['y'], row['w'], row['h']
        box = preprocess_coordinates((x, y, w, h), img_W, img_H)
        boxes.append(box)
    
    return boxes

def process_video(video_path, frame_limit=10000):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties like FPS (Frames Per Second) and frame size
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    new_images = []

    nframes = 0

    boxes = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or nframes > frame_limit:
            break  # Exit loop if there are no more frames

        print(frame.shape)

        boxes = process_image(nframes+1, frame.shape[1], frame.shape[0], "15.mp4")
        new_image = draw_boxes(frame, boxes)
        new_images.append(new_image)

        nframes+=1
        print(nframes)

    cap.release()  # Release the video capture object

    # Create a new video from the processed frames
    output_path = 'out.mp4'
    array_to_video(new_images, output_path, fps, frame_size)

    print(f"Video saved as {output_path}")

if (__name__ == "__main__"):
    vid_path = "/home/theo/Documents/Unif/Master/ChimpRec/ChimpRec-Dataset/CCR/videos/15.mp4"
    process_video(vid_path, 150)
    # df = pd.read_csv("/home/theo/Documents/Unif/Master/ChimpRec/ChimpRec-Dataset/CCR/metadata/annotations/body_data.csv")
    # df = df.loc[df["video"] == "15.mp4"].loc[df["frame"] == 2]
    # print(df)