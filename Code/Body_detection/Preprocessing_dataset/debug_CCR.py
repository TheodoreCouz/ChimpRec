import sys
sys.path.append('C:/Users/Theo/Documents/Unif/ChimpRec/Code')

from chimplib.utils import draw_bbox
from chimplib.imports import cv2, pd

# This file helps understaning whether a dataset has correctly been built.
# It helps visualise if the bounding boxes are correctly aligned with the individuals.

def preprocess_coordinates(coord, img_W, img_H):
    x, y, w, h = coord
    res = (
        int(x*img_W), 
        int(y*img_W), 
        int(x*img_W+w*img_W), 
        int((y*img_W)+h*img_H)
        )
    return res

def array_to_video(images, output_path, fps, frame_size):
    # Initialise the video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img in images:
        out.write(img)

    out.release()  # Release the video writer

def process_image(frame_count, img_W, img_H, video_name=""):
    df = pd.read_csv(".../CCR/metadata/annotations/body_data.csv")
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

        boxes = process_image(nframes+1, frame.shape[1], frame.shape[0], "example video name")
        new_image = frame
        for bbox in boxes:
            new_image = draw_bbox(new_image, (0, 204, 102), boxes, "Body")
        new_images.append(new_image)

        nframes+=1
        print(nframes)

    cap.release()  # Release the video capture object

    # Create a new video from the processed frames
    output_path = 'out.mp4'
    array_to_video(new_images, output_path, fps, frame_size)

    print(f"Video saved as {output_path}")

if (__name__ == "__main__"):
    
    vid_path = "..." # video to be annotated path
    process_video(vid_path, 150)