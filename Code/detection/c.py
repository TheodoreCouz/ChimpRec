import cv2
import tensorflow as tf
import numpy as np
import time

# Function to load the frozen .pb model
def load_megadetector_model(model_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return graph

# Function to apply MegaDetector on an image
def detect_individuals(image, graph, threshold=0.5):
    boxes = []
    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')

        # Run the model
        (boxes_out, scores_out, classes_out) = sess.run(
            [detection_boxes, detection_scores, detection_classes],
            feed_dict={input_tensor: np.expand_dims(image, axis=0)}
        )

        # Filter by threshold and class type (assuming '1' represents animals)
        for i in range(len(scores_out[0])):
            if scores_out[0][i] >= threshold and classes_out[0][i] == 1:
                boxes.append(boxes_out[0][i])
    return boxes

# Function to draw boxes on an image
def draw_boxes(image, boxes):
    for (ymin, xmin, ymax, xmax) in boxes:
        start_point = (int(xmin * image.shape[1]), int(ymin * image.shape[0]))
        end_point = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    return image

# Main function to process video
def process_video(input_video_path, output_video_path, model_path, s):
    # Load MegaDetector model
    graph = load_megadetector_model(model_path)

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    fps = 30
    frame_count = int(s * fps)  # Number of frames to process based on `s` seconds
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    frame_idx = 0
    while cap.isOpened() and frame_idx < frame_count:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and annotate individuals on the frame
        boxes = detect_individuals(frame, graph)
        annotated_frame = draw_boxes(frame, boxes)

        # Write annotated frame to output video
        out.write(annotated_frame)
        frame_idx += 1

    # Release video resources
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

# Example usage
input_video_path = "/home/theo/Documents/Unif/Master/ChimpRec/Code/Superpixel R-CNN/Video/Video_1.mp4" # Replace with your input video path
output_video_path = 'out_2.mp4'
model_path = "/home/theo/Documents/Unif/Master/ChimpRec/Code/detection/models/megadetector.pb"  # Replace with your MegaDetector model path
s = 30  # Number of seconds to process


start = time.time()
process_video(input_video_path, output_video_path, model_path, s)
print(time.time() - start)