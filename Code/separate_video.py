import cv2
import os
import random
import shutil
from PIL import Image

def extract_frames(video_path, output_folder, n_frames=None):
    filename = video_path.split("/")[-1].split(".")[0]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if n_frames is None or n_frames >= total_frames:
        selected_frames = list(range(total_frames))
    else:
        selected_frames = sorted(random.sample(range(total_frames), n_frames))
    
    frame_count = 0
    selected_idx = 0
    
    while frame_count < total_frames and selected_idx < len(selected_frames):
        ret, frame = cap.read()
        if not ret:
            break  
        
        if frame_count == selected_frames[selected_idx]:
            frame_filename = os.path.join(output_folder, f"{filename}_frame_{frame_count:04d}.png")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Conversion BGR → RGB
            image.save(frame_filename)
            selected_idx += 1
                
        frame_count += 1
    
    cap.release()

output_folder = "c:/Users/Theo/Documents/Unif/images"


folder_path = "E:/Nouvelles Vidéos"
videos = [
    "20241015 - 11h46.MP4",
    "20241015 - 12h29.MP4",
    "20241015 - 12h34-2.MP4",
    "20241015 - 12h41.MP4",
    "20241015 - 12h46.MP4",
    "20241016 - 06h58.MP4",
    "20241016 - 07h06.MP4",
    "20241016 - 07h58.MP4",
    "20241016 - 08h15.MP4",
    "20241017 - 12h14.MP4",
    "20241102 - 10h35.MP4"
]

video_paths = [f"{folder_path}/{i}" for i in videos]

for video_path in video_paths: extract_frames(video_path, output_folder, 50)
