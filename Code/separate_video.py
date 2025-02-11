import cv2
import os
import random

def extract_frames(video_path, output_folder, n_frames=None):
    filename = video_path.split("/")[-1].split(".")[0]
    print(filename)
    
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
            cv2.imwrite(frame_filename, frame)
            selected_idx += 1
        
        frame_count += 1
    
    cap.release()

video_path = '/home/theo/Documents/Unif/Master/viedeos/20241214 - 08h37.MP4'  # Remplace avec le chemin de ta vidéo
output_folder = '/home/theo/Documents/Unif/Master/last_output'  # Dossier où enregistrer les frames
extract_frames(video_path, output_folder, 720)