import cv2
import os
import shutil

def extract_frames(video_path, output_folder, frame_hop=1):
    filename = video_path.split("/")[-1].split(".")[0]
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        
        if ((frame_count % frame_hop) == 0): 
            frame_filename = os.path.join(output_folder, f"{filename}_frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()

video_path = './Chimprec Dataset/Chimp_video/Identification/Individuelle/Lwama - LM/Lwama1.mp4'  # Remplace avec le chemin de ta vidéo
output_folder = 'separated_video_folder'  # Dossier où enregistrer les frames
extract_frames(video_path, output_folder)