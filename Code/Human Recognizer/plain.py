import cv2

# change the path according to your file system
haar_cascade = cv2.CascadeClassifier("/home/theo/Documents/Unif/Master/ChimpRec/Human Recognizer/haarcascade_frontalface_default.xml") 

src_img = cv2.imread("inference/1_0.png", 0)
src_img_grey = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

faces = haar_cascade.detectMultiScale(
    src_img_grey,
    scaleFactor=1.05,
    minNeighbors=1,
    minSize=(40, 40) # to adjust
)

i = 0

for x, y, w, h in faces:
    cropped = src_img[y: y+h, x : x+h]
    target_file_name = "stored-faces/" + str(i) + ".jpg"
    cv2.imwrite(
        target_file_name,
        cropped
    )
    i+=1

import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity


face_folder = "stored-faces"

embeddings = []

for filename in os.listdir(face_folder):
    img = Image.open(f"{face_folder}/{filename}")
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    embeddings.append(embedding)

print(type(embeddings[0]))

known_embeddings = {}

for filename in os.listdir("dataset"):
    name = filename.strip(".png")
    img = Image.open(f"{"dataset"}/{filename}")
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    known_embeddings[name] = embeddings

for embed in embeddings:
    # Flatten the embedding to 1D if it's multi-dimensional
    embed = embed.reshape(1, -1)  # Reshaping to ensure it's 2D (1 sample, many features)
    
    best_sim = (0, "nobody")
    for name in known_embeddings.keys():
        # Reshape known embeddings to be 2D as well
        known_embed = np.array(known_embeddings[name]).reshape(1, -1)
        
        # Calculate cosine similarity
        sim = cosine_similarity(embed, known_embed)[0][0]
        
        # Update the best match
        if sim > best_sim[0]:
            best_sim = (sim, name)
    
    print(best_sim)

print(known_embeddings)