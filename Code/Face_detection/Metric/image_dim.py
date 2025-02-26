import cv2
import os

# Replace with your folder path containing images
folder_path = 'C:/Users/Theo/Documents/Unif/detection test set/images'

dims = set()

# Iterate over each file in the folder
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Check if it's a file (ignoring subdirectories)
    if os.path.isfile(file_path):
        # Read the image using OpenCV
        image = cv2.imread(file_path)
        
        # If the file is not an image, imread returns None
        if image is None: continue
        
        # Get the dimensions: OpenCV returns (height, width, channels)
        height, width = image.shape[:2]
        dim = (height, width)
        dims.add(dim)

print(dims)
