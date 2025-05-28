import os
import glob

# when executed, this code removes all the labels and images within a dataset. 
# It is useful when you need to rebuild a dataset but you do not want to recreate the structure manually.

path = "..." # source path to the dataset you want to clean

# List of folders to clear
folders = [
    f"{path}/images/train",
    f"{path}/images/val",
    f"{path}/images/test",
    f"{path}/labels/train",
    f"{path}/labels/val",
    f"{path}/labels/test"
]

def clear_folder(folder_path):
    # Get all files in the folder
    files = glob.glob(f"{folder_path}/*")
    
    # Remove each file
    for file_path in files:
        try:
            os.remove(file_path)
            # print(f"Removed: {file_path}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")

# Clear all specified folders
for folder in folders:
    clear_folder(folder)

print("Cleaning done")
