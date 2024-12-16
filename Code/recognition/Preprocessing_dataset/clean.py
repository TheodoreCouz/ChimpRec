import os
import shutil
import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

path = "./ChimpRec-Dataset/Chimpanzee_recognition_dataset"

if os.path.exists(path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
        print(f"Successfully removed: {path}")
    except Exception as e:
        print(f"Error removing directory {path}: {e}")
else:
    print(f"The path '{path}' does not exist.")

os.makedirs(path, exist_ok=True)
print(f"Recreated empty directory: {path}")
