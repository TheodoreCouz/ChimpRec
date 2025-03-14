import os

# Paths
original_labels_path = r"C:/Users/Theo/Documents/Unif/detection_test_set/labels_not_filtered"
filtered_labels_path = r"C:/Users/Theo/Documents/Unif/detection_test_set/labels"

# Create the filtered labels directory if it doesn't exist
os.makedirs(filtered_labels_path, exist_ok=True)

# Loop through each label file
for file in os.listdir(original_labels_path):
    if file.endswith(".txt"):
        original_file_path = os.path.join(original_labels_path, file)
        filtered_file_path = os.path.join(filtered_labels_path, file)

        with open(original_file_path, "r") as f:
            lines = f.readlines()

        # Keep only bounding boxes where class_id == 1
        filtered_lines = [line for line in lines if line.strip().startswith("1")]
        for i in range(len(filtered_lines)):
            filtered_lines[i] = f"0{filtered_lines[i][1:]}"

        # Write filtered data to new label file
        with open(filtered_file_path, "w") as f:
            f.writelines(filtered_lines)

print("Filtered labels saved to:", filtered_labels_path)
