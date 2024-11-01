# Chimpanzee Facial Recognition Dataset Documentation

This dataset is organized to support facial recognition tasks for chimpanzees. The main directory contains three primary folders: `Dataset`, `Images`, and `Videos`. Below is an overview of the folder structure and the contents within each.

## Folder Structure

├── Dataset
│   ├── A
│   │   ├── face
│   │   │   └── <cropped_face_images_of_individual_A>
│   │   ├── body
│   │   │   └── <cropped_body_images_of_individual_A>
│   │   └── full
│   │       └── <full_images_of_individual_A>
│   ├── B
│   │   ├── face
│   │   │   └── <cropped_face_images_of_individual_B>
│   │   ├── body
│   │   │   └── <cropped_body_images_of_individual_B>
│   │   └── full
│   │       └── <full_images_of_individual_B>
│   └── ...
├── Images
│   └── <unlabeled_images_for_identity_card_enhancement>
└── Videos
    └── <chimpanzee_videos_for_dataset_enhancement>


### Folder Descriptions

1. **Dataset/**: Contains labeled subfolders, each representing an individual chimpanzee. Each subfolder is temporarily labeled (e.g., A, B, C...) until we finalize the actual names of the individuals. Within each individual folder:
   - `face/`: Contains cropped images focused on the face of this chimpanzee.
   - `body/`: Contains cropped images showing the body of this individual.
   - `full/`: Contains full-body images from which the images in `face/` and `body/` have been extracted.

2. **Images/**: Contains unlabeled images of chimpanzees that may be used to enhance or enrich the profiles (or "identity cards") of each chimpanzee in `Dataset/`.

3. **Videos/**: Contains videos of chimpanzees, which can be used to extract additional images to expand the dataset in `Dataset/`.

### Notes

- The structure is designed for flexibility in updating the individual folders in `Dataset` with proper names once known.
- Images in the `face/` and `body/` folders are cropped from those in `full/` to focus specifically on facial and body regions relevant to recognition tasks.

This organization helps streamline the use of the dataset for machine learning tasks focused on facial recognition while allowing for easy updating and expansion as more data becomes available.
