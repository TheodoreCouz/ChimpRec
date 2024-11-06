import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import CocoDetection
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T

# Function to load and fine-tune Mask R-CNN model
def fine_tune_maskrcnn(video_folder_path, annotation_folder_path, num_epochs=10, batch_size=2, learning_rate=0.005):
    # Load the pre-trained Mask R-CNN model with the correct weights parameter
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = models.detection.maskrcnn_resnet50_fpn(weights=weights)
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, 2  # 2 classes: background and chimpanzee
    )

    # Define dataset and dataloader for all annotation files
    transform = T.Compose([T.ToTensor()])
    datasets = []
    for annotation_file in os.listdir(annotation_folder_path):
        if annotation_file.endswith('.json'):
            # Ensure annotations are in the correct COCO dictionary format
            datasets.append(CocoDetection(
                root=video_folder_path,
                annFile=os.path.join(annotation_folder_path, annotation_file),
                transform=transform
            ))
    dataset = torch.utils.data.ConcatDataset(datasets)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Set up device and move model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)

    # Fine-tuning loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(data_loader)}")

# Example usage
fine_tune_maskrcnn(
    video_folder_path="/home/theo/Documents/Unif/Master/ChimpRec - Datasets/ChimpBehave/chimpbehave/original", 
    annotation_folder_path="/home/theo/Documents/Unif/Master/ChimpRec - Datasets/ChimpBehave/bboxes", 
    num_epochs=10)
