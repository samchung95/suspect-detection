import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from torchvision.datasets import CocoDetection

# Load a dataset in COCO format
dataset = CocoDetection('/path/to/your/data/', '/path/to/your/annotation.json',
                        transform=T.Compose([T.ToTensor()]))

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 2  # 1 class (plushie) + background

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

from torch.utils.data import DataLoader

# move model to the right device
# model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# training loop
num_epochs = 10
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # Returns losses and detections
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # update the learning rate
    lr_scheduler.step()

torch.save(model.state_dict(), 'model_weights.pth')
