from PIL import Image
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from helper import yolo_to_bbox, create_ds, bbox_to_yolo

import os
import cv2
import imgaug.augmenters as iaa
import numpy as np

AUGMENT_IMAGE_PATH = '/path/to/images/folder'
AUGMENT_LABELS_PATH = '/path/to/labels/folder'
ORIGINAL_IMAGE_PATH = '/path/to/images/folder'
ORIGINAL_LABELS_PATH = '/path/to/labels/folder'

# Make sure the directories exist
os.makedirs(AUGMENT_IMAGE_PATH, exist_ok=True)
os.makedirs(AUGMENT_LABELS_PATH, exist_ok=True)
os.makedirs(ORIGINAL_IMAGE_PATH, exist_ok=True)
os.makedirs(ORIGINAL_LABELS_PATH, exist_ok=True)

# Define your augmentation pipeline
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

def augment(image, yolo_labels, image_file):
    # Load your image

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB
    img_height, img_width, _ = image.shape

    yolo_boxes = [(i,img_height,img_width) for i in yolo_labels]

    bbs_list = list(map(yolo_to_bbox,yolo_boxes))

    # Define your bounding boxes
    # Note: in the format (x1, y1, x2, y2)
    bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)

    # Apply the augmentation
    image_aug, bbs_aug = augmentation(image=image, bounding_boxes=bbs)
    
    new_yolo_labels = list(map())

    # Convert image back to PIL Image and save it
    image_aug_pil = Image.fromarray(image_aug)
    image_aug_pil.save(os.path.join(AUGMENT_IMAGE_PATH, f'augmented_{image_file}.jpg'))

    yolo_boxes_aug = [bbox_to_yolo(bb_aug, img_width, img_height) for bb_aug in bbs_aug.bounding_boxes]
    with open(os.path.join(AUGMENT_LABELS_PATH, f'augmented_{image_file}.txt'), 'w') as f:
        for box in yolo_boxes_aug:
            f.write(f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n')

if __name__ == "__main__":
    _dataset = create_ds(ORIGINAL_IMAGE_PATH,ORIGINAL_LABELS_PATH)
    for img,labels,file in _dataset:
        augment(img,labels,file)