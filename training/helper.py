from imgaug.augmentables.bbs import BoundingBox
from PIL import Image
import os
import cv2

def yolo_to_bbox(yolo_box, img_height, img_width):
    """
    Convert YOLO box format to imgaug BoundingBox format.

    yolo_box: list of format [object_class, x_center, y_center, width, height]
    img_height: height of the image
    img_width: width of the image

    returns: imgaug BoundingBox object
    """
    object_class, x_center, y_center, width, height = yolo_box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=object_class)

def create_ds(image_dir,label_dir):
    # Directories of images and labels
    # image_dir = "/path/to/images"
    # label_dir = "/path/to/labels"

    # Get sorted lists of image and label files
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    dataset = []

    # Loop through each image file
    for image_file in image_files:
        # Open image using PIL
        # image = Image.open(os.path.join(image_dir, image_file))
        # Open image using cv2
        image = cv2.imread(os.path.join(image_dir, image_file))

        # Find corresponding label file (assuming they share the same base name)
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"

        # Read label file
        labels = []
        with open(os.path.join(label_dir, label_file)) as f:
            for line in f:
                # Convert each line to YOLO format (class, x_center, y_center, width, height)
                labels.append(list(map(float, line.strip().split())))

        # Add image and labels to dataset
        dataset.append((image, labels, image_file))
    return dataset
    # Now 'dataset' is a list of (image, labels) pairs, where labels is a list of bounding boxes

def bbox_to_yolo(bb, img_width, img_height, class_id):
    """
    Converts bounding box from BoundingBoxesOnImage format to YOLO format.

    Parameters:
    bb (BoundingBoxesOnImage object): The bounding box in imgaug format.
    width (int): The width of the image the bounding box is associated with.
    height (int): The height of the image the bounding box is associated with.
    class_id (int): class_id of yolo label

    Returns:
    list: The bounding box in YOLO format [x_center, y_center, width, height, class].
    """

    # Getting coordinates from bounding box
    x1, y1, x2, y2 = bb.x1, bb.y1, bb.x2, bb.y2

    # Converting coordinates to YOLO format
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    w = abs(x2 - x1) / img_width
    h = abs(y2 - y1) / img_height

    return [class_id, x_center, y_center, w, h]
