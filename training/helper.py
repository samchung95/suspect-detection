from imgaug.augmentables.bbs import BoundingBox
from PIL import Image
from sklearn.model_selection import train_test_split

import shutil
import os
import cv2
import random
import yaml
import os
import shutil
import numpy as np

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

def create_yaml(train_path, val_path, class_names, output_yaml_path):
    data = dict(
        train = train_path,
        val = val_path,
        nc = len(class_names),
        names = class_names,
    )

    with open(output_yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def getImagesWithClass(path,class_name):
    label_files = sorted(os.listdir(path))
    label_files = [i for i in label_files if '.txt' in i]
    for file in label_files:
        filepath = os.path.join(path, file)
        with open(filepath) as f:
            lines = f.readlines()
            #print(lines)
            for line in lines:
                class_ = line.split(' ')[0]
                if class_ == class_name:
                    print(file)

def prepareValidation(trainpath,valpath,newtrainpath):
    all_files = sorted(os.listdir(trainpath))
    label_files = [i for i in all_files if ".txt" in i]
    labels_dict = {}
    for label_file in label_files:
        with open(os.path.join(trainpath,label_file)) as f:
            for line in f:
                class_id = line.strip().split()[0]
                if class_id in labels_dict:
                    labels_dict[class_id].add(label_file)
                else:
                    labels_dict[class_id] = set([label_file])
    # print(labels_dict)
    # _t=[]
    # for id,labels in labels_dict.items():
    #     _t.append((id,len(labels)))
    # print(sorted(_t,key=lambda x: x[1]), len(_t))
    all_labels = set(label_files)
    all_train = set()
    all_val = set()
    for id,labels in labels_dict.items():

        length = len(labels)
        val_labels = set(random.sample(list(labels),k=int(length*0.20)))
        all_val = all_val | val_labels
        
        
    all_train= all_labels - all_val

    for lab in all_val:
        shutil.copy(os.path.join(trainpath,lab), os.path.join(valpath,lab))
        img_file = lab.replace('.txt','.png')
        shutil.copy(os.path.join(trainpath,img_file), os.path.join(valpath,img_file))
        
    for lab in all_train:
        shutil.copy(os.path.join(trainpath,lab), os.path.join(newtrainpath,lab))
        img_file = lab.replace('.txt','.png')
        shutil.copy(os.path.join(trainpath,img_file), os.path.join(newtrainpath,img_file))


def get_labels_for_image(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            labels.append({'class_id': int(class_id), 'bbox': (x_center, y_center, width, height)})
    return labels

def splitByClass(input_filepath='dataset/train', class_filepath='dataset/classes'):
    # Loop over files in input directory
    for filename in os.listdir(input_filepath):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # assuming these image formats, add more if needed
            # Open image
            image = cv2.imread(os.path.join(input_filepath, filename))

            # Get the labels for this image. 
            label_path = os.path.join(input_filepath, os.path.splitext(filename)[0] + '.txt')
            labels = get_labels_for_image(label_path)

            for label in labels:
                class_name = str(label['class_id'])  # replace with mapping from class id to name if available
                bbox = label['bbox']
                x_center, y_center, width, height = bbox
                # Convert from relative coordinates to absolute
                abs_x_center, abs_y_center = int(x_center * image.shape[1]), int(y_center * image.shape[0])
                abs_width, abs_height = int(width * image.shape[1]), int(height * image.shape[0])
                # Convert to xmin, ymin, xmax, ymax
                xmin, ymin = abs_x_center - abs_width // 2, abs_y_center - abs_height // 2
                xmax, ymax = xmin + abs_width, ymin + abs_height
                # Crop the image
                cropped_image = image[ymin:ymax, xmin:xmax] 

                # Destination folder for this class
                dest_folder = os.path.join(class_filepath, class_name)

                # If destination folder doesn't exist, create it
                os.makedirs(dest_folder, exist_ok=True)

                # Save the cropped image in the respective class folder
                cv2.imwrite(os.path.join(dest_folder, filename), cropped_image)


def splitTrainVal(input_filepath='dataset/train', train_filepath='dataset/newtrain', validation_filepath='dataset/validation'):
    if not os.path.exists(train_filepath):
        os.makedirs(train_filepath)
    if not os.path.exists(validation_filepath):
        os.makedirs(validation_filepath)

    # Get list of all files in input directory
    file_list = os.listdir(input_filepath)

    # Prepare empty lists to hold file paths
    all_files = []

    # Iterate over all files
    for filename in file_list:
        # Ensure we're only working with .txt files (YOLOv5 labels are stored in .txt files)
        if filename.endswith(".txt"):
            # Read the content of the file
            with open(os.path.join(input_filepath, filename), 'r') as f:
                content = f.readlines()

            # Change class to 0 for all labels in this file
            new_content = []
            for line in content:
                items = line.strip().split(' ')
                items[0] = '0'  # Change class label to 0
                new_content.append(' '.join(items) + '\n')

            # Add to list
            all_files.append((filename.replace('.txt','.png'), new_content))
            

    # Split dataset into training and validation sets (80% - 20% split)
    train_files, validation_files = train_test_split(all_files, test_size=0.2, random_state=42)

    # Save files to new directories
    for filename, content in train_files:
        if filename.endswith(".txt"):
            with open(os.path.join(train_filepath, filename), 'w') as f:
                f.writelines(content)
            # Move corresponding image file
            shutil.copy2(os.path.join(input_filepath, filename.replace('.txt', '.png')), train_filepath)
        elif filename.endswith((".jpg", ".png")):
            shutil.copy2(os.path.join(input_filepath, filename), train_filepath)
            with open(os.path.join(train_filepath, filename.replace('.png', '.txt')), 'w') as f:
                f.writelines(content)
    for filename, content in validation_files:
        if filename.endswith(".txt"):
            with open(os.path.join(validation_filepath, filename), 'w') as f:
                f.writelines(content)
            # Copy corresponding image file
            shutil.copy2(os.path.join(input_filepath, filename.replace('.txt', '.png')), validation_filepath)
        elif filename.endswith((".jpg", ".png")):
            shutil.copy2(os.path.join(input_filepath, filename), validation_filepath)
            with open(os.path.join(validation_filepath, filename.replace('.png', '.txt')), 'w') as f:
                f.writelines(content)





if __name__ == "__main__":
    # class_names = ["plushie"+str(i) for i in range(200)]  # Replace with your class names
    # root_path = os.getcwd()  # Gets the current working directory
    # train_path = os.path.join(root_path, 'dataset\\train')  # Path to your training images
    # val_path = os.path.join(root_path, 'dataset\\validation')  # Path to your validation images
    # output_yaml_path = os.path.join(root_path, 'dataset\\data.yaml')  # Output path for the YAML file

    # create_yaml(train_path, val_path, class_names, output_yaml_path)
    # prepareValidation('dataset/train','dataset/validation','dataset/newtrain')
    # getImagesWithClass('dataset/train','6')
	#splitByClass()
    splitTrainVal()
