from ultralytics import YOLO
import cv2
import numpy as np

if __name__ == "__main__":
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # load a pretrained model (recommended for training)
    model = YOLO("runs/detect/train36/weights/best.pt")
    # Load the image
    # image = cv2.imread("dataset/augment/augmented_image_0012.png")
    # # Use the model
    # model.train(data="dataset/data.yaml",
    #             epochs=200,
    #             device=0,
    #             batch=14,
    #             optimizer='Adam',
    #             lr0=0.01,
    #             lrf=0.001,
    #             flipud = 0.5,
    #             fliplr = 0.5)  # train the model
    # metrics = model.val()  # evaluate model performance on the validation set
    results = model("dataset/test/image_1468.png")  # predict on an image
    # print(results)
    print(results[0].tojson())
    # For each detection
    # for detection in results:
    #     print(detection['boxes'])
    #     class_idx, prob, (x, y, w, h) = detection
    #     # Get the label for the class index
    #     label = f"{class_idx}: {prob * 100:.2f}%"

    #     # Draw a rectangle for the bounding box
    #     cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)

    #     # Put the label near the top of the bounding box
    #     cv2.putText(image, label, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # # Display the image with detections
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # success = model.export(format="onnx")  # export the model to ONNX format
