from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # load a pretrained model (recommended for training)
    model = YOLO("yolov8x.pt")

    # Use the model
    model.train(data="dataset/data.yaml",
                epochs=100,
                patience=20,
                device=0,
                batch=14,
                optimizer='Adam',
                lr0=0.001,
                lrf=0.001,
                flipud = 0.5,
                fliplr = 0.5)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    success = model.export(format="onnx")  # export the model to ONNX format
