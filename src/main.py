from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO('yolo11n.pt') 

    model.train(
        data='../pallet.yaml',
        epochs=50,
        imgsz=800, 
        batch=8, 
        device=0,
        workers=4,
        name='pallet_model_v1'
    )

if __name__ == '__main__':
    main()