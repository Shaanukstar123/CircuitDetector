from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO('yolo11s.pt') 

    model.train(
        data='../pallet.yaml',
        epochs=100,
        imgsz=1024, 
        batch=8, 
        device=0,
        workers=8,
        mosaic=0.0,
        name='pallet_model_v1'
    )

if __name__ == '__main__':
    main()