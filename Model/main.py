import sys
import json
from YOLO import YOLO_Satellite_Detection


def main():
    if len(sys.argv) < 1:
        print("Usage: python main.py <command>")
        sys.exit(1)

    command = sys.argv[1]
    model = sys.argv[2]
    secondary = sys.argv[3]

    Yolo = YOLO_Satellite_Detection("/app/data", model_size=model)
    if command == "train":
        if len(sys.argv) > 3:
            epochs = int(secondary)
        else:
            epochs = 10
        result = Yolo.train(epochs=int(epochs))
        print("Training complete!")
    elif command == "eval":
        print(f"Model location: {secondary}")
        result = Yolo.evaluate(secondary)
        print("Evaluation complete!")
    elif command == "convert":
        result = Yolo.convert_coco_to_yolo()
        print("Conversion complete!")
    else:
        print("Invalid command. Use 'train', 'eval', or 'convert'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
