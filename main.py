import argparse
from Model.TorchScript import TorchScript_Satellite_Detection


def main() -> None:
    TorchScript = TorchScript_Satellite_Detection()

    parser = argparse.ArgumentParser(description="YOLO Model Trainer and Manager CLI")

    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Sub-command to create a new YOLO model
    create_parser = subparsers.add_parser("create", help="Create a new YOLO model")
    create_parser.add_argument(
        "model_size", type=str, help="Name of the model to create (n,s,m,l,x)"
    )

    # Sub-command to save the trained model
    save_parser = subparsers.add_parser("save", help="Save the trained YOLO model")
    save_parser.add_argument("model_name", type=str, help="Name of the model to save")
    save_parser.add_argument(
        "save_location", type=str, help="Path to save the model file"
    )

    # Sub-command to load an existing YOLO model
    load_parser = subparsers.add_parser("load", help="Load an existing YOLO model")
    load_parser.add_argument(
        "model_path", type=str, help="Path to the model file to load"
    )

    args = parser.parse_args()

    if args.command == "create":
        TorchScript.new_model(args.model_size)

    elif args.command == "load":
        TorchScript.load(args.model_path)


if __name__ == "__main__":
    main()
