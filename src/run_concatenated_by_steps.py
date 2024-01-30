from diffusers_data_pipeline import concatenated_by_steps
import argparse

def parse_args(input_args = None):
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_name", 
        type=str,
        default="concatenated_by_steps"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    # concatenated_by_steps("logs/jjanggu_and_scene16", "logs/jjanggu_and_scene16", "scene16_500/scene16_1000/scene16_1500/scene16_2000")
    concatenated_by_steps(args.file_path, args.output_path, args.image_name, args.keywords)
