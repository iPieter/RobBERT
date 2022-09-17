from app import create_app
import argparse

def create_parser():
    "Utility function to create the CLI argument parser"

    parser = argparse.ArgumentParser(
        description="Create a REST endpoint for for 'die' vs 'dat' disambiguation."
    )

    parser.add_argument("--model-path", help="Path to the finetuned RobBERT identifier.", required=False)
    parser.add_argument("--fast-model-path", help="Path to the mlm RobBERT identifier.", required=False)

    return parser

if __name__ == "__main__":
    arg_parser = create_parser()
    args = arg_parser.parse_args()

    create_parser()
    create_app(args.model_path, args.fast_model_path).run()