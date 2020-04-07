from app import create_app
import argparse

def create_parser():
    "Utility function to create the CLI argument parser"

    parser = argparse.ArgumentParser(
        description="Create a REST endpoint for for 'die' vs 'dat' disambiguation."
    )

    parser.add_argument("--model-path", help="Path to the finetuned RobBERT folder.", required=True)


    return parser

if __name__ == "__main__":
    arg_parser = create_parser()
    args = arg_parser.parse_args()

    create_parser()
    create_app(args.model_path).run()