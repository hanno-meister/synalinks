import argparse
import os
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Copy a serialized program to a specified directory."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the file to be copied.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory.",
    )
    args = parser.parse_args()

    if not args.filepath.endswith(".json"):
        raise ValueError("The filepath must ends with `.json`")

    if not os.path.exists(args.output_dir):
        print(f"[*] Output directory does not exist. Creating: '{args.output_dir}'")
        os.makedirs(args.output_dir)

    filename = os.path.basename(args.filepath)
    destination = os.path.join(args.output_dir, filename)
    print(f"[*] Copying file from '{args.filepath}' to '{destination}'...")
    shutil.copy2(args.filepath, destination)
    print(f"[*] Program exported to '{destination}'")


if __name__ == "__main__":
    main()
