import synalinks
import os
import glob
import numpy as np

DATASET_FOLDER = "examples/datasets/knowledge_graph"


class Document(synalinks.DataModel):
    filename: str = synalinks.Field(
        description="The document's filename",
    )
    content: str = synalinks.Field(
        description="The document's content",
    )


def load_data(folder: str = DATASET_FOLDER):
    dataset = []

    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Warning: Dataset folder '{folder}' does not exist")
        return np.array(dataset, dtype="object")

    # Define file patterns to search for
    file_patterns = ["*.md", "*.txt"]

    # Iterate through each pattern and find matching files
    for pattern in file_patterns:
        # Use glob to find all files matching the pattern (only in the specified folder)
        file_path_pattern = os.path.join(folder, pattern)
        matching_files = glob.glob(file_path_pattern)

        # Process each matching file
        for file_path in matching_files:
            try:
                # Read file content
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Extract filename from path
                filename = os.path.basename(file_path)

                # Create Document object
                document = Document(filename=filename, content=content)

                dataset.append(document)

            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                continue

    print(f"Loaded {len(dataset)} documents from {folder}")
    return np.array(dataset, dtype="object")
