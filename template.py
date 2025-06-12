import os
from pathlib import Path
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: [%(message)s]')

# File and folder list
list_of_files = [
    "__init__.py",
    "config.py",
    "data",
    "src/data_scrap.py",
    "src/data_preprocessing.py",
    "llm_pipeline.py",
    "setup.py",
    "app.py",
    ".env"
]

# Process each path
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # If it's a directory (no file extension), create it
    if filepath.suffix == "" and not filepath.exists():
        os.makedirs(filepath, exist_ok=True)
        logging.info(f"Creating directory: {filepath}")

    # Otherwise, it's a file
    else:
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Creating directory: {filedir} for the file: {filename}")

        # Create the file if it doesn't exist or is empty
        if not filepath.exists() or filepath.stat().st_size == 0:
            with open(filepath, "w") as f:
                pass
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filename} already exists")
