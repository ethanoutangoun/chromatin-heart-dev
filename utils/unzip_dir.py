#!/usr/bin/env python3
"""
unzip_dir.py

Unzips all `.gz` files in a specified directory and deletes the original compressed files.
Useful for batch processing zipped datasets such as those from ENCODE or other repositories.

Usage:
    python unzip_dir.py path/to/directory

Inputs:
    - directory: path to a folder containing `.gz` files

Output:
    - Each `.gz` file is extracted in-place and the original `.gz` file is deleted

"""

import os
import gzip
import shutil
import argparse

def unzip_and_cleanup(directory):
    """
    Unzip all .gz files in the given directory and delete the original .gz files.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename[:-3])
            
            with gzip.open(file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            print(f"Unzipped: {filename} -> {filename[:-3]}")
            os.remove(file_path)
            print(f"Deleted zipped file: {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Unzip all .gz files in a directory and remove the originals."
    )
    parser.add_argument(
        "directory",
        help="Path to the directory containing .gz files"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        parser.error(f"Directory not found: {args.directory}")

    unzip_and_cleanup(args.directory)

if __name__ == "__main__":
    main()