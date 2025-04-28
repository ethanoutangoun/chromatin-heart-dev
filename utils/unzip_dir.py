#!/usr/bin/env python3
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