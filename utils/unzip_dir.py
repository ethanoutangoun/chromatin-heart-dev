import os
import gzip
import shutil

def unzip_and_cleanup(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename[:-3]) 
            
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            print(f"Unzipped: {filename} -> {filename[:-3]}")

            os.remove(file_path)
            print(f"Deleted zipped file: {filename}")

# Usage example
directory = 'data/transcription_factors'  
unzip_and_cleanup(directory)