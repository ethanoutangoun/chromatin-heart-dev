# Python script to filter .bed.gz URLs from a file

def filter_bedgz_urls(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    bedgz_lines = [line.strip() for line in lines if line.strip().endswith('.bed.gz')]
    
    with open(output_file, 'w') as file:
        for line in bedgz_lines:
            file.write(line + '\n')

input_file = 'tf_files.txt'
output_file = 'tf_urls.txt'  

filter_bedgz_urls(input_file, output_file)
