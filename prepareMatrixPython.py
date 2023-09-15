import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import ROOT 

arg1       = sys.argv[1]
file_path  = arg1  # Replace with the actual file path
file  	   = open(file_path)

#name of the file
filename = os.path.basename(file.name)
#extract name the name of the output folder
desired_part=filename.split("_")[1] +"_"+ filename.split("_")[2] +"_"+filename.split("_")[3]
#extract the number of run
desired_part1=filename.split("_")[0]


# Initialize variables
matrices = []
current_matrix = []

# Flag to indicate matrix extraction
extract_matrix = False

# Read the file
for line in file:
    if line.startswith('total_hits:'):
        # Start extracting the matrix
        extract_matrix = True
    elif line.startswith('end_event:'):
        if current_matrix:
            matrices.append(current_matrix)
            current_matrix = []
        # Stop extracting the matrix
        extract_matrix = False
    elif extract_matrix and line.startswith('hit:'):
        numbers = line.split()[2:9]
        current_matrix.append(numbers)

# Create a NumPy array for each matrix
matrix_arrays = []
for matrix in matrices:
    matrix_array = np.array(matrix, dtype=np.float32)
    matrix_arrays.append(matrix_array)

# Save the matrices as compressed NumPy arrays
output_file = "../Matrix/"+str(desired_part)+"/"+str(desired_part1)+".root"

np.savez_compressed(output_file, *matrix_arrays)

# Close the text file
file.close()

