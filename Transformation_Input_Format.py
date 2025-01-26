import numpy as np
import sys
import os
import ROOT 

arg1       = sys.argv[1]
file_path  = arg1 
file  	   = open(file_path)

filename = os.path.basename(file.name)

desired_part=filename.split("_")[1] +"_"+ filename.split("_")[2] +"_"+filename.split("_")[3]

desired_part1=filename.split("_")[0]

matrices = []
current_matrix = []

extract_matrix = False

for line in file:
    if line.startswith('total_hits:'):
        extract_matrix = True
    elif line.startswith('end_event:'):
        if current_matrix:
            matrices.append(current_matrix)
            current_matrix = []
        extract_matrix = False
    elif extract_matrix and line.startswith('hit:'):
        numbers = line.split()[2:9]
        current_matrix.append(numbers)

matrix_arrays = []
for matrix in matrices:
    matrix_array = np.array(matrix, dtype=np.float32)
    matrix_arrays.append(matrix_array)

output_file = "../Matrix/"+str(desired_part)+"/"+str(desired_part1)+".root"

np.savez_compressed(output_file, *matrix_arrays)

file.close()

