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

# Create a ROOT file
filePathOut = "../Matrix/"+str(desired_part)+"/"+str(desired_part1)+".root"
output_file = ROOT.TFile(filePathOut, 'RECREATE')

# Save each matrix as a TH2D histogram
for i, matrix in enumerate(matrices):
    # Create a TH2D histogram
    hist_name = f'matrix_{i+1}'
    hist_title = f'Matrix {i+1}'
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    hist = ROOT.TH2D(hist_name, hist_title, num_rows, 0, num_rows, num_cols, 0, num_cols)

    # Normalize each column by dividing by the maximum element in that column
    for j in range(num_cols):
        column = [float(row[j]) for row in matrix]
        max_value = max(column)
        
        # Skip normalization if max_value is zero
        if max_value == 0:
            continue

        normalized_column = [round(value / max_value, 3) for value in column]
        #normalized_column = [value / max_value for value in column]

        # Fill the histogram with normalized column values
        for k, value in enumerate(normalized_column):
            hist.SetBinContent(k+1, j+1, value)

    # Write the histogram to the ROOT file
    hist.Write()

# Close the ROOT file
output_file.Close()

# Close the text file
file.close()
