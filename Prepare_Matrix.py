import h5py
import numpy as np
import matplotlib.pyplot as plt
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

filePathOut = "../Matrix/"+str(desired_part)+"/"+str(desired_part1)+".root"
output_file = ROOT.TFile(filePathOut, 'RECREATE')

for i, matrix in enumerate(matrices):
    
    hist_name = f'matrix_{i+1}'
    hist_title = f'Matrix {i+1}'
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    hist = ROOT.TH2D(hist_name, hist_title, num_rows, 0, num_rows, num_cols, 0, num_cols)

    for j in range(num_cols):
        column = [float(row[j]) for row in matrix]
        max_value = max(column)
        
        if max_value == 0:
            continue

        normalized_column = [round(value / max_value, 3) for value in column]
        #normalized_column = [value / max_value for value in column]

        for k, value in enumerate(normalized_column):
            hist.SetBinContent(k+1, j+1, value)

    hist.Write()

output_file.Close()

file.close()
