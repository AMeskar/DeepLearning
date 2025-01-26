import numpy as np
InputFile1 = open("/home/meskar/Desktop/Files_Antares/061752_anue_b_CC_km3.evt")

hiits = []
max   = 1
Norm1 = -10.
Norm2 = -10.
Norm3 = -10.
Norm4 = -10.
Norm5 = -10.
Norm6 = -10.
Norm7 = -10.

for line in InputFile1:
    if line.startswith("total_hits:"):
        if max < int(line.split()[1:][0]):
            max = int(line.split()[1:][0])
        hiits.append(line)
    if line.startswith("hit:") < 20:
        line = line.split()[2:]
        print(line[0])
        line = [float(elem) for elem in line]
        if Norm1 < line[0]: Norm1 = line[0]
        if Norm2 < line[1]: Norm2 = line[1]
        if Norm3 < line[2]: Norm3 = line[2]
        if Norm4 < line[3]: Norm4 = line[3]
        if Norm5 < line[4]: Norm5 = line[4]
        if Norm6 < line[5]: Norm6 = line[5]
        if Norm7 < line[6]: Norm7 = line[6]
        hiits.append(line)
        
for i in hiits:
    if i[0] != "t":
        i[0]=i[0]/Norm1
        i[1]=i[1]/Norm2
        i[2]=i[2]/Norm3
        i[3]=i[3]/Norm4
        i[4]=i[4]/Norm5
        i[5]=i[5]/Norm6
        i[6]=i[6]/Norm7
        
# Part 2:

Matrix = []
VectorOfMatrix = []

for i in hiits:
    if(i[0] != "t"):
        Matrix.append(i)
    if(i[0] == "t"):
        if (np.shape(Matrix) != (0,)): VectorOfMatrix.append(Matrix)
        Matrix = []

NewVectorMatrix = []
for i in VectorOfMatrix:
    NewVectorMatrix.append(np.array(i))

# Part 3: 

def resize_matrix(matrix, max_rows, max_cols):
    new_matrix1 = np.zeros((max_rows, max_cols))
    new_matrix1[:matrix.shape[0], :matrix.shape[1]] = matrix
    return(new_matrix1)

FinalMatrix1 = []
for i in NewVectorMatrix:
    newmatrix = resize_matrix(i, max, max)
    FinalMatrix1.append(newmatrix)
    
def resize_matrix(matrix, max_rows, max_cols):
    new_matrix1 = np.zeros((max_rows, max_cols))
    new_matrix1[:matrix.shape[0], :matrix.shape[1]] = matrix
    return(new_matrix1)

def returnImages(InputFile):
    hiits = []
    max   = 1
    Norm1 = -10.
    Norm2 = -10.
    Norm3 = -10.
    Norm4 = -10.
    Norm5 = -10.
    Norm6 = -10.
    Norm7 = -10.

    for line in InputFile1:
        if line.startswith("total_hits:"):
            if max < int(line.split()[1:][0]):
                max = int(line.split()[1:][0])
            hiits.append(line)
        if line.startswith("hit:"):
            line  = line.split()[2:]
            line  = [float(elem) for elem in line]
            if Norm1 < line[0]: Norm1 = line[0]
            if Norm2 < line[1]: Norm2 = line[1]
            if Norm3 < line[2]: Norm3 = line[2]
            if Norm4 < line[3]: Norm4 = line[3]
            if Norm5 < line[4]: Norm5 = line[4]
            if Norm6 < line[5]: Norm6 = line[5]
            if Norm7 < line[6]: Norm7 = line[6]
            hiits.append(line)

    for i in hiits:
        if i[0] != "t":
            i[0]=i[0]/Norm1
            i[1]=i[1]/Norm2
            i[2]=i[2]/Norm3
            i[3]=i[3]/Norm4
            i[4]=i[4]/Norm5
            i[5]=i[5]/Norm6
            i[6]=i[6]/Norm7


    # Part 2:

    Matrix = []
    VectorOfMatrix = []

    for i in hiits:
        if(i[0] != "t"):
            Matrix.append(i)
        if(i[0] == "t"):
            if (np.shape(Matrix) != (0,)): VectorOfMatrix.append(Matrix)
            Matrix = []

    NewVectorMatrix = []
    for i in VectorOfMatrix:
        NewVectorMatrix.append(np.array(i))

    # Part 3: 
    FinalMatrix1 = []
    for i in NewVectorMatrix:
        newmatrix = resize_matrix(i, max, max)
        FinalMatrix1.append(newmatrix) 
 
import matplotlib.pyplot as plt

cmap = plt.get_cmap('YlOrRd') # Choose a colormap

for i in range(10):
    matrix = FinalMatrix1[i]
    
    plt.figure(figsize=(8, 6), dpi=28)
    plt.imshow(matrix, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'/home/meskar/Desktop/Files_Antares/anue-cc/data_{i}') 
    plt.close() 
