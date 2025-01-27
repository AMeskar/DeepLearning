
# ANTARES Data Processing and Neural Network Training

This project processes simulated ANTARES data, prepares it for machine learning, and implements various neural network models for classification tasks. It demonstrates proficiency in Python, shell scripting, machine learning, and data preprocessing, showcasing an end-to-end pipeline.

The scripts provided achieve the following:

- Extract and normalize data from files.

- Transform data into usable matrices for machine learning.

- Implement Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).

- Automate data preparation using shell scripts.

## Table of Contents

1. [Features](https://github.com/AMeskar/DeepLearning?tab=readme-ov-file#features)
2. [Project Workflow](https://github.com/AMeskar/DeepLearning?tab=readme-ov-file#Project-Workflow)
3. [Code Breakdown]()
4. [Setup Instructions]()
5. [Future Improvements]()
6. [Contact Information]()
## Features

- Data Processing:

  - Parsing **.evt** files to extract event hits.

  - Normalizing and reshaping data into matrices.

- Model Development:

  - Training an ANN and CNN for classification tasks.

  - Leveraging TensorFlow/Keras for deep learning.

- Automation:

  - Bash scripts to streamline file preparation and processing.

- Visualization:

  - Generating performance plots and saving intermediate matrices.
## Project Workflow

1. Data Preparation
 
- Input files: **.evt** simulation files containing event data.

- Conversion to matrices using Python (**Matrix.py**, **Prepare_Matrix.py**).

- Optional storage as .npz or ROOT files for efficient handling.

2. Automation
\
Shell scripts (**Auto_Read_text.sh**, **Auto_Run_All.sh**) to handle batch processing of multiple **.evt** files, automating data extraction, and preparing input matrices.

3. Model Training
\
ANN (Artificial Neural Network):

- Multi-layer dense model for classification.

- Training and validation with accuracy and loss tracking.

CNN (Convolutional Neural Network):

- Conv2D layers for spatial feature extraction.

- Used for image-like matrix data.
## Code Breakdown

1. Matrix.py:

- Reads .evt files and extracts hit data.

- Normalizes and resizes matrices for uniform dimensions.

- Generates and saves final matrices.

2. Prepare_Matrix.py:

- Parses .evt files and generates .npz or .root files.

- Highlights use of ROOT framework for scientific data handling.

3. Artificial_Neural_Network.py:

- Loads matrices from directories.

- Splits data into train/test sets.

- Implements a dense ANN model with dropout layers.

- Plots training curves and saves model performance metrics.

4. Convolutional_Neural_Network.py:

- Loads normalized matrices as image-like data.

- Implements a CNN with Conv2D and pooling layers.

- Monitors model performance using callbacks like early stopping.

- Saves the best model and generates training plots.

5. Shell Scripts:
\
Auto_Read_text.sh:

- Installs dependencies and processes .evt files.

Auto_Run_All.sh:

- Automates parallel processing of files in a directory.
## Basic Math Explanation
