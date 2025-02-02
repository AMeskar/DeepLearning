
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
3. [Code Breakdown](https://github.com/AMeskar/DeepLearning?tab=readme-ov-file#code-breakdown)
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
# Mathematical Explanation of ANN and CNN

This document provides a mathematical explanation of **Artificial Neural Networks (ANNs)** and **Convolutional Neural Networks (CNNs)**. It focuses purely on the theoretical mathematical foundations of these networks without details on implementation.

---

# 1️⃣ Artificial Neural Networks (ANN)

### 🔹 Structure of an ANN
An **Artificial Neural Network (ANN)** consists of multiple layers:
- **Input Layer**: Represents the input data.
- **Hidden Layers**: Applies a transformation using weights and biases.
- **Output Layer**: Produces the final prediction.

### 🔹 Forward Propagation
Each neuron in layer \( l \) receives inputs from layer \( l-1 \), applies a weighted sum, adds a bias, and passes it through an activation function.

**Neuron Output Equation** $$y_j^{(l)} = h \left( \sum_{i=1}^{n} w_{ji}^{(l)} x_i^{(l-1)} + b_j^{(l)} \right)$$

where:
- \( x_i^{(l-1)} \) are the inputs from the previous layer.
- \( w_{ji}^{(l)} \) are the weights.
- \( b_j^{(l)} \) is the bias term.
- \( h(x) \) is an **activation function**.

### 🔹 Activation Functions
Activation functions introduce non-linearity. Common ones include:

- **Sigmoid**  $$\sigma(x) = rac{1}{1 + e^{-x}}$$
- **Tanh**  $$tanh(x) = rac{e^x - e^{-x}}{e^x + e^{-x}}$$
- **ReLU**  $$ReLU(x) = \max(0, x)$$
- **Softmax**  $$Softmax(x_i) = rac{e^{x_i}}{\sum_j e^{x_j}}$$

### 🔹 Loss Function
A **loss function** measures the difference between the predicted output and the actual target.

Examples:
- **Mean Squared Error (MSE)**  $$MSE = rac{1}{N} \sum_{i=1}^{N} (y_i - t_i)^2$$
- **Cross-Entropy Loss**  $$CE = -rac{1}{N} \sum_{i=1}^{N} y_i \log(t_i)$$

### 🔹 Backpropagation & Gradient Descent
To optimize the network, **gradient descent** is used to update weights.

**Gradient Descent Update Rule**$$w_i (t+1) = w_i (t) - \eta 
abla E(w_i (t))$$

where:
- \( \eta \) is the learning rate.
- \( E \) is the loss function.
- \( 
abla E \) is the gradient.

---

# 2️⃣ Convolutional Neural Networks (CNN)

### 🔹 Convolution Operation
Each convolutional layer applies a **filter (kernel)** of size \( h 	imes w \) to an input tensor.

**Convolutional Layer Output**$$y_{ij} = \sum_{m=1}^{h} \sum_{n=1}^{w} K_{mn} X_{(i+m)(j+n)}$$

where:
- \( X \) is the input matrix.
- \( K \) is the kernel (filter).
- \( y_{ij} \) is the output feature map.

**Output Size Formula**$$H_{out} = rac{H_{in} - h_{kernel} + 2 	imes padding}{stride} + 1$$

$$W_{out} = rac{W_{in} - w_{kernel} + 2 	imes padding}{stride} + 1$$

where:
- \( H_{in}, W_{in} \) are input dimensions.
- \( h_{kernel}, w_{kernel} \) are kernel dimensions.
- **Padding** adds extra zeros to input borders.
- **Stride** determines the step size of the filter.

### 🔹 Pooling Layer
Pooling reduces dimensionality by selecting important features.

**Pooling Output Size**$$H_{out} = rac{H_{in} - h_{kernel}}{stride} + 1$$

$$W_{out} = rac{W_{in} - w_{kernel}}{stride} + 1$$

### 🔹 Fully Connected Layer
Once convolutional layers extract features, fully connected layers process them for classification.

**Fully Connected Neuron Output**$$y_j = h \left( \sum_{i=1}^{n} w_{ji} x_i + b_j 
ight)$$

where:
- \( x_i \) are the inputs from the previous layer.
- \( w_{ji} \) are the weights.
- \( b_j \) is the bias term.
- \( h(x) \) is an activation function.

### 🔹 Loss Function for CNNs
For classification tasks, **cross-entropy loss** is typically used:

$$CE = -\sum_{i=1}^{N} y_i \log(t_i)$$

where:
- \( y_i \) is the predicted probability.
- \( t_i \) is the actual label (one-hot encoded).

### 🔹 Optimization: Backpropagation in CNNs
CNNs use **gradient descent** to update weights based on **backpropagation**, just like ANNs.

**Gradient Update Rule**$$w_i (t+1) = w_i (t) - \eta 
abla E(w_i (t))$$

where:
- \( \eta \) is the learning rate.
- \( E \) is the loss function.
- \( 
abla E \) is the gradient of the loss with respect to the weights.

---

# 3️⃣ CNN vs ANN: Mathematical Differences

| Feature | ANN | CNN |
|---------|-----|-----|
| **Data Type** | 1D Input (e.g., tabular data) | 2D/3D Input (e.g., images) |
| **Weight Sharing** | No | Yes (Kernels) |
| **Feature Extraction** | Manual | Automatic |
| **Fully Connected** | Yes | No (except final layer) |
| **Computational Efficiency** | Lower | Higher due to fewer parameters |

---

# Conclusion
- **ANNs** use fully connected layers and are suited for structured data.
- **CNNs** leverage **convolutional layers** to extract spatial hierarchies from image data.
- Both use **backpropagation and gradient descent** for optimization.

---

### 📌 License
This document is for educational purposes.

