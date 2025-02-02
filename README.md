# ANTARES Data Processing and Neural Network Training

This project processes simulated ANTARES data, prepares it for machine learning, and implements various neural network models for classification tasks. It demonstrates expertise in Python, shell scripting, machine learning, and data preprocessing and provides a complete pipeline from raw data to trained models.

## Table of Contents

1. [Features](#features)
2. [Project Workflow](#project-workflow)
3. [Code Breakdown](#code-breakdown)
4. [Basic Math Explanation](#Basic-Math-Explanation)
5. [Setup Instructions](#setup-instructions)
6. [Future Improvements](#future-improvements)
7. [Contact Information](#contact-information)


## Features

The project processes **.evt** simulation files, extracts and normalizes event hit data, and converts it into structured matrices. For data transformation, it employs Python scripts such as **Matrix.py** and **Prepare_Matrix.py** and efficiently stores matrices using formats like **.npz** or **ROOT**.

Neural network models are implemented using **TensorFlow/Keras**. For classification tasks, both Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs) are available. The ANN uses fully connected layers, while the CNN leverages **Conv2D layers** for spatial feature extraction, making it ideal for image-like data representations.

To streamline data handling, automation is achieved through **Bash scripts**, including **Auto_Read_text.sh** and **Auto_Run_All.sh**, which facilitate batch processing and matrix preparation. Additionally, visualization tools generate performance plots, track accuracy, and save intermediate matrices for analysis.

## Project Workflow

### Data Preparation

The project starts with raw **.evt** simulation files containing event data. These files are parsed and converted into structured matrices using Python scripts. The processed data can be stored as **.npz** or **ROOT** files to optimize storage and retrieval.

### Automation

Shell scripts automate the extraction and transformation of data, handling batch processing efficiently. **Auto_Read_text.sh** ensures dependencies are installed and processes individual files, while **Auto_Run_All.sh** executes batch processing across multiple files.

### Model Training

The ANN model consists of multiple dense layers, with dropout layers to prevent overfitting. It is trained using labeled matrices, tracking accuracy and loss over iterations. The CNN model, designed for spatial data, incorporates **Conv2D layers** and pooling mechanisms to extract hierarchical features. Early stopping and model checkpoints are employed to ensure optimal performance.

## Code Breakdown

### Matrix.py

This script reads **.evt** files, extracts hit data, normalizes values, and structures them into uniform matrices. The output is stored in a suitable format for machine learning.

### Prepare_Matrix.py

This script processes **.evt** files, converts them into **.npz** or **ROOT** formats, and highlights the use of ROOT for efficient scientific data handling.

### Artificial_Neural_Network.py

The ANN model is implemented in this script. It loads matrices from directories, splits data into train/test sets, defines a multi-layer dense network with dropout regularization, and plots training performance metrics.

### Convolutional_Neural_Network.py

This script loads structured matrices and formats them as image-like data for CNN training. It defines a convolutional architecture using **Conv2D** and pooling layers, applies **early stopping**, and saves the best-performing model while generating accuracy and loss plots.

### Shell Scripts

**Auto_Read_text.sh** installs dependencies and processes individual **.evt** files. **Auto_Run_All.sh** automates the parallel processing of multiple files in a given directory, ensuring seamless data extraction and preparation.

## Setup Instructions

To set up the project, ensure **Python 3.x**, TensorFlow, and ROOT are installed. Clone the repository and run the scripts in the specified order to process data and train models.

## Future Improvements

Enhancements could include hyperparameter tuning for model optimization, augmentation techniques to increase dataset robustness, and additional deep learning architectures for improved classification accuracy.

  
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

**Neuron Output Equation** $$y_j^{(l)} = h \left( \sum\limits_{i=1}^{n} \ w_{ji}^{(l)} \ x_i^{(l-1)} \ + \ b_j^{(l)} \right) $$

where:
- $$\( x_i^{(l-1)} \)$$ are the inputs from the previous layer.
- $$\( w_{ji}^{(l)} \)$$ are the weights.
- $$\( b_j^{(l)} \)$$ is the bias term.
- $$\( h(x) \)$$ is an **activation function**.

### 🔹 Activation Functions
Activation functions introduce non-linearity. Common ones include:

- **Sigmoid**:  $$\sigma(x) = rac{1}{1 + e^{-x}}$$
- **Tanh**:   $$tanh(x) = rac{e^x - e^{-x}}{e^x + e^{-x}}$$
- **ReLU**:  $$ReLU(x) = \max(0, x)$$
- **Softmax**:   $$Softmax(x_i) = rac{e^{x_i}}{\sum_j e^{x_j}}$$

### 🔹 Loss Function
A **loss function** measures the difference between the predicted output and the actual target.

Examples:
- **Mean Squared Error (MSE)**:  $$MSE = rac{1}{N} \sum_{i=1}^{N} (y_i - t_i)^2$$
- **Cross-Entropy Loss**:   $$CE = -rac{1}{N} \sum_{i=1}^{N} y_i \log(t_i)$$

### 🔹 Backpropagation & Gradient Descent
To optimize the network, **gradient descent** is used to update weights.

**Gradient Descent Update Rule**  $$w_i (t+1) = w_i (t) - \eta \nabla E(w_i (t))$$

where:
- $$\( \eta \)$$ is the learning rate.
- $$\( E \)$$ is the loss function.
- $$\( \nabla E \)$$ is the gradient.

---

# 2️⃣ Convolutional Neural Networks (CNN)

### 🔹 Convolution Operation
Each convolutional layer applies a **filter (kernel)** of size $$\( h \times w \)$$ to an input tensor.

**Convolutional Layer Output**  $$y_{ij} = \sum_{m=1}^{h} \sum_{n=1}^{w} K_{mn} X_{(i+m)(j+n)}$$

where:
- $$\( X \)$$ is the input matrix.
- $$\( K \)$$ is the kernel (filter).
- $$\( y_{ij} \)$$ is the output feature map.

**Output Size Formula**  $$H_{out} = \frac{H_{in} - h_{kernel} + 2 \times padding}{stride} + 1$$

$$W_{out} = \frac{W_{in} - w_{kernel} + 2 \times padding}{stride} + 1$$

where:
- $$\( H_{in}, W_{in} \)$$ are input dimensions.
- $$\( h_{kernel}, w_{kernel} \)$$ are kernel dimensions.
- **Padding** adds extra zeros to input borders.
- **Stride** determines the step size of the filter.

### 🔹 Pooling Layer
Pooling reduces dimensionality by selecting important features.

**Pooling Output Size**  $$H_{out} = rac{H_{in} - h_{kernel}}{stride} + 1$$

$$W_{out} = rac{W_{in} - w_{kernel}}{stride} + 1$$

### 🔹 Fully Connected Layer
Once convolutional layers extract features, fully connected layers process them for classification.

**Fully Connected Neuron Output**  $$y_j = h \left( \sum_{i=1}^{n} w_{ji} x_i + b_j \right)$$

where:
- $$\( x_i \)$$ are the inputs from the previous layer.
- $$\( w_{ji} \)$$ are the weights.
- $$\( b_j \)$$ is the bias term.
- $$\( h(x) \)$$ is an activation function.

### 🔹 Loss Function for CNNs
For classification tasks, **cross-entropy loss** is typically used:

$$CE = -\sum_{i=1}^{N} y_i \log(t_i)$$

where:
- $$\( y_i \)$$ is the predicted probability.
- $$\( t_i \)$$ is the actual label (one-hot encoded).

### 🔹 Optimization: Backpropagation in CNNs
CNNs use **gradient descent** to update weights based on **backpropagation**, just like ANNs.

**Gradient Update Rule**  $$w_i (t+1) = w_i (t) - \eta \nabla E(w_i (t))$$

where:
- $$\( \eta \)$$ is the learning rate.
- $$\( E \)$$ is the loss function.
- $$\( \nabla E \)$$ is the gradient of the loss with respect to the weights.

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

