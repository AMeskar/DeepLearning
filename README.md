
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

This document provides a mathematical explanation of **Artificial Neural Networks (ANNs)** and **Convolutional Neural Networks (CNNs)**. It focuses purely on the theoretical mathematical foundations of these networks without any implementation details.

---

## 1️⃣ Artificial Neural Networks (ANN)

### Structure of an ANN
An **Artificial Neural Network (ANN)** consists of multiple layers:
- **Input Layer**: Represents the input data.
- **Hidden Layers**: Apply transformations using weights and biases.
- **Output Layer**: Produces the final prediction.

### Forward Propagation
Each neuron in layer \\( l \\) receives inputs from the previous layer (\\( l-1 \\)), performs a weighted sum, adds a bias, and passes the result through an activation function.

For a given neuron \\( j \\) in layer \\( l \\), the output is computed as:

\\[
y_j^{(l)} = h \\left( \\sum_{i=1}^{n} w_{ji}^{(l)} x_i^{(l-1)} + b_j^{(l)} \\right)
\\]

where:
- \\( x_i^{(l-1)} \\) are the inputs from the previous layer.
- \\( w_{ji}^{(l)} \\) are the weights.
- \\( b_j^{(l)} \\) is the bias.
- \\( h(x) \\) is an **activation function**.

### Activation Functions
Activation functions introduce non-linearity. Common examples include:

- **Sigmoid:** \\( \\sigma(x) = \\frac{1}{1 + e^{-x}} \\)
- **Tanh:** \\( \\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}} \\)
- **ReLU:** \\( ReLU(x) = \\max(0, x) \\)
- **Softmax:** \\( Softmax(x_i) = \\frac{e^{x_i}}{\\sum_j e^{x_j}} \\) (typically used in the output layer for classification)

### Loss Function
A **loss function** quantifies the difference between the predicted output and the actual target.

Examples:
- **Mean Squared Error (MSE):**  
  \\[
  MSE = \\frac{1}{N} \\sum_{i=1}^{N} \\left( y_i - t_i \\right)^2
  \\]
- **Cross-Entropy Loss:**  
  \\[
  CE = -\\frac{1}{N} \\sum_{i=1}^{N} t_i \\log(y_i)
  \\]
  where \\( t_i \\) is typically a one-hot encoded target.

### Backpropagation & Gradient Descent
To optimize the network, weights are updated using **gradient descent** based on the gradient of the loss function with respect to the weights. For a given weight \\( w_i \\), the update rule is:

\\[
w_i(t+1) = w_i(t) - \\eta \\nabla E\\left(w_i(t)\\right)
\\]

where:
- \\( \\eta \\) is the learning rate.
- \\( E \\) is the loss function.
- \\( \\nabla E \\) is the gradient with respect to \\( w_i \\).

---

## 2️⃣ Convolutional Neural Networks (CNN)

### What is a CNN?
A **Convolutional Neural Network (CNN)** is a specialized type of neural network mainly used for image processing. It extracts hierarchical features using convolutional operations.

### Layers in a CNN
1. **Convolutional Layer:** Applies filters (kernels) to extract spatial features.
2. **Pooling Layer:** Reduces the spatial dimensions (width and height) of the feature maps.
3. **Fully-Connected Layer:** Maps the extracted features to the final output classes.

### Convolution Operation
In a convolutional layer, a **kernel (filter)** of size \\( h \\times w \\) is applied to the input tensor. The output at position \\( (i, j) \\) is computed as:

\\[
y_{ij} = \\sum_{m=1}^{h} \\sum_{n=1}^{w} K_{mn} \\; X_{(i+m)(j+n)}
\\]

where:
- \\( X \\) is the input matrix.
- \\( K \\) is the kernel.
- \\( y_{ij} \\) is the output feature map at position \\( (i, j) \\).

The **output dimensions** of a convolutional layer are calculated by:

\\[
H_{out} = \\frac{H_{in} - h_{kernel} + 2 \\times padding}{stride} + 1
\\]

\\[
W_{out} = \\frac{W_{in} - w_{kernel} + 2 \\times padding}{stride} + 1
\\]

where:
- \\( H_{in} \\) and \\( W_{in} \\) are the height and width of the input.
- \\( h_{kernel} \\) and \\( w_{kernel} \\) are the dimensions of the kernel.
- **Padding** adds extra borders to the input.
- **Stride** is the step size for moving the kernel.

### Pooling Layer
Pooling layers reduce the dimensionality of feature maps. Common types include:
- **Max Pooling:** Selects the maximum value within a defined region.
- **Average Pooling:** Computes the average value within a region.

The output size for a pooling operation is given by:

\\[
H_{out} = \\frac{H_{in} - h_{kernel}}{stride} + 1
\\]

\\[
W_{out} = \\frac{W_{in} - w_{kernel}}{stride} + 1
\\]

### Fully-Connected Layer
After convolutional and pooling layers, fully-connected layers process the features for classification. Each neuron is computed as:

\\[
y_j = h \\left( \\sum_{i=1}^{n} w_{ji} x_i + b_j \\right)
\\]

---

## 3️⃣ Optimization: Adam Optimizer

The **Adam optimizer** is an extension of gradient descent that computes adaptive learning rates for each parameter.

### Adam Algorithm Equations
For each parameter \\( \\theta \\), Adam maintains an exponentially decaying average of past gradients (\\( m_t \\)) and squared gradients (\\( v_t \\)):

1. **Gradient Computation:**
   \\[
   g_t = \\nabla_{\\theta} J(\\theta_t)
   \\]

2. **Exponential Moving Averages:**
   \\[
   m_t = \\beta_1 m_{t-1} + (1 - \\beta_1) g_t
   \\]
   \\[
   v_t = \\beta_2 v_{t-1} + (1 - \\beta_2) g_t^2
   \\]

3. **Bias Correction:**
   \\[
   \\hat{m}_t = \\frac{m_t}{1 - \\beta_1^t}
   \\]
   \\[
   \\hat{v}_t = \\frac{v_t}{1 - \\beta_2^t}
   \\]

4. **Parameter Update:**
   \\[
   \\theta_{t+1} = \\theta_t - \\alpha \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon}
   \\]

where:
- \\( \\beta_1 \\) and \\( \\beta_2 \\) are hyperparameters (commonly set to 0.9 and 0.999, respectively).
- \\( \\alpha \\) is the learning rate.
- \\( \\epsilon \\) is a small constant to avoid division by zero (e.g., \\( 10^{-8} \\)).

---

## 4️⃣ Evaluation Metric: Accuracy

**Accuracy** is a common metric used to evaluate the performance of classification models. It is defined as the ratio of correctly predicted examples to the total number of examples:

\\[
\\text{Accuracy} = \\frac{\\text{Number of Correct Predictions}}{\\text{Total Number of Predictions}} = \\frac{1}{N} \\sum_{i=1}^{N} \\mathbb{1}\\{\\hat{y}_i = y_i\\}
\\]

where:
- \\( N \\) is the total number of samples.
- \\( \\hat{y}_i \\) is the predicted class for sample \\( i \\).
- \\( y_i \\) is the true class label for sample \\( i \\).
- \\( \\mathbb{1}\\{\\cdot\\} \\) is the indicator function, which is 1 if the condition is true and 0 otherwise.

---

## Loss Function Examples

Loss functions quantify the error between the predicted output and the true target. Here are some common examples:

### 1. Mean Squared Error (MSE)
The Mean Squared Error is used primarily for regression problems. It measures the average of the squares of the differences between predictions \\(y_i\\) and true values \\(t_i\\):

\\[
MSE = \\frac{1}{N} \\sum_{i=1}^{N} \\left(y_i - t_i\\right)^2
\\]

where:
- \\(N\\) is the number of samples.
- \\(y_i\\) is the predicted value for the \\(i^{th}\\) sample.
- \\(t_i\\) is the true value for the \\(i^{th}\\) sample.

### 2. Mean Absolute Error (MAE)
The Mean Absolute Error measures the average absolute difference between the predicted values and the actual values:

\\[
MAE = \\frac{1}{N} \\sum_{i=1}^{N} \\left| y_i - t_i \\right|
\\]

MAE is less sensitive to outliers than MSE.

### 3. Cross-Entropy Loss
Cross-Entropy Loss is commonly used for classification tasks. It measures the performance of a classification model whose output is a probability value between 0 and 1.

For binary classification, the cross-entropy loss is:

\\[
CE = -\\frac{1}{N} \\sum_{i=1}^{N} \\left[ t_i \\log(y_i) + (1-t_i) \\log(1-y_i) \\right]
\\]

For multi-class classification (with one-hot encoded targets), the loss is:

\\[
CE = -\\frac{1}{N} \\sum_{i=1}^{N} \\sum_{j=1}^{C} t_{ij} \\log(y_{ij})
\\]

where:
- \\(C\\) is the number of classes.
- \\(t_{ij}\\) is the true label (0 or 1) for class \\(j\\) of sample \\(i\\).
- \\(y_{ij}\\) is the predicted probability for class \\(j\\) of sample \\(i\\).

### 4. Hinge Loss
Hinge Loss is commonly used for training Support Vector Machines (SVMs). For a sample \\(i\\) with true label \\(y_i \\in \\{-1, +1\\}\\) and prediction score \\(s_i\\), the hinge loss is given by:

\\[
Hinge\\ Loss = \\max(0, 1 - y_i s_i)
\\]

For a multi-class scenario, the loss for each sample can be written as:

\\[
L_i = \\sum_{j \\neq y_i} \\max\\left(0, s_j - s_{y_i} + \\Delta\\right)
\\]

where:
- \\(s_j\\) is the score for class \\(j\\),
- \\(s_{y_i}\\) is the score for the correct class,
- \\(\\Delta\\) is a margin parameter (commonly set to 1).

---

These examples illustrate different ways to measure error in various types of problems. Choosing the right loss function is crucial as it directly affects the performance and convergence of the learning algorithm.


---

## 5️⃣ Summary of Mathematical Differences

| Aspect                  | ANN                                               | CNN                                                     |
|-------------------------|---------------------------------------------------|---------------------------------------------------------|
| **Input Type**          | Typically 1D data (e.g., tabular data)            | 2D/3D data (e.g., images)                               |
| **Architecture**        | Fully connected layers                           | Convolutional + pooling layers followed by FC layers    |
| **Parameter Sharing**   | No                                               | Yes (via convolutional kernels)                         |
| **Feature Extraction**  | Manual feature engineering is often required     | Automatic hierarchical feature extraction               |
| **Optimization**        | Gradient Descent (with variants like Adam)         | Gradient Descent (with variants like Adam)               |

---
