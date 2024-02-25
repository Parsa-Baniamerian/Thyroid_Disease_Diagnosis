# Thyroid_Disease_Diagnosis
This repository contains a Python implementation of a Multi-Layer Perceptron (MLP) classifier for thyroid disease diagnosis. The classifier utilizes a feedforward neural network with one hidden layer to classify thyroid function based on input data.

## Features

- Utilizes a feedforward neural network with one hidden layer.
- Implements backpropagation for training.
- Provides functions for testing and calculating accuracy.
- Includes a plotting function to visualize training history.

## Installation

To use this code, you need to have Python installed on your system. Additionally, the following Python libraries are required:

- numpy
- pandas
- matplotlib

You can install these libraries using pip:
```bash
pip install numpy pandas matplotlib
```

## Usage

1. Clone the repository to your local machine:
```bash
git clone https://github.com/Parsa-Baniamerian/Thyroid_Disease_Diagnosis.git
```

2. Navigate to the repository directory:
```bash
cd MLP_Thyroid
```

4. Run the main script:
```bash
python main.py
```

4. The script will train the neural network using the provided data and display the training loss and validation accuracy over epochs. After training, it will calculate the accuracy on the test dataset and display the results.

## Dataset

The dataset used in this project is the "Thyroid Dataset", which is commonly used in machine learning and originates from MATLAB.

- **Description**: The Thyroid Dataset consists of data related to thyroid gland function. It is categorized into three classes:
  1. **Normal, not hyperthyroid**
  2. **Hyperfunction**
  3. **Subnormal functioning**

The dataset used for training, validation, and testing is provided in Excel format:

- `thyroidInputs.xlsx`: Input data for the neural network.
- `thyroidTargets.xlsx`: Target labels for the input data.

The dataset is divided into three parts:

- **Training Data**: Used to train the neural network.
- **Validation Data**: Used to tune hyperparameters and avoid overfitting.
- **Test Data**: Used to evaluate the final performance of the trained model.

## Example

An example usage of the classifier is provided in the `main()` function of the script. You can modify this function or use it as a template to integrate the classifier into your own projects.

