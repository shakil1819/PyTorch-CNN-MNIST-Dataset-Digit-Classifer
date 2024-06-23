<h1 align="center">MNIST Neural Network Project</h1>

<p align="center">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/CNN-FF6F61?style=for-the-badge&logo=cnn&logoColor=white" />
<img src="https://img.shields.io/badge/ML-0076A8?style=for-the-badge&logo=ml&logoColor=white" />
<img src="https://img.shields.io/badge/Computer%20Vision-FFCE00?style=for-the-badge&logo=computervision&logoColor=black" />
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=seaborn&logoColor=white" />
<img src="https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
</p>
## Objective

The objective of this project is to build a neural network from scratch to evaluate the MNIST dataset. The MNIST dataset is a collection of handwritten digits and is commonly used as a benchmark for image classification tasks in machine learning. By achieving high accuracy on this dataset, the model demonstrates its ability to perform well on similar classification tasks.

## Task Description

The project involves the following steps:

1. **Dataset Loading and Preprocessing**:
   - Loading the MNIST dataset using `torchvision.datasets`.
   - Applying necessary transformations like converting images to tensors and normalizing them.

2. **Exploratory Data Analysis**:
   - Visualizing a few samples from the dataset to understand its structure and the nature of the images.

3. **Building the Neural Network**:
   - Designing a neural network architecture using `torch.nn` and `torch.nn.functional`.
   - Initializing the model, specifying the loss function, and defining the optimizer.

4. **Training the Model**:
   - Training the neural network on the training dataset.
   - Validating the model on the validation dataset during training.
   - Recording and plotting the training and validation loss over epochs.

5. **Evaluating the Model**:
   - Testing the model on the test dataset to compute the final accuracy.
   - Visualizing the model’s predictions against actual labels.

6. **Improving the Model**:
   - Tweaking hyperparameters like learning rate and training the model again for better accuracy.

7. **Saving the Model**:
   - Saving the trained model for future use.

8. **Sanity Checks**:
   - Loading the saved model and ensuring it performs as expected.
   - Visualizing predictions on random samples from the test dataset.
   - Generating a confusion matrix to evaluate model performance across different classes.

## Tech Stacks Used

- **Python**: The primary programming language used for the project.
- **PyTorch**: Used for building and training the neural network.
- **Torchvision**: Used for loading the MNIST dataset and applying transformations.
- **Matplotlib**: For plotting loss curves and visualizing images.
- **Seaborn**: For plotting the confusion matrix.
- **Scikit-learn**: For generating the confusion matrix.

## Usage

To run the project, ensure you have the necessary libraries installed. You can install the required packages using:

```bash
conda install torch torchvision matplotlib seaborn scikit-learn
```

Then, execute the notebook or script to train the model and evaluate its performance on the MNIST dataset.

## Conclusion

This project provides a comprehensive workflow for building, training, and evaluating a neural network on the MNIST dataset. The final model achieves a high accuracy, demonstrating its effectiveness in classifying handwritten digits. The project also includes steps to improve the model, save it, and perform sanity checks to ensure its reliability.

![image](https://github.com/shakil1819/PyTorch-CNN-MNIST-Dataset-Digit-Classifer/assets/58840439/0b6b7924-1a33-4eef-ab15-54bcbc3ba9c2)
