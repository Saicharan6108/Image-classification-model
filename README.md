# Image-classification-model

*COMPANY*: CODTECH IT SOLUTIONS  
*NAME*: MOLIGE SAI CHARAN  
*INTERN ID*: CT04WY43  
*DOMAIN*: MACHINE LEARNING  
*DURATION*: 4 WEEKS  
*MENTOR*: NEELA SANTOSH

Image Classification Using Convolutional Neural Networks (CNNs)
In this project, we built an image classification model using Convolutional Neural Networks (CNNs), one of the most effective deep learning architectures for visual data. The model was developed as part of the internship task assigned by CODTECH, specifically Task-3, which required building a functional CNN-based image classifier using either TensorFlow or PyTorch.

Objective
The goal of this project is to classify images into multiple categories using a neural network that can learn from pixel data. This classification task was conducted on the CIFAR-10 dataset, a well-known benchmark dataset that consists of 60,000 32x32 color images across 10 different classes, including:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

Each class has 6,000 images, ensuring a balanced dataset suitable for deep learning training.

Tools and Libraries Used
Python: Primary programming language used for writing and executing the model.

TensorFlow/Keras: Used for building and training the CNN model. Keras is a high-level API built into TensorFlow for rapid model prototyping.

Matplotlib: Used for plotting the training and validation accuracy curves.

NumPy: Implicitly used through TensorFlow for numerical computation and matrix manipulation.

Jupyter Notebook: Development environment where the model was coded, tested, and visualized.

Dataset Preparation
The CIFAR-10 dataset was loaded directly using the TensorFlow keras.datasets API. The dataset is automatically split into:

50,000 training images

10,000 test images

The images were normalized by scaling pixel values to a range between 0 and 1 by dividing by 255. This normalization helps the neural network converge faster during training.

Model Architecture
The CNN model consists of the following layers:

Conv2D (32 filters) + ReLU activation

MaxPooling2D to reduce spatial dimensions

Conv2D (64 filters) + ReLU activation

MaxPooling2D

Conv2D (64 filters) + ReLU activation

Flatten to convert 2D matrices into a 1D vector

Dense layer (64 units) + ReLU

Output layer with 10 neurons (one for each class), with no activation (as we used from_logits=True in the loss function)

The model was compiled using the Adam optimizer, Sparse Categorical Crossentropy as the loss function, and accuracy as the evaluation metric.

Training and Evaluation
The model was trained for 10 epochs, with validation performed on the test set during training. After training, the model was evaluated using:

Accuracy Score: The percentage of correct predictions out of total test samples.

Validation Accuracy Graph: A plot showing how training and validation accuracy evolved over each epoch, helping to diagnose overfitting or underfitting.

After training, the model was evaluated on the test dataset using model.evaluate(), and the final test accuracy was printed out (typically between 70–75% for this basic model on CIFAR-10).

Conclusion
This project successfully demonstrates the implementation of a Convolutional Neural Network for image classification using the TensorFlow framework. We used a standard dataset (CIFAR-10), preprocessed the images, built a CNN from scratch, trained the model, and evaluated its performance. The pipeline—from loading data to evaluating the model—follows a real-world machine learning workflow and lays the foundation for more advanced computer vision applications.

This project fulfills the requirements of Task-3 from CODTECH Internship and is ready for submission.

#Output

