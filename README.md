# Object_detection_CNN

## Custom CNN Project
This project aims to implement a custom convolutional neural network (CNN) model to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Model Architecture
The custom CNN model is a stack of multiple layers, including convolutional layers, batch normalization layers, max pooling layers, dropout layers, and fully connected layers. The architecture of the model is designed to extract features from the input image, and then use these features for classification.

The first layer of the model is a convolutional layer with 32 filters of size 3x3, a padding of 'same' and a ReLU activation function. This layer is followed by a batch normalization layer. Then, we have another convolutional layer with 32 filters of size 3x3, a padding of 'same' and a ReLU activation function. This layer is followed by a max pooling layer which is used to reduce the spatial size of the data. The next two layers are convolutional layers with 64 filters of size 3x3, a padding of 'same' and a ReLU activation function. These layers are also followed by a max pooling layer.

After this, we have a flatten layer which reshapes the output of the previous layers into a 1D vector. Then, we have a fully connected layer with 512 units and a ReLU activation function. This layer is followed by a dropout layer with a rate of 0.5, which is used to prevent overfitting. Finally, we have a fully connected layer with 10 units and a softmax activation function. This layer corresponds to the output layer of the model, which produces the class probabilities for the input image.

## Training and Evaluation
The model is trained using the Adam optimizer with a learning rate of 1e-3 and a categorical cross-entropy loss function. The training is done for 10 epochs and the performance is evaluated on the test dataset.

The training and validation accuracy achieved by the model is around 88%. The model can be further improved by adding more layers, increasing the number of filters in the convolutional layers, and experimenting with different architectures and hyperparameters.

## Conclusion
In this project, we have implemented a custom CNN model for image classification on the CIFAR-10 dataset. The model is composed of multiple layers, including convolutional layers, batch normalization layers, max pooling layers, dropout layers, and fully connected layers. The architecture of the model is designed to extract features from the input image and use them for classification. The model achieved a training and validation accuracy of around 88%. With further fine-tuning, the model can be improved to achieve better performance.
