# **Values to Users**
    - multi-class Image Classification task
    - CNN
    - Batch Normalization
    - Data Augmentation

In this notebook, we do image classification on the CIFAR-10 dataset, which is a multi-classes classification task. We first create a Convolutional Neural Network model as our benchmark model. Then, by adding Batch Normalization and Data Augmentation, we improve the classification accuracy from $71.75\%$ to $75.84\%$, which has $4.09\%$ improvement.

# **Enhancement Roadmap** 
    - Baseline CNN: 71.75%
    - Enhanced model by adding Batch Normalization and Data Augmentation: $75.84\%$

# **Data Set Information:**
The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains $60000$ $32 \times 32$ color images in $10$ different classes. The $10$ different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are $6000$ images of each class. The dataset is divided into five training batches and one test batch, each with $10000$ images. The test batch contains exactly $1000$ randomly-selected images from each class.

**Model Type:**

Classification

# **Solution Overview**

## 1. Load Data
Load and showcase Data

## 2. Benchmark model

### 2.1 The baseline Convolutional Neural Network(CNN) architecture

- Input layer: Accepts 32x32x3 images
- Convolution layer: 6 filters, filter size 5x5x3, stride 1
- ReLU layer: Accepts and returns a 32x32x6 tensor
- Max pooling layer: Pooling size 2x2, stride 2
- Convolution layer: 12 filters, filter size 5x5x6, stride 1
- ReLU layer: Accepts and returns a 16x16x12 tensor
- Max pooling layer: Pooling size 2x2, stride 2
- Convolution layer: 24 filters, filter size 5x5x12, stride 1
- ReLU layer: Accepts and returns a 8x8x24 tensor
- Max pooling layer: Pooling size 2x2, stride 2
- Fully connected network with 2 hidden layers: Accepts a flattened tensor of dimension 4x4x24 and outputs a 10-dimensional tensor containing the predicted class labels. The layer dimensions should be [384,120,84,10].

Notice how in each layer the number of filters is doubled, while the feature resolution is halved (***make sure you use the appropriate amount of padding to achieve this!***).

### 2.2 Define the CNN model

Use the multi-class cross entropy loss function and the Adam optimizer with a learning rate of $10^{-3}$. Train the network for a total of $20000$ iterations using a batch size of $128$ images. Record the loss as a function of the training iterations, and report the resulting confusion matrix for the test data-set. What do you notice here?

### 2.3 Data normalization

Check the shape of the inputs (***It is really a good habbit, which can prevent many problems!***)

### 2.4 Train the model

### 2.5 Test the model

### 2.6 Report the results -- accuracy and confusion matrix

## 3. Model improvement -- Batch Normalization and Data Augmentation

### 3.1 Batch Normalization
Batch Normalization is a technique used in Convolutional Neural Networks (CNNs), and neural networks in general, to improve their performance and stability. It was introduced by Sergey Ioffe and Christian Szegedy in a 2015 paper titled "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift". The basic idea of Batch Normalization is to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is analogous to the preprocessing step in traditional machine learning where we normalize the input data to make it zero-centered with unit variance. However, in the case of Batch Normalization, this normalization process is not just done at the beginning of the network, but it is carried out at every layer. This has several benefits:
- **Speeds up learning:** The normalization process helps to speed up the learning process by reducing the amount the hidden unit values shift around (covariate shift). 
- **Regularizes the model:** Batch Normalization adds a little bit of noise to your network. In some cases, such as Inception modules, where batch normalization is used at every layer, it can use a smaller dropout or no dropout at all. 
- **Allows higher learning rates:** Gradient descent usually requires small learning rates for the network to converge. As networks get deeper, gradients can get smaller during back propagation, slowing down learning. Since Batch Normalization regulates the values in each layer, higher learning rates can be used. 
- **Reduces the sensitivity to the initial weights:** With Batch Normalization, the network can converge even if the initial weights are not set perfectly.

### 3.2 Data Augmentation

Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks. In the context of Convolutional Neural Networks (CNNs) and other types of image data, data augmentation can involve a range of operations:
- **Image rotation:** Images are rotated by a certain angle. This helps the model to learn to recognize the object at different orientations. 
- **Image flipping:** Images can be flipped horizontally or vertically.
- **Image scaling:** Images can be scaled by a certain amount, either enlarging or shrinking. This can help the model learn to recognize objects of different sizes.
- **Image translation:** Images can be translated, meaning that they are moved to the left, right, up, or down.
- **Brightness and contrast adjustment:** The brightness or contrast of an image can be adjusted, simulating different lighting conditions. 
- **Noise injection:** Noise can be added to the image, helping the model to learn to recognize the object even when the image quality is not perfect. 
- **Cutout:** Random sections of the image are cut out, forcing the network to learn from other parts of the image. 
These techniques can be applied individually or in combination. They make the assumption that the label of the image should not change when these transformations are applied. For instance, a cat is still a cat whether the image is rotated, flipped, or translated. The main advantages of using data augmentation are:
    - It increases the amount of training data: The more data, the better a deep learning model can learn from it. 
    - It reduces overfitting: Since data augmentation creates variations in the training set, it makes the model more robust and less likely to overfit on the training data. 
    - It improves model performance: As the model learns from more diverse data, it can generalize better to new, unseen data.
    These techniques are usually applied on-the-fly during training, meaning that they are applied only during training time and the augmented images are never actually stored, which is space-efficient.

### 3.4 Report the results of improvement model -- accuracy and confusion matrix

We could find that the classification accuracy does improve!