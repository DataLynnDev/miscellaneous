# **Values to Users**
    - multi-class Image Classification task
    - CNN
    - padding, stride
    - max pooling
    - dropout

In this notebook, we do image classification on the MNIST dataset. We introduce the key concepts of Convolutional Neural Network (CNN), including convolutional layer, padding, stride, max pooling layers and dropout. Comparing with our benchmark model (simple CNN), we improves the classification accuracy from $98.28\%$ to $99.11\%$.

# **Enhancement Roadmap** 
    - Baseline CNN (padding and stride): 98.28%
    - Baseline CNN with max pooling: 98.89%
    - Baseline CNN with max pooling and dropout: 99.11%

# **Data Set Information:**

The MNIST dataset (Modified National Institute of Standards and Technology) consists of $70,000$ handwritten digit images, each of which is $28 \times 28$ pixels in size. These images are grayscale, and each pixel value ranges from $0$ (black) to $255$ (white). The dataset is divided into two parts: $60,000$ images for training and $10,000$ images for testing. Each image is labeled with the digit it represents, which ranges from $0$ to $9$. 

**Model Type:**

Classification

# **Solution Overview**

## 1. Data Preparation

### 1.1 load data

### 1.2 Visualize the dataset

## 2. Convolutional Neural Networks (CNN)

The goal is to build a model that successfully classifies hand-written digits. To do this, we will use a **convolutional neural network (CNN)**, which is a particular kind of neural network commonly used for computer vision. CNNs are just like the normal feed-forward networks from the Neural Network Worksheet, except that they have some extra layers like convolutional layers, max-pooling layer, and so on.

Now, we'll walk through the different types of layers typically present in a CNN.

### 2.1 Key Component: Fully connected layers

In a fully connected layer, each unit computes a weighted sum over all the input units and applies a non-linear function to this weighted sum. You have used such layers many times already in the previous worksheet. As you have already seen, these are implemented in PyTorch using the `nn.Linear` class. Usually, you will use the fully connected layers at the end of the model pipeline. The following is similar to the one you have seen from the Neural Network Worksheet.

### 2.2 Key Component: Convolutional layers, Padding, Stride

In a convolutional layer, each unit computes a weighted sum over a two-dimensional $K \times K$ patch of inputs. The units are arranged in **channels** (see figure below), whereby units in the same channel compute the same weighted sum over different parts of the input, using the weights of that channel's **convolutional filter (or kernel)**. The output of a convolutional layer is thus a three-dimensional tensor of shape $C^{out} \times H \times W$, where $C^{out}$ is the number of channels (i.e. the number of convolutional filters/kernels), and $H$ and $W$ are the height and width of the input.

![1](https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/tutorials/static/convnet.png)

Such layers can be implemented in Python using the PyTorch class `nn.Conv2d`, which takes the same arguments as `nn.Conv1d` (documentation [here](https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html)).

**Padding**: adding extra pixels of filler around the boundary of our input image, thus increasing the effective size of the image. Typically, we set the values of the extra pixels to zero. In the following figure, we pad a  3×3  input, increasing its size to  5×5 . The corresponding output then increases to a  4×4  matrix. The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation:  0×0+0×1+0×2+0×3=0.

![2](https://github.com/yidezhao/cis520/blob/master/padding.png?raw=true)

**Stride**: Normally, we default to sliding one element at a time. However, sometimes, either for computational efficiency or because we wish to downsample, we move our window more than one element at a time, skipping the intermediate locations. The following figure shows a two-dimensional cross-correlation operation with a stride of 3 vertically and 2 horizontally. The shaded portions are the output elements as well as the input and kernel tensor elements used for the output computation:  0×0+0×1+1×2+2×3=8 ,  0×0+6×1+0×2+0×3=6 . We can see that when the second element of the first column is outputted, the convolution window slides down three rows. The convolution window slides two columns to the right when the second element of the first row is outputted. When the convolution window continues to slide two columns to the right on the input, there is no output because the input element cannot fill the window (unless we add another column of padding).

![3](https://github.com/yidezhao/cis520/blob/master/stride.png?raw=true)

The following is an example of a one convolutional layer CNN. For the convolutional layer, the kernal size is 3*3, the stride is 1 and padding is 1.

Implement a CNN in the following code snippet. The CNN has 2 convolutional layers. The first layer has 1 in_channels, 32 out_channels, kernel size 3, 1 stride and 1 padding. The second layer has 32 in_channels, 64 out_channels. Choose the same kernel size, stride, and padding. 

*HINT: make sure your dimensions fit and flatten the image before passing into dense layer.*

*HINT: Please refer to [this link](https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9) for additional guidance on linear layer dimensionalities if needed.*

### 2.3 Baseline model

## 3. Baseline CNN with Max Pooling

In a max pooling layer, each unit computes the maximum over a small two-dimensional $K^{pool} \times K^{pool}$ patch of inputs. Given a multi-channel input of dimensions $C \times H \times W$, the output of a max pooling layer has dimensions $C \times H^{out} \times W^{out}$, where:
$$
\begin{align}
  H^{out} &= \left\lfloor \frac{H}{K^{pool}} \right\rfloor\\
  W^{out} &= \left\lfloor \frac{W}{K^{pool}} \right\rfloor
\end{align}
$$
$\lfloor\cdot\rfloor$ denotes rounding down to the nearest integer below (i.e. floor division `//` in Python).

Max pooling layers can be implemented with the PyTorch `nn.MaxPool2d` class, which takes as a single argument the size $K^{pool}$ of the pooling patch. Note that we need to calculate the dimensions of its output in order to set the dimensions of the subsequent fully connected layer.

Implement a CNN in the following code snippet. The CNN should have 2 convolutional layers, each followed by a max pooling layer. The first layer has been already implemented for you. The second convolutional layer has 32 in_channels, 64 out_channels. Choose the same kernel size, stride, and padding. Add a max pooling layer after the second convolutional layer. 

*HINT: make sure your dimensions fit and flatten the image before passing into dense layer.*

*HINT: Please refer to [this link](https://towardsdatascience.com/mnist-handwritten-digits-classification-using-a-convolutional-neural-network-cnn-af5fafbc35e9) for additional guidance on linear layer dimensionalities if needed.*

## 4. Baseline CNN with Max Pooling and Dropout
In standard dropout regularization, each layer by normalizing by the fraction of nodes that were retained (not dropped out). In other words, with dropout probability  𝑝 , each intermediate activation  ℎ  is replaced by a random variable  ℎ′  as follows:
$$
\begin{equation}
h^{\prime}=
\begin{cases}
0 & \text { with probability } p \\
\frac{h}{1-p} & \text { otherwise }
\end{cases}
\end{equation}
$$
By design, the expectation remains unchanged, i.e.,  𝐸[ℎ′]=ℎ .


When we apply dropout to a hidden layer, zeroing out each hidden unit with probability  𝑝 , the result can be viewed as a network containing only a subset of the original neurons. In the following figure,  ℎ2  and  ℎ5  are removed. Consequently, the calculation of the outputs no longer depends on  ℎ2  or  ℎ5  and their respective gradient also vanishes when performing backpropagation. In this way, the calculation of the output layer cannot be overly dependent on any one element of  ℎ1, … ,ℎ5.

![4](https://github.com/yidezhao/cis520/blob/master/dropout.png?raw=ture)

Now, test the accuracy of the model on the test set.