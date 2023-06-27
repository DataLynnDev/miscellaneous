# **Values to Users**
    - Super resolution
    - CNN
    - SISR
    - Attention mechanism

In this notebook, we take Convolutional Neural Networks (CNN) as our benchmark model to do super resolution task. By adding attention mechanism, we decrease the validation loss from $0.0204$ to $0.0191$.

# **Enhancement Roadmap** 
  - Baseline CNN: $0.0204$
  - CNN with attention mechanism: $0.0191$

# **Concept of Super Resolution**

Super-resolution is a term in the field of image processing and computer vision. It refers to the process of improving the resolution of an image or a set of images. This process essentially reconstructs a high-resolution image from one or more low-resolution images.

The most common application for super-resolution techniques is in video and image enhancement, where details might be obscured due to low resolution. Super-resolution methods can "infer" details that were lost during the original image capture or subsequent compression. This is useful in various fields like medical imaging, satellite imaging, and video surveillance.

There are several types of super-resolution:

- Single-image super-resolution (SISR): This method uses only one low-resolution image to generate a high-resolution counterpart. The deep learning techniques, particularly convolutional neural networks (CNNs), have been used successfully for this task.

- Multi-frame super-resolution: This method uses multiple low-resolution images (which capture the same scene but from slightly different views) to reconstruct a high-resolution image. This technique often relies on subtle differences and movement between the images to infer extra detail.

- Learning-based super-resolution: These methods utilize machine learning or deep learning models that have been trained on pairs of low and high-resolution images. The model learns how to map low-res to high-res images and applies this mapping to new low-res images.

It's important to note that while super-resolution methods can significantly improve perceived image quality, they are not capable of perfectly reconstructing the true high-resolution image. This is due to the fundamental information loss that occurs when an image is reduced in resolution.

**Model Type:**

Prediction

# **Solution Overview**

## 1. Data Preparation

### 1.1 load data

### 1.2 Visualize the dataset

## 2. Benchmark Model - CNN

## 3. Model Improvement - CNN + Attention

### 3.1 Attention mechanism

The attention mechanism is an approach used in various machine learning models to allow them to focus on certain parts of the input data that are more important for a given task. This concept was inspired by the way humans pay "attention" to only a certain portion of their field of vision at any one time.

In the context of deep learning, the attention mechanism is commonly used with recurrent neural networks (RNNs), especially in tasks such as machine translation, text summarization, and speech recognition. For example, in sequence-to-sequence tasks like machine translation, an attention mechanism allows the model to focus on different words in the input sentence at each step of output generation.

As for its application in convolutional neural networks (CNNs) and image processing, attention mechanisms can be used to improve the performance of image classification, object detection, and segmentation tasks. The idea is to make the CNN focus on the most important parts of the image that are most relevant to the task at hand.

One example is the Squeeze-and-Excitation (SE) network which introduces an attention mechanism that adaptively recalibrates channel-wise feature responses by explicitly modeling interdependencies between channels. In other words, it allows the model to pay more attention to certain channels (features) over others.

Another example is the CBAM (Convolutional Block Attention Module) method. It employs both channel and spatial attention to guide the model towards relevant features in both domains.

The key benefit of using attention in CNNs is that it can enhance the model's ability to focus on important features while reducing the influence of less important ones. This can potentially improve model accuracy and generalization capability. However, it's also worth noting that introducing attention mechanisms can increase the complexity of the model and the computational resources required.

### 3.2 Prediction results

Lastly, the validation loss of our enhanced model is $0.0191$. Meanwhile, we show the predicted results of the test set as follows.