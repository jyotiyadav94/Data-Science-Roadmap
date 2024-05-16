# 1. What is Deep Learning ? 

Deep learning is a subset of machine learning that uses multi-layered neural networks, called deep neural networks. It is capable of learning complex patterns and relationships within data
Deep learning involves training artificial neural networks with multiple layers (hence "deep") to learn and extract features from data.


Components:
* Input layer: Receives input data.
* Hidden layers: Intermediate layers that perform computations.
* Output layer: Produces the final output.

In a fully connected Deep neural network, there is an input layer and one or more hidden layers connected one after the other. Each neuron receives input from the previous layer neurons or the input layer. The output of one neuron becomes the input to other neurons in the next layer of the network, and this process continues until the final layer produces the output of the network. The layers of the neural network transform the input data through a series of nonlinear transformations, allowing the network to learn complex representations of the input data.

![Alt text](<images/deep-Learning.jpg>)

![Alt text](<images/hidden.jpg>)

input layer, which is the first layer, receives input from external sources and passes it on to the hidden layer, which is the second layer. Each neuron in the hidden layer gets information from the neurons in the previous layer, computes the weighted total, and then transfers it to the neurons in the next layer. These connections are weighted, which means that the impacts of the inputs from the preceding layer are more or less optimized by giving each input a distinct weight. These weights are then adjusted during the training process to enhance the performance of the model

# What are the Types of Neural Networks? 
The most widely used architectures in deep learning are feedforward neural networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs).

* Feedforward neural networks (FNNs) are the simplest type of ANN, with a linear flow of information through the network. FNNs have been widely used for tasks such as image classification, speech recognition, and natural language processing.
Feedforward neural networks (FNNs) are a fundamental type of artificial neural network (ANN) where connections between the neurons do not form cycles. In simpler terms, the information moves in only one direction, forward, from the input nodes through the hidden nodes (if any) to the output nodes.

    * Input Layer: This layer consists of input nodes where the initial data is fed into the network. Each node represents a feature or input variable.
    * Hidden Layers: These are layers between the input and output layers where intermediate processing occurs. Each hidden layer consists of multiple neurons (nodes), and the number of hidden layers and neurons in each layer can vary based on the complexity of the problem.
    * Output Layer: This layer produces the final output of the network based on the input and the parameters learned during training. The number of nodes in the output layer depends on the nature of the task (e.g., classification, regression).

Training:
* Backpropagation: Adjusts network weights based on the difference between predicted and actual outputs.
* Gradient descent: Optimizes the network by minimizing the loss function.

During training, these weights are adjusted based on the error between the predicted output and the actual output, using techniques like backpropagation and gradient descent, to minimize the error and improve the accuracy of the model.


* Convolutional Neural Networks (CNNs) are specifically for image and video recognition tasks. CNNs are able to automatically learn features from the images, which makes them well-suited for tasks such as image classification, object detection, and image segmentation.
They are particularly powerful for tasks involving pattern recognition and image classification due to their ability to capture spatial hierarchies and local patterns in data.

![Alt text](<images/CNN.jpg>)
Convolutional Neural Network consists of multiple layers like the input layer, Convolutional layer, Pooling layer, and fully connected layers. 

    * Convolutional layer - applies filters to the input image to extract features,
      ![Alt text](<images/convolutionlayer.jpg>)


    * Pooling layer - downsamples the image to reduce computation
        ![Alt text](<images/maxpooling.jpg>)

    * Dense Layer - the fully connected layer makes the final prediction

The network learns the optimal filters through backpropagation and gradient descent



* Recurrent Neural Networks (RNNs) are a type of neural network that is able to process sequential data, such as time series and natural language. RNNs are able to maintain an internal state that captures information about the previous inputs, which makes them well-suited for tasks such as speech recognition, natural language processing, and language translation





# 2. Difference between Dense layer & convolutionall layer neural networks ? 
The main difference between a dense layer (fully connected layer) and a convolutional layer lies in their architecture and the way they process input data.

* Dense Layer (Fully Connected Layer):

Accepts an Input Shape - Accepts 1D, 2D, or flattened input

1. Each neuron in a dense layer is connected to every neuron in the previous layer, forming a fully connected structure.
Dense layers are typically used in the later stages of neural networks, after the feature extraction process.
2. They are good at learning global patterns and relationships across the entire input space.
3. Dense layers have a large number of parameters (weights) due to the fully connected nature, which can lead to overfitting if the training data is limited.
4. They are not spatially aware, meaning they don't preserve the spatial relationships in the input data (e.g., pixels in an image).

More suitable for - Regression, classification

* Convolutional Layer:

Accepts an Input Shape - Typically accepts 2D input (e.g., images)

1. Convolutional layers are composed of learnable filters or kernels that are convolved (slid) across the input data.
2. They are particularly effective for processing grid-like data, such as images or sequences, by capturing local patterns and features.
3. Convolutional layers preserve the spatial relationships in the input data by applying the same filter at every position of the input.
4. They have fewer parameters compared to dense layers, making them more efficient and less prone to overfitting when working with high-dimensional data like images.
5. Convolutional layers are often followed by pooling layers, which help reduce the spatial dimensions and introduce translation invariance.

More suitable for - Image recognition, object detection

In summary, dense layers are used for global feature learning and final classification/regression tasks, while convolutional layers are designed for local feature extraction and capturing spatial patterns in data like images or sequences. Convolutional layers are particularly well-suited for processing grid-like data, while dense layers are more general and can be used in various types of neural networks.


# 3. A layer which have 20 input neurons & 15 output nodes what is the structure of our weights?
Total number of weights = (Number of input neurons) * (Number of output neurons) = 20 * 15 = 300.
Each output node has a weight associated with each input neuron, resulting in 15 sets of weights.
Each set of weights consists of 20 individual weights connecting each input neuron to the corresponding output neuron.


# 4. Explain RNN ? 

RNN's are a type of neural networks that are particularly well suited for processing sequential data or data with temporal dependencies.Sequential data includes information like time series, audio signals, or text data where the order of elements matters.
Feedforward neural networks, which process inputs independently or let's say they process it information is processed layer by layer. But RNN's uses a loop cycle on the input information. So  RNNs consider previous information while processing current inputs. RNN's considers both the current & previous data. 

For example: I wake up at 7 AM. In case of Neural Network If we try to predict the next word the model the model may already forgot the information like I wake up at etch. 

But in case of RNN it will also have the information. It would use every output for each word and loop them back so the model would not forget.RNNs would consider the current and previous data input when processing this information. 
They can be trained using backpropagation through time (BPTT).

* The model shares the same weights across all the time steps, which allows the neural network to use the same parameter in each step.
* Having the memory of past input makes RNN suitable for any sequential data.

Disadvantages of RNN's 
* RNN is susceptible to both vanishing and exploding gradients. This is where the gradient result is the near-zero value (vanishing), causing network weight to only be updated for a tiny amount, or the gradient result is so significant (exploding) that it assigns an unrealistic enormous importance to the network.

* Long time of training because of the sequential nature of the model.

* Short-term memory means that the model starts to forget the longer the model is trained. There is an extension of RNN called LSTM to alleviate this problem.


# Explain What are autoencoders in Deep Learning ? Explain the different layers of autoencoders and mention three practical usages of them?

Traditional feedforward neural networks can be great at performing tasks such as classification and regression, but what if we would like to implement solutions such as signal denoising or anomaly detection? 


What are Autoencoders?
Autoencoders are one of the deep learning types used for unsupervised learning. There are key layers of autoencoders, which are the input layer, encoder, bottleneck hidden layer, decoder, and output.

The three layers of the autoencoder are:-

* Encoder - Compresses the input data to an encoded representation which is typically much smaller than the input data.
* Latent Space Representation/ Bottleneck/ Code - Compact summary of the input containing the most important features
* Decoder - Decompresses the knowledge representation and reconstructs the data back from its encoded form. Then a loss function is used at the top to compare the input and output images. NOTE- It's a requirement that the dimensionality of the input and output be the same. Everything in the middle can be played with.

Autoencoders have a wide variety of usage in the real world. The following are some of the popular ones:

Transformers and Big Bird (Autoencoders is one of these components in both algorithms): Text Summarizer, Text Generator
* Anomly detection
* Image compression
* Nonlinear version of PCA


If an Autoencoder is provided with a set of input features completely independent of each other, then it would be really difficult for the model to find a good lower-dimensional representation without losing a great deal of information (lossy compression).

Autoencoders can, therefore, also be considered a dimensionality reduction technique, which compared to traditional techniques such as Principal Component Analysis (PCA), can make use of non-linear transformations to project data in a lower dimensional space. If you are interested in learning more about other Feature Extraction techniques, additional information is available in this feature extraction tutorial..

Additionally, compared to standard data compression algorithms like gzpi, Autoencoders can not be used as general-purpose compression algorithms but are handcrafted to work best just on similar data on which they have been trained on.

Some of the most common hyperparameters that can be tuned when optimizing your Autoencoder are:

* The number of layers for the Encoder and Decoder neural networks
* The number of nodes for each of these layers
* The loss function to use for the optimization process (e.g., binary cross-entropy or mean squared error)
* The size of the latent space (the smaller, the higher the compression, acting, therefore as a regularization mechanism)
* Finally, Autoencoders can be designed to work with different types of data, such as tabular, time-series, or image data, and can, therefore, be designed to use a variety of layers, such as convolutional layers, for image analysis.

Ideally, a well-trained Autoencoder should be responsive enough to adapt to the input data in order to provide a tailor-made response but not so much as to just mimic the input data and not be able to generalize with unseen data (therefore overfitting).

Types of Autoencoders
Over the years, different types of Autoencoders have been developed:

Undercomplete Autoencoder
Sparse Autoencoder
Contractive Autoencoder
Denoising Autoencoder
Convolutional Autoencoder
Variational Autoencoder
Let’s explore each in more detail.

Undercomplete Autoencoder
This is the simplest version of an autoencoder. In this case, we don’t have an explicit regularization mechanism, but we ensure that the size of the bottleneck is always lower than the original input size to avoid overfitting. This type of configuration is typically used as a dimensionality reduction technique (more powerful than PCA since its also able to capture non-linearities in the data).

Sparse Autoencoder
A Sparse Autoencoder is quite similar to an Undercomplete Autoencoder, but their main difference lies in how regularization is applied. In fact, with Sparse Autoencoders, we don’t necessarily have to reduce the dimensions of the bottleneck, but we use a loss function that tries to penalize the model from using all its neurons in the different hidden layers (Figure 2).

This penalty is commonly referred to as a sparsity function, and it's quite different from traditional regularization techniques since it doesn’t focus on penalizing the size of the weights but the number of nodes activated.

Figure 2: Sparse Autoencoder Architecture (Image by Author).

In this way, different nodes could specialize for different input types and be activated/deactivated depending on the specifics of the input data. This sparsity constraint can be induced by using L1 Regularization and KL divergence, effectively preventing the model from overfitting.

Contractive Autoencoder
The main idea behind Contractive Autoencoders is that given some similar inputs, their compressed representation should be quite similar (neighborhoods of inputs should be contracted in small neighborhood of outputs). In mathematical terms, this can be enforced by keeping input hidden layer activations derivatives small when fed similar inputs.

Denoising Autoencoder
With Denoising Autoencoders, the input and output of the model are no longer the same. For example, the model could be fed some low-resolution corrupted images and work for the output to improve the quality of the images. In order to assess the performance of the model and improve it over time, we would then need to have some form of labeled clean image to compare with the model prediction.

Convolutional Autoencoder
To work with image data, Convolutional Autoencoders replace traditional feedforward neural networks with Convolutional Neural Networks for both the encoder and decoder steps. Updating type of loss function, etc., this type of Autoencoder can also be made, for example, Sparse or Denoising, depending on your use case requirements.

Variational Autoencoder
In every type of Autoencoder considered so far, the encoder outputs a single value for each dimension involved. With Variational Autoencoders (VAE), we make this process instead probabilistic, creating a probability distribution for each dimension. The decoder can then sample a value from each distribution describing the different dimensions and construct the input vector, which it can then be used to reconstruct the original input data.

One of the main applications of Variational Autoencoders is for generative tasks. In fact, sampling the latent model from distributions can enable the decoder to create new forms of outputs that were previously not possible using a deterministic approach.

If you are interested in testing an online a Variational Autoencoder trained on the MNIST dataset, you can find a live example.


# What is an activation function and discuss the use of an activation function? Explain three different types of activation functions?
An activation function in a neural network is a mathematical function applied to each neuron's output (or "activation") to determine whether it should be activated or not.It maps the resulting values in between 0 to 1 or -1 to 1 etc(depending upon the function). This function introduces non-linearity into the model, enabling the network to learn complex patterns and make sense of intricate data. Without activation functions, a neural network would essentially be a linear regression model, regardless of the number of layers it has.In other words, activation functions are what make a linear regression model different from a neural network. 

Key Purposes of Activation Functions
Non-linearity: They allow the network to capture non-linear relationships in the data.
Bounded output: Some activation functions provide bounded output, which can make training more stable.
Differentiability: Most activation functions are differentiable, which is crucial for backpropagation in training the network.

There are a lot of activation functions:

1. Sigmoid function
![Alt text](<images/image.png>)

2. Softmax Function
![Alt text](<images/image copy 3.png>)

3. Leaky RELU 
![Alt text](<images/image copy 2.png>)

4. RELU function
![Alt text](<images/image copy.png>)


Selection of Activation Function
* Hidden Layers: ReLU and its variants (Leaky ReLU, PReLU, ELU) are commonly used due to their efficiency and ability to mitigate the vanishing gradient problem.
* Output Layer: Depends on the task:
* Binary Classification: Sigmoid function.
* Multi-Class Classification: Softmax function.
* Regression: Linear activation (identity function).


# You are using a deep neural network for a prediction task. After training your model, you notice that it is strongly overfitting the training set and that the performance on the test isn’t good. What can you do to reduce overfitting?

### Reducing Overfitting in Deep Neural Networks

Overfitting occurs when a deep neural network performs well on training data but poorly on validation or test data. It means the model has learned the noise and specific details of the training data rather than generalizing to unseen data. To reduce overfitting, modifications can be made at three stages: input data, network architecture, and training process.

### Input Data

1. **Feature Availability and Reliability**
   - Ensure all features are correctly gathered and relevant to the prediction task. Reliable and relevant features contribute to better model performance.

2. **Consistent Distribution**
   - Verify that the training, validation, and test datasets share the same distribution. If the validation set distribution differs, the model will struggle to generalize as it encounters patterns it hasn't seen before.

3. **Data Contamination/Leakage**
   - Ensure there's no overlap or leakage of information between the training and validation/test datasets. Leakage can give the model an unfair advantage and lead to overfitting.

4. **Dataset Size**
   - If the dataset is small, consider data augmentation techniques to artificially increase its size. This is particularly useful for image data where transformations (like rotation, flipping, and scaling) can create new training examples.

5. **Balanced Dataset**
   - Ensure the dataset is balanced, meaning all classes are represented equally. Imbalanced datasets can cause the model to be biased towards the majority class.

## Network Architecture

1. **Model Complexity**
   - Overly complex models (with too many layers or neurons) can overfit the training data. Simplify the architecture to match the complexity of the task.

2. **Layer Types**
   - Consider replacing fully connected layers with convolutional and pooling layers, especially for tasks involving spatial data like images. Convolutional layers can capture spatial hierarchies in data better.

3. **Pre-trained Models**
   - Use pre-trained models or transfer learning, which involves fine-tuning models pre-trained on large datasets. This can help when the training data is limited or the task is similar to the one the pre-trained model was trained on.

4. **Regularization**
   - Apply regularization techniques like:
     - **L1 Regularization (Lasso)**: Adds a penalty equal to the absolute value of the magnitude of coefficients.
     - **L2 Regularization (Ridge)**: Adds a penalty equal to the square of the magnitude of coefficients.
     - **Elastic Net**: Combines L1 and L2 regularization.

5. **Dropout**
   - Add dropout layers, which randomly drop neurons during training to prevent the network from becoming too reliant on specific neurons, encouraging more robust feature learning.

6. **Batch Normalization**
   - Add batch normalization layers to stabilize and accelerate training by normalizing the inputs of each layer, reducing the sensitivity to initial weights.

## Training Process

1. **Early Stopping**
   - Monitor the validation loss during training and stop when it no longer decreases. This prevents the model from continuing to learn the noise in the training data.

2. **Restore Best Weights**
   - Use callbacks to restore the model weights from the epoch with the best validation performance. This ensures that the final model is the one that performed best on the validation set.

3. **Cross-Validation**
   - Implement cross-validation to ensure the model's performance is consistent across different subsets of the data, reducing the likelihood of overfitting to any single subset.

4. **Learning Rate Schedules**
   - Adjust the learning rate during training. Starting with a higher learning rate and gradually decreasing it can help the model converge more effectively without overfitting.

By implementing these strategies, you can reduce overfitting and improve the generalization performance of your deep neural network.

# Why should we use Batch Normalization?
Batch normalization is a technique for training very deep neural networks that standardizes the inputs to a layer for each mini-batch.

Usually, a dataset is fed into the network in the form of batches where the distribution of the data differs for every batch size. By doing this, there might be chances of vanishing gradient or exploding gradient when it tries to backpropagate. In order to combat these issues, we can use BN (with irreducible error) layer mostly on the inputs to the layer before the activation function in the previous layer and after fully connected layers.

Batch Normalisation has the following effects on the Neural Network:

* Robust Training of the deeper layers of the network.
* Better covariate-shift proof NN Architecture.
* Has a slight regularisation effect.
* Centred and Controlled values of Activation.
* Tries to Prevent exploding/vanishing gradient.
* Faster Training/Convergence to the minimum loss function

![Alt text](<images/image copy 4.png>)

# How to know whether your model is suffering from the problem of Exploding Gradients?
By taking incremental steps towards the minimal value, the gradient descent algorithm aims to minimize the error. The weights and biases in a neural network are updated using these processes. However, at times, the steps grow excessively large, resulting in increased updates to weights and bias terms to the point where the weights overflow (or become NaN, that is, Not a Number). An exploding gradient is the result of this, and it is an unstable method.

There are some subtle signs that you may be suffering from exploding gradients during the training of your network, such as:

The model is unable to get traction on your training data (e g. poor loss).
The model is unstable, resulting in large changes in loss from update to update.
The model loss goes to NaN during training.
If you have these types of problems, you can dig deeper to see if you have a problem with exploding gradients. There are some less subtle signs that you can use to confirm that you have exploding gradients:

The model weights quickly become very large during training.
The model weights go to NaN values during training.
The error gradient values are consistently above 1.0 for each node and layer during training.

# Can you name and explain a few hyperparameters used for training a neural network?

### Understanding Hyperparameters in Neural Networks

Hyperparameters are parameters that affect the performance of a model but are not learned from the data during training. Instead, they are set manually by the user before the training process begins. These parameters influence the training process and the structure of the model.

###  Key Hyperparameters

1. **Number of Nodes**
   - Refers to the number of neurons or units in each layer of the neural network. The choice of the number of nodes can affect the model's capacity to learn complex patterns. Too few nodes might lead to underfitting, while too many might lead to overfitting.

2. **Batch Normalization**
   - Batch normalization is a technique to normalize the inputs of each layer to have a mean of zero and a standard deviation of one. This helps stabilize and accelerate training by reducing internal covariate shift and making the network less sensitive to initialization.

3. **Learning Rate**
   - The learning rate determines the size of the steps the optimization algorithm takes when updating the weights during training. A small learning rate can make training slow, while a large learning rate can cause the training process to overshoot minima.

4. **Dropout Rate**
   - Dropout rate is the fraction of neurons that are randomly turned off during each forward pass. This helps prevent overfitting by ensuring that the network does not become too reliant on specific neurons and forces the network to learn more robust features.

5. **Kernel**
   - In the context of convolutional neural networks (CNNs), a kernel (or filter) is a matrix used to perform convolution operations on input data (such as images). The kernel slides over the input matrix to extract features like edges, textures, and patterns.

6. **Activation Function**
   - The activation function defines how the weighted sum of inputs is transformed into the output of a neuron. Common activation functions include:
     - **Sigmoid**: Outputs values between 0 and 1.
     - **Tanh**: Outputs values between -1 and 1.
     - **ReLU (Rectified Linear Unit)**: Outputs the input directly if positive, otherwise, it outputs zero.
     - **Softmax**: Used in the output layer for multi-class classification, outputs a probability distribution.

7. **Number of Epochs**
   - An epoch is one complete pass through the entire training dataset. The number of epochs is the total number of passes the algorithm makes through the data during training. More epochs can lead to better learning, but too many can cause overfitting.

8. **Batch Size**
   - Batch size is the number of training examples used in one iteration of training. For instance, if the dataset has 1000 records and the batch size is set to 100, the dataset will be divided into 10 batches. Smaller batches make training more stochastic and can help generalize better, while larger batches make the training process more stable and efficient.

9. **Momentum**
   - Momentum is a technique to accelerate gradient descent by adding a fraction of the previous update to the current update. This helps to smooth out oscillations and speed up convergence. It can be particularly useful in navigating the cost function's landscape.

10. **Optimizers**
    - Optimizers are algorithms used to adjust the weights and biases of the network to minimize the loss function. They determine how the model updates during training.
      - **Adagrad**: Adapts the learning rate to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters.
      - **Adadelta, RMSProp, Adam**: These optimizers further refine the adaptation of learning rates and incorporate momentum to improve convergence and model performance.

11. **Learning Rate**
    - This is reiterated for emphasis as it is crucial. The learning rate controls how much the weights and biases are adjusted during training after each batch. Optimizers help to adjust this parameter dynamically during training to achieve better results.

## Summary

To achieve a well-trained model, it is essential to carefully tune these hyperparameters. The right combination can significantly enhance the model's performance and generalization ability. Often, finding the best hyperparameters involves experimentation and validation to determine the most effective configuration for the specific task and dataset.
