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

