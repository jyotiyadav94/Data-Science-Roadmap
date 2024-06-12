# What is NLP ? 

Natural Language Processing (NLP) is a branch of artificial intelligence that deals with the interaction between computers and humans using natural languages, such as English, Spanish, or Mandarin. The goal of NLP is to enable computers to understand, interpret, and generate human language in a way that is valuable and meaningful.
NLP involves several tasks and techniques, including:

1. Natural Language Understanding (NLU): This involves enabling computers to comprehend and interpret human language. It includes tasks such as:

Named Entity Recognition (NER): Identifying and classifying named entities like people, organizations, and locations in text.
Sentiment Analysis: Determining the sentiment (positive, negative, or neutral) expressed in a given text.
Text Summarization: Generating concise summaries of longer texts while preserving the most important information.


2. Natural Language Generation (NLG): This involves enabling computers to generate human-readable text from data or representations. Examples include:

Machine Translation: Translating text from one language to another.
Text Generation: Producing coherent and fluent text based on a given input or prompt.
Dialogue Systems: Building conversational agents or chatbots that can engage in natural language interactions.


* Speech Recognition: Converting spoken language into text.
* Text-to-Speech: Converting written text into synthesized speech.

NLP relies on various techniques from different fields, including machine learning, deep learning, computational linguistics, and information retrieval. Some common approaches used in NLP include:

* Rule-based methods: Developing hand-crafted rules and grammars to analyze and generate language.
Statistical methods: Using probabilistic models and statistical techniques to learn patterns from large datasets.
Deep Learning: Employing neural networks, such as Recurrent Neural Networks (RNNs) and Transformers, to learn representations and patterns from data.

NLP has numerous applications across various domains, including:

* Virtual assistants and chatbots
* Machine translation
* Sentiment analysis for social media monitoring and customer feedback analysis
* Automated question answering systems
* Text summarization for information extraction and content curation
* Spam and fraud detection
* Intelligent writing assistants

As NLP techniques continue to advance, they enable more natural and effective communication between humans and machines, leading to improved user experiences and increased productivity in various industries.
# Recurrent Neural Networks (RNN)

RNN - https://www.kdnuggets.com/comparing-natural-language-processing-techniques-rnns-transformers-bert

- **Purpose and Application:**
  - RNNs are a specific type within the neural network family.
  - They are primarily used for processing sequential data.
  - Sequential data includes information like time series, audio signals, or text data where the order of elements matters.

- **Processing Mechanism:**
  - Unlike regular feed-forward neural networks, RNNs process information differently.
  - In feed-forward networks, information is processed layer by layer.
  - RNNs, on the other hand, use a loop cycle on the input information.
  - This allows RNNs to consider previous information while processing current inputs.

- **Illustration of Differences:**
  - For a visual representation of the differences between RNNs and feed-forward networks, refer to the accompanying image.

![Alt text](<images/RNNs.png>)

RNNs model implements a loop cycle during the information processing. RNNs would consider the current and previous data input when processing this information. That’s why the model is suitable for any type of sequential data.

we have the sentence “I wake up at 7 AM”, and we have the word as input.

If we take an example in the text data, imagine we have the sentence “I wake up at 7 AM”, and we have the word as input. In the feed-forward neural network, when we reach the word “up,” the model would already forget the words “I,” “wake,” and “up.” However, RNNs would use every output for each word and loop them back so the model would not forget.

In the NLP field, RNNs are often used in many textual applications, such as text classification and generation. It’s often used in word-level applications such as Part of Speech tagging, next-word generation, etc.

RNNs model implements a loop cycle during the information processing. RNNs would consider the current and previous data input when processing this information. That’s why the model is suitable for any type of sequential data.

Looking at the RNNs more in-depth on the textual data, there are many types of RNNs. For example, the below image is the many-to-many types.

![Alt text](</images/many-to-mant RNNs.png>)

Another RNN type used in many NLP applications is the encoder-decoder type (Sequence-to-Sequence). The structure is shown in the image below.

![Alt text](<images/encode-decoder.png>)

This structure introduces two parts that are used in the model. The first part is called Encoder, which is a part that receives data sequence and creates a new representation based on it. The representation would be used in the second part of the model, which is the decoder. With this structure, the input and output lengths don’t necessarily need to be equal. The example use case is a language translation, which often does not have the same length between the input and output.

### Advantages:

 - RNN can be used to process text input without length limitations.
 - The model shares the same weights across all the time steps, which allows the neural network to use the same parameter in each step.
 - Having the memory of past input makes RNN suitable for any sequential data.

Traditional neural networks, such as feed-forward neural networks, often struggle with processing sequences of variable lengths, such as sentences or paragraphs of text.
However, Recurrent Neural Networks (RNNs) excel in handling sequences of arbitrary lengths.
This capability is particularly useful in natural language processing tasks where the length of the input text can vary significantly from one instance to another.
RNNs can effectively process sentences, paragraphs, or even entire documents without the need for truncation or padding to fit a fixed input size.
This flexibility in handling variable-length sequences makes RNNs well-suited for a wide range of natural language processing tasks, including language modeling, machine translation, sentiment analysis, and more.


### Disadvantages as well:

 - RNN is susceptible to both vanishing and exploding gradients. This is where the gradient result is the near-zero value (vanishing), causing network weight to only be updated for a tiny amount, or the gradient result is so significant (exploding) that it assigns an unrealistic enormous importance to the network.

 - Long time of training because of the sequential nature of the model.

 - Short-term memory means that the model starts to forget the longer the model is trained. There is an extension of RNN called LSTM to alleviate this problem.

# RNN vs LSTM: Memory and Differences

## Memory in RNNs

**RNNs (Recurrent Neural Networks)**:
- **Memory Capability**: RNNs have a form of memory, as they use their internal state (hidden state) to process sequences of data. This internal state allows them to capture information from previous time steps.
- **Challenges**: RNNs struggle with long-term dependencies due to the vanishing gradient problem, where gradients become very small during backpropagation, making it difficult to learn from distant past data.

**LSTMs (Long Short-Term Memory Networks)**:
- **Memory Capability**: LSTMs are designed to overcome the limitations of RNNs by introducing a more complex memory mechanism. They have cells that can store information for long periods and gates to control the flow of information, which helps in learning long-term dependencies.
- **Advantages**: LSTMs can retain information over longer sequences and are less susceptible to the vanishing gradient problem.

## Differences Between RNN and LSTM

| Feature                          | RNN (Recurrent Neural Network) | LSTM (Long Short-Term Memory)         |
|----------------------------------|---------------------------------|---------------------------------------|
| **Architecture**                 | Simple recurrent structure with one hidden state. | More complex structure with cells, and three gates (input, output, forget). |
| **Memory Capability**            | Limited, short-term memory.     | Enhanced, long-term memory.           |
| **Components**                   | Hidden state.                   | Cell state, hidden state, input gate, forget gate, output gate. |
| **Gradient Issues**              | Prone to vanishing gradient problem. | Designed to mitigate vanishing gradient problem. |
| **Long-Term Dependencies**       | Struggles with learning long-term dependencies. | Effective at learning long-term dependencies. |
| **Complexity**                   | Simpler, fewer parameters.      | More complex, more parameters.        |
| **Training Time**                | Generally faster to train.      | Slower to train due to complexity.    |
| **Usage**                        | Suitable for simple sequences where short-term context is sufficient. | Suitable for complex sequences where long-term context is important. |
| **Applications**                 | Basic sequence modeling tasks, simple time series. | Complex sequence modeling tasks, speech recognition, language modeling, time series prediction. |
| **Memory Gates**                 | No gates for memory control.    | Uses gates to control information flow (input, forget, output gates). |
| **Cell State**                   | No explicit cell state.         | Maintains an explicit cell state for long-term memory. |

## Explanation of LSTM Components

- **Cell State**: The long-term memory of the network that carries information across different time steps. It is modified by the gates.
- **Hidden State**: The short-term memory of the network that is used for the current output.
- **Input Gate**: Controls how much of the new information from the current input and previous hidden state should be added to the cell state.
- **Forget Gate**: Decides how much of the information in the cell state should be discarded.
- **Output Gate**: Determines the output based on the cell state and hidden state.

## Summary

While both RNNs and LSTMs have memory capabilities, LSTMs are designed to handle long-term dependencies more effectively due to their specialized structure with gates and cell states. This makes LSTMs more suitable for tasks that require understanding and retaining long-term context, whereas RNNs might be sufficient for simpler tasks with short-term dependencies.



## What is vanishing gradient problem ? 
The vanishing gradient problem refers to the issue where gradients become very small during the training of deep neural networks, especially in networks with many layers. These tiny gradients make it difficult for the model to learn effectively, as they provide little information for updating the parameters of earlier layers in the network. This can lead to slow or stalled learning and prevents the network from capturing meaningful patterns in the data.

When training a neural network using backpropagation, the gradients of the

loss function 
�
J with respect to the parameters of the model are computed using the chain rule of calculus. In each layer of the network, the gradient of the loss with respect to the activations of that layer is multiplied by the gradient of the activations with respect to the inputs of that layer, resulting in the gradient of the loss with respect to the inputs of that layer. This process is repeated layer by layer, propagating gradients backward through the network.

Mathematically, this can be represented as:



# Vanishing Gradient Problem and Its Mathematical Explanation

The vanishing gradient problem is a phenomenon that occurs during the training of deep neural networks, particularly those with many layers. It arises due to the nature of the backpropagation algorithm, which is used to compute gradients of the loss function with respect to the parameters of the model.

## Problem Description

During backpropagation, gradients are calculated recursively using the chain rule of calculus. The gradients are then used to update the model parameters through optimization algorithms like gradient descent. However, in deep networks, as the gradients are propagated backward through multiple layers, they can diminish exponentially, becoming very small or even vanishing altogether.

### Mathematical Explanation

Let's consider a deep neural network with L layers, and let \( \sigma(z) \) denote the activation function applied element-wise to the input \( z \) of a neuron in a particular layer. The output of the neuron can be written as:

\[ a^{(l)} = \sigma(z^{(l)}) \]

where \( l \) denotes the layer index.

During backpropagation, the gradient of the loss function \( J \) with respect to the parameters of the network is computed recursively using the chain rule:

\[ \frac{\partial J}{\partial w^{(l)}} = \frac{\partial J}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial w^{(l)}} \]

where:
- \( w^{(l)} \) are the weights of layer \( l \),
- \( z^{(l)} \) is the input to layer \( l \),
- \( a^{(l)} \) is the output of layer \( l \),
- \( \frac{\partial J}{\partial a^{(l)}} \) is the gradient of the loss with respect to the output of layer \( l \),
- \( \frac{\partial a^{(l)}}{\partial z^{(l)}} \) is the derivative of the activation function, and
- \( \frac{\partial z^{(l)}}{\partial w^{(l)}} \) is the derivative of the pre-activation.

The vanishing gradient problem occurs when the derivative of the activation function \( \frac{\partial a^{(l)}}{\partial z^{(l)}} \) becomes very small, especially for large or small input values. This causes the overall gradient \( \frac{\partial J}{\partial w^{(l)}} \) to diminish, making it difficult to update the parameters of the earlier layers in the network effectively.

Conclusion

The vanishing gradient problem hinders the training of deep neural networks by causing gradients to become very small as they are propagated backward through multiple layers. This leads to slow or stalled learning and prevents the network from effectively capturing meaningful patterns in the data.



### Archeitecture of RNN: 


# LSTM 

LSTM (Long Short-Term Memory) models are a type of recurrent neural network (RNN) architecture designed to address the vanishing and exploding gradient problems that traditional RNNs face when dealing with long sequences of data.
The key components of an LSTM model are:

* Cell State: This is the horizontal line running through the LSTM, which acts as a conveyor belt that transfers relevant information throughout the sequence. The cell state allows the LSTM to selectively retain or forget information, enabling it to capture long-term dependencies.

* Gates: LSTMs have three gates that regulate the flow of information into and out of the cell state:

  * Forget Gate: Decides what information from the previous cell state should be forgotten or retained.
  * Input Gate: Determines what new information from the current input and previous hidden state should be added to the cell state.
  * Output Gate: Controls what information from the current cell state and input should be used to generate the output and the next hidden state.

These gates are implemented as different types of neural network layers, typically sigmoid and tanh layers, which apply element-wise operations to control the flow of information.
The LSTM architecture works as follows:

The forget gate reads the previous hidden state and the current input, and outputs a value between 0 and 1 for each element in the cell state. A value of 0 means "completely forget," while a value of 1 means "completely keep."
The input gate determines what new information from the current input and previous hidden state will be added to the cell state.
The cell state is updated by forgetting the parts indicated by the forget gate and adding the new information from the input gate.
The output gate decides what parts of the cell state will be used to generate the output and the next hidden state.

This gating mechanism allows LSTMs to selectively retain or forget information, making them better at capturing long-term dependencies compared to traditional RNNs.
LSTMs have been widely successful in various NLP tasks, such as:

* Machine Translation
* Text Generation
* Speech Recognition
* Sentiment Analysis
* Language Modeling

While LSTMs are powerful, they can still struggle with very long sequences and have been largely superseded by more recent architectures like Transformers and attention-based models in many NLP applications.

### LSTM 
Resource - https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/

LSTM (Long Short-Term Memory) is a recurrent neural network (RNN) architecture widely used in Deep Learning. It excels at capturing long-term dependencies, making it ideal for sequence prediction tasks.

Unlike traditional neural networks, LSTM incorporates feedback connections, allowing it to process entire sequences of data, not just individual data points. This makes it highly effective in understanding and predicting patterns in sequential data like time series, text, and speech.

### Archeitecture of LSTM: 
![Alt text](<images/gates.png>)

An LSTM unit that consists of these three gates and a memory cell or lstm cell can be considered as a layer of neurons in traditional feedforward neural network, with each neuron having a hidden layer and a current state.
There are three parts of an LSTM unit are known as gates.

1. Forgot Gate
The first Gate is Forgot Gate.The first part chooses whether the information coming from the previous timestamp is to be remembered or is irrelevant and can be forgotten  

2. Input Gate
The second gate is Input Gate.In the second part, the cell tries to learn new information from the input to this cell.

3. Output Gate
The third Gate is Output Gate. At last, in the third part, the cell passes the updated information from the current timestamp to the next timestamp.
![Alt text](<images/Screenshot 2024-02-22 at 11.27.41.png>)

Just like a simple RNN, an LSTM also has a hidden state where H(t-1) represents the hidden state of the previous timestamp and Ht is the hidden state of the current timestamp. In addition to that, LSTM also has a cell state represented by C(t-1) and C(t) for the previous and current timestamps, respectively.

Here the hidden state is known as Short term memory, and the cell state is known as Long term memory. Refer to the following image.

It is interesting to note that the cell state carries the information along with all the timestamps.
![Alt text](<images/cell state.png>)

Example of LSTM working : 
Here we have two sentences separated by a full stop. The first sentence is “Bob is a nice person,” and the second sentence is “Dan, on the Other hand, is evil”. It is very clear, in the first sentence, we are talking about Bob, and as soon as we encounter the full stop(.), we started talking about Dan.

As we move from the first sentence to the second sentence, our network should realize that we are no more talking about Bob. Now our subject is Dan. Here, the Forget gate of the network allows it to forget about it. Let’s understand the roles played by these gates in LSTM architecture.



LSTM VS RNN 

| Aspect                     | LSTM (Long Short-Term Memory)                                    | RNN (Recurrent Neural Network)                               |
|----------------------------|------------------------------------------------------------------|--------------------------------------------------------------|
| Architecture               | A type of RNN with additional memory cells                       | A basic type of RNN                                          |
| Memory Retention           | Handles long-term dependencies and prevents vanishing gradient   | Struggles with long-term dependencies and vanishing gradient |
| Cell Structure             | Complex cell structure with input, output, and forget gates      | Simple cell structure with only hidden state                 |
| Handling Sequences         | Suitable for processing sequential data                         | Also designed for sequential data, but limited memory        |
| Training Efficiency        | Slower training process due to increased complexity              | Faster training process due to simpler architecture          |
| Performance on Long Sequences | Performs better on long sequences                             | Struggles to retain information on long sequences            |
| Usage                      | Best suited for tasks requiring long-term memory                | Appropriate for simple sequential tasks                      |
| Vanishing Gradient Problem | Addresses the vanishing gradient problem                        | Prone to the vanishing gradient problem                      |

### LSTM Limitations:

- LSTM (Long Short-Term Memory) and similar recurrent neural network (RNN) architectures struggle with capturing long-range dependencies in sequential data.
- Despite being designed to address the vanishing gradient problem, LSTMs still face challenges in effectively capturing dependencies across very long sequences.
- Computational Complexity: LSTMs are computationally more intensive compared to other neural network architectures like feedforward networks or simple RNNs. Training LSTMs can be slower and may require more resources.
- Overfitting: Like other deep learning models, LSTMs are susceptible to overfitting when there is insufficient training data. Regularization techniques like dropout can help mitigate this issue.
- Hyperparameter Tuning: LSTMs have several hyperparameters to tune, such as the number of LSTM units, the learning rate, and the sequence length. Finding the right set of hyperparameters for a specific problem can be a challenging and time-consuming process.
- Limited Interpretability: LSTMs are often considered as “black-box” models, making it challenging to interpret how they arrive at a particular decision. This can be a drawback in applications where interpretability is crucial.
- Long Training Times: Training deep LSTM models on large datasets can be time-consuming and may require powerful hardware, such as GPUs or TPUs.


# Bidirectional LSTMs 

Bidirectional LSTMs (Bi-LSTMs) are a variant of traditional LSTMs (Long Short-Term Memory) networks that process sequential data in both forward and backward directions.
In a regular LSTM, the output at each time step depends only on the current input and the previous hidden state, which means the model can only access past context. However, in many sequence modeling tasks, such as natural language processing, the future context can also be valuable for making predictions or understanding the current input.
Bi-LSTMs address this limitation by introducing two separate LSTM layers: one that processes the sequence in the forward direction (from left to right) and another that processes the sequence in the backward direction (from right to left). The outputs from these two layers are then combined at each time step, allowing the model to capture both past and future context.
The architecture of a Bi-LSTM can be summarized as follows:

* The input sequence is fed into two LSTM layers: one for the forward direction and one for the backward direction.
* The forward LSTM processes the sequence from start to end, updating its hidden state and producing an output at each time step based on the current input and the previous hidden state.
* The backward LSTM processes the sequence in reverse order, from end to start, updating its hidden state and producing an output at each time step based on the current input and the previous hidden state.
* At each time step, the outputs from the forward and backward LSTMs are combined, typically by concatenation or summation, to produce the final output.

Bi-LSTMs are particularly useful in tasks where the entire context is important for making predictions or understanding the input, such as:

Named Entity Recognition (NER)
* Part-of-speech (POS) tagging
* Machine Translation
* Sentiment Analysis
* Speech Recognition

By incorporating both past and future context, Bi-LSTMs can capture more relevant information and make more informed decisions compared to unidirectional LSTMs. However, this added capability comes at the cost of increased computational complexity and training time.
It's worth noting that while Bi-LSTMs have been successful in many sequence modeling tasks, they have been largely superseded by more recent architectures like Transformers and attention-based models in state-of-the-art NLP systems.

# Transformers

Resources - https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/

Transformers are a type of neural network architecture that have revolutionized natural language processing (NLP) and other fields in machine learning since their introduction in the paper "Attention is All You Need" by Vaswani et al. in 2017. Here’s an overview of the key concepts and components:
Transformers have this attention mechanism is a crucial component in improving the performance of encoder-decoder architectures, especially in handling long input sequences.

Transformers is an NLP model architecture that tries to solve the sequence-to-sequence tasks previously encountered in the RNNs. As mentioned above, RNNs have problems with short-term memory. The longer the input, the more prominent the model was in forgetting the information. This is where the attention mechanism could help solve the problem.

The attention mechanism is introduced in the paper by Bahdanau et al. (2014) to solve the long input problem, especially with encoder-decoder type of RNNs. I would not explain the attention mechanism in detail.Basically, it is a layer that allows the model to focus on the critical part of the model input while having the output prediction.
The Transformer was proposed in the paper Attention Is All You Need.

### Attention Mechanism Overview:

- The attention mechanism is essentially a layer within the model that dynamically selects which parts of the input sequence are most relevant for each step of the output sequence generation.
- Instead of relying solely on a fixed-length context vector from the encoder, the attention mechanism enables the model to weigh different parts of the input sequence differently based on their relevance to the current step of generating the output sequence.

### Example:

- Let's consider the task of translating a sentence from English to Indonesian.
- Suppose we have the English input sentence: "The clock is ticking."
- During the translation process, the model needs to generate the corresponding Indonesian output.
- Now, the word "clock" in English could correspond to different words or phrases in Indonesian, depending on the context. For example, it could be translated as "Jam" (which means "clock" in Indonesian) or "Jam kacau" (which means "time is running out" in Indonesian).
- With the attention mechanism, the model can dynamically focus on relevant parts of the input sentence (e.g., the word "clock") while generating the corresponding output in Indonesian.
- In this example, the attention mechanism would likely assign higher weights to the word "clock" in the input sequence when generating the corresponding output in Indonesian, thus capturing the relevant context for accurate translation.


Here, “transduction” means the conversion of input sequences into output sequences. The idea behind Transformer is to handle the dependencies between input and output with attention and recurrence completely.

![Alt text](<images/transformers.png>)

ow focus on the below image. The Encoder block has 1 layer of a Multi-Head Attention followed by another layer of Feed Forward Neural Network. The decoder, on the other hand, has an extra Masked Multi-Head Attention.

The encoder and decoder blocks are actually multiple identical encoders and decoders stacked on top of each other. Both the encoder stack and the decoder stack have the same number of units.


### Limitations of the Transformer
Transformer is undoubtedly a huge improvement over the RNN based seq2seq models. But it comes with its own share of limitations:

* Attention can only deal with fixed-length text strings. The text has to be split into a certain number of segments or chunks before being fed into the system as input

* This chunking of text causes context fragmentation. For example, if a sentence is split from the middle, then a significant amount of context is lost. In other words, the text is split without respecting the sentence or any other semantic boundary

### sequence-to-sequence Examples:

1. Machine Translation:
Input Sequence: A sentence in one language (e.g., English).
Output Sequence: The corresponding translation of the sentence in another language (e.g., French).
Example: Input: "Hello, how are you?" -> Output: "Bonjour, comment ça va ?"


2. Text Summarization:
Input Sequence: A long document or article.
Output Sequence: A concise summary of the document.
Example: Input: Long article about a recent scientific discovery -> Output: Concise summary highlighting key findings.

3. Speech Recognition:
Input Sequence: Audio waveform of spoken words.
Output Sequence: Text transcript of the spoken words.
Example: Input: Audio recording of a person speaking -> Output: Transcription of the spoken words into text.

4. Question Answering:
Input Sequence: A question posed in natural language.
Output Sequence: The corresponding answer to the question.
Example: Input: "Who is the president of the United States?" -> Output: "Joe Biden."

5. Image Captioning:
Input Sequence: A static image.
Output Sequence: A descriptive caption that describes the contents of the image.
Example: Input: Image of a beach scene -> Output: "A group of people enjoying a day at the beach."


Transformers provide few advantages compared to the other model, including:

1. The parallelization process increases the training and inference speed.
2. Capable of processing longer input, which offers a better understanding of the context

There are still some disadvantages to the transformers model:

1. High computational processing and demand.
2. The attention mechanism might require the text to be split because of the length limit it can handle.
3. Context might be lost if the split were done wrong.


# BERT (Bidirectional Search)

### What is BERT?

BERT stands for Bidirectional Encoder Representations from Transformers. It's a type of machine learning model designed for natural language processing tasks like understanding text.
Imagine a sentence: “She plays the violin beautifully.” Traditional language models would process this sentence from left to right, missing the crucial fact that the identity of the instrument (“violin”) impacts the interpretation of the entire sentence. BERT read it from both the sides left and the right. BERT, however, understands that the context-driven relationship between words plays a pivotal role in deriving meaning. It captures the essence of bidirectionality, allowing it to consider the complete context surrounding each word, revolutionizing the accuracy and depth of language understanding.

![Alt text](<images/BERT.png>)

### How BERT Works Conceptually

1. *Text Input*: BERT takes in text as input. This text is broken down into smaller parts called tokens (usually words or subwords).

2. *Tokenization*: The text is tokenized, which means it's split into tokens. Special tokens like [CLS] (for classification) and [SEP] (to separate sentences) are added.

3. *Embedding*: Each token is converted into a numerical vector (embedding). These vectors represent the meaning of the tokens in a way the model can understand.

4. *Transformer Layers*: The tokens pass through multiple transformer layers. These layers help the model understand the context of each token by looking at the tokens around it (both before and after). This is what makes BERT bidirectional.

5. *Attention Mechanism*: Within each transformer layer, an attention mechanism helps the model focus on important tokens in the context. It calculates the importance of each token relative to others.

6. *Output Representation*: After passing through the transformer layers, each token has a new representation that captures its meaning in context. The [CLS] token's representation is often used for tasks like classification.

### Using BERT for Token and Sequence Classification

1. *Token Classification*: This is used for tasks like Named Entity Recognition (NER), where you classify each token (word) in a sentence. 
    - *Process*: Each token's output representation from BERT is fed into a classifier to predict a label (like "Person" or "Location").
    - *Example*: In the sentence "John lives in New York," BERT can classify "John" as a "Person" and "New York" as a "Location."

2. *Sequence Classification*: This is used for tasks like sentiment analysis, where you classify the entire sentence.
    - *Process*: The [CLS] token's representation is fed into a classifier to predict the label (like "Positive" or "Negative").
    - *Example*: In the sentence "I love this movie," BERT can classify it as having a "Positive" sentiment.

### Steps to Use BERT

1. *Load Pre-trained Model*: Use a pre-trained BERT model available from libraries like Hugging Face’s Transformers.

2. *Tokenization*: Tokenize your text using BERT’s tokenizer to prepare it for the model.

3. *Model Input*: Pass the tokenized text into BERT.

4. *Feature Extraction*: Extract the output representations for each token or the [CLS] token for the entire sequence.

5. *Classification Layer*: Add a classification layer on top of BERT’s output to predict labels for tokens or the entire sequence.

6. *Training*: Train the model on your specific dataset if fine-tuning is needed.

### Example of Usage

For token classification (NER):
- Tokenize the sentence.
- Pass tokens through BERT.
- Use the token representations to classify each token.

For sequence classification (Sentiment Analysis):
- Tokenize the sentence.
- Pass tokens through BERT.
- Use the [CLS] token representation to classify the sentence.

BERT has two main objectives during its pre-training phase: *Masked Language Modeling (MLM)* and *Next Sentence Prediction (NSP)*. These objectives help BERT to learn rich representations of text that can be fine-tuned for various downstream tasks.

### 1. Masked Language Modeling (MLM)

*Objective*: To predict a randomly masked token in a sequence.

*How It Works*:
- A portion of the input tokens is randomly masked (replaced with a special [MASK] token).
- The model attempts to predict the original token based on the context provided by the surrounding tokens.

*Example*:
- Input Sentence: "The cat sits on the [MASK]."
- Goal: Predict that [MASK] should be "mat".

This allows BERT to learn bidirectional representations of text, meaning it can understand the context from both the left and the right of a given token.

### 2. Next Sentence Prediction (NSP)

*Objective*: To predict if a given pair of sentences is consecutive in the original text or not.

*How It Works*:
- During training, the model is fed pairs of sentences.
- 50% of the time, the second sentence is the actual next sentence in the text.
- 50% of the time, the second sentence is a random sentence from the corpus.
- The model learns to predict whether the second sentence follows the first one.

*Example*:
- Sentence A: "The cat sits on the mat."
- Sentence B (Positive Example): "It purrs softly."
- Sentence B (Negative Example): "The sky is blue."

*Goal*:
- For the positive example, the model should predict that the second sentence follows the first.
- For the negative example, the model should predict that the second sentence does not follow the first.

### Purpose of These Objectives

1. *MLM*: Enables BERT to understand context from both directions, making it more effective for tasks requiring a deep understanding of language, like question answering and sentiment analysis.

2. *NSP*: Helps BERT understand relationships between sentences, which is crucial for tasks like natural language inference and paragraph-level reasoning.

By training on these objectives, BERT becomes a powerful model capable of handling various NLP tasks through fine-tuning. The pre-training phase allows BERT to develop a strong foundational understanding of language, which can then be specialized to specific tasks with further training.

### What is a Tokenizer?

A Tokenizer is a tool that splits text into smaller pieces called tokens. Tokens can be words, subwords, or even characters. This process is essential for feeding text into models like BERT, which require numerical input rather than raw text.

### How Does Tokenization Work?

1. *Splitting Text*: The tokenizer breaks the input text into tokens. In BERT's case, this often involves splitting on spaces and punctuation.

2. *Mapping to IDs*: Each token is mapped to a unique numerical ID from a predefined vocabulary that the model understands.

3. *Handling Special Tokens*: The tokenizer adds special tokens like [CLS] at the beginning and [SEP] at the end of the input text to help the model understand sentence boundaries and context.

4. *Padding and Truncation*: Sentences are often padded to a fixed length with a [PAD] token or truncated if they exceed a maximum length.

### Example of Tokenization

Let's take the sentence: "Hello, how are you?"

1. *Splitting Text*: 
   - Original sentence: "Hello, how are you?"
   - Tokens: ["Hello", ",", "how", "are", "you", "?"]

2. *Adding Special Tokens*:
   - Tokens with special tokens: ["[CLS]", "Hello", ",", "how", "are", "you", "?", "[SEP]"]

3. *Mapping to IDs*:
   - Each token is mapped to an ID from the vocabulary.
   - Example mapping (IDs are illustrative): [CLS] = 101, "Hello" = 7592, "," = 1010, "how" = 2129, "are" = 2024, "you" = 2017, "?" = 1029, [SEP] = 102

4. *Final Token IDs*:
   - [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]

### Using BERT Tokenizer

Libraries like Hugging Face’s Transformers provide an easy-to-use tokenizer for BERT.

from transformers import BertTokenizer

### Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

###  Example text
text = "Hello, how are you?"

###  Tokenize the text
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

###  Convert tokens to input IDs
input_ids = tokenizer.encode(text, add_special_tokens=True)
print("Input IDs:", input_ids)


### How Tokenizer Works for BERT Tasks

1. *Preparation*: Tokenizer processes the input text, splitting it into tokens and adding special tokens like [CLS] and [SEP].

2. *Conversion*: Each token is converted into a numerical ID based on BERT's vocabulary.

3. *Padding/Truncation*: The sequence is padded or truncated to a fixed length to match the model’s input requirements.

4. *Attention Masks*: Along with input IDs, the tokenizer often generates attention masks, indicating which tokens are actual data and which are padding.

### Examples of Tokenizer Output

For the text "Hello, how are you?":

- *Tokens*: ['hello', ',', 'how', 'are', 'you', '?']
- *Input IDs*: [101, 7592, 1010, 2129, 2024, 2017, 1029, 102]
- *Attention Mask*: [1, 1, 1, 1, 1, 1, 1, 1] (indicating all tokens are actual data, no padding in this short example)

### Working of Tokenizer in BERT

1. *Text Input*: You provide the raw text.
2. *Tokenization*: The tokenizer breaks the text into tokens and adds special tokens.
3. *ID Conversion*: Tokens are converted to numerical IDs.
4. *Attention Masks*: These masks indicate the real tokens versus any padding.
5. *Model Input*: The IDs and masks are fed into BERT for processing.

By handling tokenization, the tokenizer prepares the text in a format that BERT can understand and work with effectively, ensuring that the model can perform tasks like classification or token prediction accurately.

### Token Classification

Token classification is used for tasks like Named Entity Recognition (NER), where each token in a sequence needs a label.

1. *Load Pre-trained BERT Model*: Use a pre-trained BERT model and tokenizer.
2. *Prepare Data*: Tokenize the input text. Ensure that each token in the original text is correctly aligned with the tokens produced by the tokenizer.
3. *Add Classification Layer*: Add a classification layer on top of BERT to predict labels for each token.
4. *Train the Model*: Train the model on your labeled dataset.

#### Example Code
from transformers import BertTokenizer, BertForTokenClassification
import torch

###  Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

###  Example text
text = "Hello, how are you?"

###  Tokenize text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

###  Forward pass (during training)
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits

###  logits now contain the classification scores for each token

### Sequence Classification

Sequence classification is used for tasks like sentiment analysis, where you classify the entire sequence.

1. *Load Pre-trained BERT Model*: Use a pre-trained BERT model and tokenizer.
2. *Prepare Data*: Tokenize the input text. Ensure the [CLS] token is at the beginning of the sequence.
3. *Add Classification Layer*: Add a classification layer on top of BERT to predict a single label for the entire sequence.
4. *Train the Model*: Train the model on your labeled dataset.

#### Example Code

python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

###  Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

###  Example text
text = "I love this movie!"

###  Tokenize text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

###  Forward pass (during training)
outputs = model(input_ids, attention_mask=attention_mask)
logits = outputs.logits

###  logits now contain the classification scores for the entire sequence


Again, during training, you would compute the loss between the logits and the true label for the sequence.

### Using BERT for Different Tasks

The flexibility of BERT allows it to be adapted for various NLP tasks by adding appropriate layers on top of the BERT base. Here are a few common tasks and how to set up BERT for them:

1. *Token Classification*: Add a token-level classification layer to predict labels for each token.
2. *Sequence Classification*: Add a sequence-level classification layer to predict a single label for the entire input text.
3. *Question Answering*: Add layers to predict the start and end positions of the answer span within the text.
4. *Text Generation*: Fine-tune BERT as a language model to generate text sequences.
5. *Text Pair Classification*: Use BERT for tasks like paraphrase identification or entailment, where two text sequences are classified as related or not.

### General Steps for Any Task

1. *Load Pre-trained BERT*: Use the base BERT model and tokenizer from a library like Hugging Face.
2. *Prepare Data*: Tokenize your input text. Ensure special tokens are added correctly.
3. *Add Task-specific Layers*: Depending on the task, add the necessary classification or regression layers on top of BERT.
4. *Train the Model*: Fine-tune the model on your specific dataset by training it with the appropriate loss function and optimizer.
5. *Evaluate and Predict*: Use the trained model to make predictions on new data.

By following these steps, you can adapt a pre-trained BERT model for a variety of NLP tasks, even if you don't know exactly how it was originally trained.


# RO-BERTA 

Resources - https://www.analyticsvidhya.com/blog/2022/10/a-gentle-introduction-to-roberta/

* RoBERTa (A Robustly Optimized BERT Pretraining Approach)
* An improvement on BERT from Facebook AI Research in 2019
* Trained on more data (160GB of text)
* Removes next sentence prediction, trains longer, and dynamically masks words
* Outperforms BERT in several benchmarks


It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.


The key differences between RoBERTa and BERT can be summarized as follows:

- RoBERTa is a reimplementation of BERT with some modifications to the key hyperparameters and minor embedding tweaks. It uses a byte-level BPE as a tokenizer (similar to GPT-2) and a different pretraining scheme.
- RoBERTa is trained for longer sequences, too, i.e. the number of iterations is increased from 100K to 300K and then further to 500K.
- RoBERTa uses larger byte-level BPE vocabulary with 50K subword units instead of character-level BPE vocabulary of size 30K used in BERT.
- In the Masked Language Model (MLM) training objective, RoBERTa employs dynamic masking to generate the masking pattern every time a sequence is fed to the model.
- RoBERTa doesn’t use token_type_ids, and we don’t need to define which token belongs to which segment. Just separate segments with the separation token tokenizer.sep_token (or ).
- The next sentence prediction (NSP) objective is removed from the training procedure.
Larger mini-batches and learning rates are used in RoBERTa’s training.

- This implementation is the same as BertModel with a tiny embeddings tweak as well as a setup for Roberta pretrained models.

- RoBERTa has the same architecture as BERT, but uses a byte-level BPE as a tokenizer (same as GPT-2) and uses a different pretraining scheme.

- RoBERTa doesn’t have token_type_ids, you don’t need to indicate which token belongs to which segment. Just separate your segments with the separation token tokenizer.sep_token (or </s>)

- Same as BERT with better pretraining tricks:

- dynamic masking: tokens are masked differently at each epoch, whereas BERT does it once and for all
together to reach 512 tokens (so the sentences are in an order than may span several documents)
train with larger batches
- use BPE with bytes as a subunit and not characters (because of unicode characters)
- CamemBERT is a wrapper around RoBERTa. Refer to this page for usage examples.


# XLM-ROBERTA

* XLM-RoBERTa (Cross-lingual Language Model Pretraining)
* An adaptation of RoBERTa from Facebook in 2019
* Trained on 2.5TB of filtered CommonCrawl data in 100 languages
* Adds translational language modeling objective during pre-training
* State-of-the-art cross-lingual understanding without any modification





### Transformers archeitecture

Transformers are a type of neural network architecture that was introduced in the 2017 paper "Attention is All You Need" by Vaswani et al. The key innovation of the Transformer architecture is the use of attention mechanisms, which allow the model to dynamically focus on the most relevant parts of the input when producing an output, rather than relying on a fixed, sequential processing of the input.A transformer is a type of artificial intelligence model that learns to understand and generate human-like text by analyzing patterns in large amounts of text data.

![Alt text](<images/transformers.jpeg>)

There are four components of the Transformer architecture:
### Tokenization:
Tokenization is the process of breaking down the input text into smaller, meaningful units called tokens.In the context of Transformers, the input text is typically tokenized into individual words, sub-words, or characters, depending on the specific tokenization approach used.
For example, the sentence "The quick brown fox jumps over the lazy dog" could be tokenized as: ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'].


### Embedding and Position Encoding:
After tokenization, the tokens are converted into numerical representations called embeddings.
Embeddings are learned, dense vectors that capture the semantic and syntactic information of the tokens.In addition to the token embeddings, Transformers also use position embeddings to encode the position of each token in the sequence.The position embeddings are added to the token embeddings, allowing the model to understand the order and relative positions of the tokens in the input.

![Alt text](<images/postional.jpeg>)

### Attention:
The next step is attention. As you learned attention mechanism deals with a very important problem: the problem of context. Sometimes, as you know, the same word can be used with different meanings. This tends to confuse language models, since an embedding simply sends words to vectors, without knowing which definition of the word they’re using.

Attention is a very useful technique that helps language models understand the context. In order to understand how attention works, consider the following two sentences:



Sentence 1: The bank of the river.
Sentence 2: Money in the bank.
As you can see, the word ‘bank’ appears in both, but with different definitions. In sentence 1, we are referring to the land at the side of the river, and in the second one to the institution that holds money. The computer has no idea of this, so we need to somehow inject that knowledge into it. What can help us? Well, it seems that the other words in the sentence can come to our rescue. For the first sentence, the words ‘the’, and ‘of’ do us no good. But the word ‘river’ is the one that is letting us know that we’re talking about the land at the side of the river. Similarly, in sentence 2, the word ‘money’ is the one that is helping us understand that the word ‘bank’ is now referring to the institution that holds money.

![Alt text](<images/attention.jpeg>)

Attention helps give context to each word, based on the other words in the sentece (or text).
In short, what attention does is it moves the words in a sentence (or piece of text) closer in the word embedding. In that way, the word “bank” in the sentence “Money in the bank” will be moved closer to the word “money”. Equivalently, in the sentence “The bank of the river”, the word “bank” will be moved closer to the word “river”. That way, the modified word “bank” in each of the two sentences will carry some of the information of the neighboring words, adding context to it.

The attention step used in transformer models is actually much more powerful, and it’s called multi-head attention. In multi-head attention, several different embeddings are used to modify the vectors and add context to them. Multi-head attention has helped language models reach much higher levels of efficacy when processing and generating text.

### Transformer Block:
The Transformer block is the core building block of the Transformer architecture.
It consists of two main sub-layers:
* Multi-Head Attention: This allows the model to attend to different parts of the input sequence when computing the representation of a given token.
* Feed-Forward Network: A simple, fully-connected neural network that processes each token representation independently.
The Transformer block also includes residual connections and layer normalization to stabilize the training process and improve the model's performance.
#### Encoder and Decoder:
The Transformer architecture consists of an encoder and a decoder.
* The encoder takes the input sequence and produces a sequence of representations, one for each input token.
* The decoder takes the output of the encoder and generates the output sequence one token at a time.
The encoder and decoder both use Transformer blocks, but the decoder also includes a "Masked Multi-Head Attention" sub-layer to prevent the model from attending to future tokens when generating the output.

* Encoder: The encoder takes the input sequence (e.g., a sentence) and produces a sequence of representations, one for each input token. This is done using a stack of encoder layers, each of which has two main sub-layers:
* Multi-Head Attention: This allows the model to attend to different parts of the input sequence when computing the representation of a given token.
Feed-Forward Network: A simple, fully-connected neural network that processes each token representation independently.
* Decoder: The decoder takes the output of the encoder and generates the output sequence (e.g., a translation of the input sentence) one token at a time. The decoder also has a stack of decoder layers, each of which has three main sub-layers:
* Masked Multi-Head Attention: This allows the model to attend to the previously generated output tokens when predicting the next token, but masks out future tokens to prevent the model from "cheating".
* Multi-Head Attention: This allows the decoder to attend to the encoder's output when generating each output token.
* Feed-Forward Network: Similar to the encoder's feed-forward sub-layer.
The key advantage of the Transformer architecture is its ability to capture long-range dependencies in the input and output sequences, thanks to the attention mechanisms. This makes 
Transformers particularly well-suited for tasks like machine translation, language modeling, and text generation, where capturing context and relationships between distant parts of the input or output is crucial.

### Softmax:
Now that you know that a transformer is formed by many layers of transformer blocks, each containing an attention and a feedforward layer, you can think of it as a large neural network that predicts the next word in a sentence. The transformer outputs scores for all the words, where the highest scores are given to the words that are most likely to be next in the sentence.

The last step of a transformer is a softmax layer, which turns these scores into probabilities (that add to 1), where the highest scores correspond to the highest probabilities. Then, we can sample out of these probabilities for the next word. In the example below, the transformer gives the highest probability of 0.5 to “Once”, and probabilities of 0.3 and 0.2 to “Somewhere” and “There”. Once we sample, the word “once” is selected, and that’s the output of the transformer.
![Alt text](<images/softmax.jpg>)

Now what? Well, we repeat the step. We now input the text “Write a story. Once” into the model, and most likely, the output will be “upon”. Repeating this step again and again, the transformer will end up writing a story, such as “Once upon a time, there was a …”.

Softmax is the final activation function used in the Transformer model, particularly in the output layer.It is used to convert the logits (raw output scores) of the model into a probability distribution over the output vocabulary.The softmax function ensures that the output probabilities sum up to 1, making it suitable for classification tasks where the model needs to predict the most likely output token.

Transformers have become a dominant architecture in natural language processing and have also been successfully applied to other domains, such as computer vision and speech recognition. Their flexibility and scalability have made them a popular choice for a wide range of deep learning applications.


### The Challenges with RNNs
RNNs were once the backbone of sequence modeling, designed to process data sequentially and capture temporal dependencies. However, several critical limitations hindered their effectiveness and efficiency:

![Alt text](<images/RNN_Transformers.jpg>)

#### Difficulty in Capturing Long-Term Dependencies
* Vanishing Gradient Problem: RNNs struggle with the vanishing gradient problem, where gradients become exceedingly small during backpropagation, making it challenging to learn correlations between distant elements in a sequence.

* Exploding Gradient Problem: Conversely, gradients can also grow exponentially, leading to the exploding gradient problem, destabilizing the learning process.

#### Sequential Processing Constraints
* Inherent Sequential Nature: The sequential processing nature of RNNs limits their ability to parallelize operations, leading to slower training and inference times, especially for long sequences.

#### Computational and Memory Intensity
* High Computational Load: RNNs, especially variants like LSTMs and GRUs, are computationally intensive due to their complex structures designed to mitigate the vanishing gradient problem.

* Memory Constraints: Managing the hidden states for long sequences demands significant memory, posing challenges for scalability.


## The Transformer Architecture: A Solution
### Parallel Processing and Efficiency
* Self-Attention Mechanism: Unlike RNNs, Transformers use self-attention to weigh the importance of different parts of the input data, allowing for a more nuanced understanding of sequences.
* Parallelization: The Transformer architecture facilitates parallel processing of data, significantly speeding up training and inference times.

### Overcoming Long-Term Dependency Challenges
* Global Context Awareness: Through self-attention, Transformers can consider the entire sequence simultaneously, effectively capturing long-term dependencies without the constraints of sequential processing.

### Scalability and Flexibility
* Reduced Memory Requirements: By eliminating the need for recurrent connections, Transformers require less memory, making them more scalable and efficient.

* Adaptability: The Transformer’s architecture, consisting of stacked encoders and decoders, is highly adaptable to a wide range of tasks beyond NLP, including computer vision and speech recognition.


## Understanding Attention Mechanisms

### Key Components of Attention

* Queries, Keys, and Values: The attention mechanism operates on these three vectors derived from the input data. Queries and keys interact to determine the focus level on different parts of the input, while values carry the actual information to be processed.

* Attention Score: This score measures the relevance between different parts of the input data, guiding the model on where to focus more.

* Self-attention, a specific type of attention mechanism, enables a model to weigh the importance of different parts of the input data relative to each other. It is the cornerstone of the Transformer architecture, allowing it to efficiently process sequences of data in parallel, unlike its predecessors that processed data sequentially.

## How attention works 

Attention mechanism allows models to focus on specific parts of input data.
It assigns weights to different parts of input, emphasizing important features.
Weights are learned during training based on relevance to the task.
Attention helps models make more informed decisions by giving higher importance to relevant information.
It improves performance in tasks like machine translation, summarization, and image captioning.

## Advantages of Attention
* Parallelization: Self-attention allows for the simultaneous processing of all parts of the input data, leading to significant improvements in training speed and efficiency.

* Long-Range Dependencies: It can capture relationships between elements in a sequence regardless of their positional distance, overcoming a major limitation of earlier models like RNNs and LSTMs.


## Understanding Attention and Attention Map

Imagine you’re looking at a picture of a park with lots of dogs and people. Now, if I ask you to find all the yellow balls in the picture, your brain automatically starts to focus on parts of the picture where yellow balls might be, ignoring most of the dogs and people. This focusing is like the attention mechanism in machine learning. It helps the model to focus on important parts of the data (in this case, the yellow balls) that are relevant to the task at hand, ignoring less relevant information (like the dogs and people).

* An attention map is like a map of the picture that shows where you focused your attention when looking for the yellow balls. It would highlight areas with yellow balls and dim down the rest. In machine learning, an attention map visually represents where the model is focusing its attention in the data to make decisions or predictions. So, using the same example, the attention map would highlight the parts of the input (the picture) that the model thinks are important for finding yellow balls, helping us understand why the model makes its decisions.


<img width="842" alt="Screenshot 2024-05-28 at 11 18 07" src="https://github.com/jyotiyadav94/Data-Science-Roadmap/assets/72126242/39ea0c78-c64e-43aa-8647-a35a9cbb803a">


You can focus at a high level on where these processes are taking place.

The transformer architecture is split into two distinct parts, the encoder and the decoder.

These components work in conjunction with each other and they share a number of similarities.

As we all know, Machine learning models are like big computers that understand numbers but not words. So, before we can let these models work with text, we need to turn the words into numbers. This process is called tokenizing. It’s like giving each word a unique number based on a big list (dictionary) of all the words the model knows. This way, the model can understand and work with the text.

Once the text input is represented as a number, this can be passed to the embedding layer.

Every word (which we call a “token”) is turned into a small list of numbers known as a vector. These vectors are special because we can teach the computer to adjust them through a process called training. This means that as the computer learns more from the data it sees, it tweaks these numbers to get better at its job. This process of adjusting the numbers is what we refer to as “trainable vector embedding.” It’s a way of representing words in a form that computers can understand and learn from, improving their ability to process and make sense of text.

Once we have the embedding, we can pass the embedding to the SELF — ATTENTION LAYER. The model analyses the relationship between the tokens in your input sequence.

#### Understanding Multi Head Self Attention
When we talk about processing language or images, there’s a cool technique called “Multi-Head Self Attention” that’s used the Transformer architecture.

Imagine you’re at a busy party, trying to listen to a friend’s story. Your brain automatically picks up on important words they’re saying, while also tuning into the overall noise to catch if someone says your name or if a favourite song starts playing.

Multi-Head Self Attention does something similar for computers. It helps the model to focus on different parts of the sentence or image at the same time, understanding not just the main point but also catching the context and nuances by looking at the information from multiple perspectives.

This technique is like giving the machine learning model a set of specialised lenses to look through. Each “lens” or “head” pays attention to different parts or aspects of the data. So, in a Transformer model, Multi-Head Self Attention allows the model to get a richer understanding of the input.

It’s not just about knowing which word comes next in a sentence; it’s about understanding the whole sentence’s meaning, how each word relates to the others, and even picking up on subtleties like sarcasm or emphasis. This makes Transformers really powerful for tasks like translating languages, summarizing articles, or even generating new text that sounds surprisingly human.

The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. This does not happen just once. There are multiple sets of self-attention weights or heads are learned in parallel. Independent of each other.

#### The Prediction Process
The basic idea is that each part of the self-attention mechanism in a machine learning model looks at different features of language. For instance, one part might understand the connection between characters in a sentence, another part might focus on what action is happening, and a third part might look at something else, like whether the words sound similar. It’s interesting because we don’t decide in advance what these parts, or “heads,” will focus on. They start with random settings, and as we feed them lots of data and give them time to learn, they each pick up different language features on their own. Some of what they learn makes sense to us, like the examples we talked about, but some might be harder to figure out.

After the model has applied all these attention details to the input data, it then processes the output through a network layer that connects everything fully. This produces a list of numbers (logits) that relate to how likely each word from the model’s vocabulary is to be the next word. These logits are then turned into probabilities using a softmax layer, which means every word in the model’s vocabulary gets a score showing how likely it is to come next. There will be thousands of these scores, but usually, one word’s score is higher than the others, making it the model’s top pick for the next word.

#### Final Throughts
The Transformer model has significantly advanced the field of NLP, enabling the development of powerful large language models like GPT and PaLM by providing a more efficient and effective architecture for processing language.

By introducing an innovative approach to sequence modeling, the Transformer model has not only enhanced machine translation but also paved the way for advancements across a broad spectrum of NLP applications, setting a new standard for future research and development in the field.

## Frequently Asked Questions (FAQs)
1. What is the Transformer model?
The Transformer model is a neural network architecture introduced by Google researchers in 2017, focusing on an attention-based mechanism to improve natural language processing tasks.

2. How does the Transformer model differ from RNNs and CNNs?
Unlike RNNs and CNNs, the Transformer uses self-attention mechanisms to process sequences, allowing for more effective parallelization and handling of long-term dependencies.

3. What are the key components of the Transformer architecture?
The Transformer consists of an encoder and a decoder, each made up of layers containing a multi-head self-attention mechanism and a feed-forward neural network.



### Resources

https://jalammar.github.io/illustrated-transformer/

https://medium.com/@amanatulla1606/transformer-architecture-explained-2c49e2257b4c

https://medium.com/@adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09

https://www.datacamp.com/tutorial/how-transformers-work


