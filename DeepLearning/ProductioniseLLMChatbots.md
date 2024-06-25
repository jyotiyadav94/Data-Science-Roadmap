How to deploy a LLM chatbot
4 ways to productionize and bring the LLM chatbot to the customers

https://mrmaheshrajput.medium.com/how-to-deploy-a-llm-chatbot-7f1e10dd202e

The article presents four ways to deploy a chatbot, focusing on how to present the output to the user.

### 1. Adding an LLM-Powered QA Bot to Discord

Ideal for internal feedback or if your customers use Discord.
Steps involve creating a Discord bot, registering commands, creating a lambda function to handle interactions, and deploying an LLM to a Sagemaker endpoint.

### 2. Hugging Face ChatUI for a Plug-and-Play Solution

Offers a simple way to bring your own model or use a Sagemaker endpoint.
Requires a running MongoDB instance to store chat history.
Set up involves cloning the repository, configuring environment variables, and running the application.

### 3. Using Gradio

Enables quick development of chatbot interfaces for machine learning models.
Leverages a Hugging Face endpoint for text generation.
Requires careful consideration for production environments due to its focus on rapid prototyping.

### 4. Streamlit HugChat Plugin

Suitable if you're already using Streamlit, otherwise not recommended.
Provides a simple way to integrate a chatbot into Streamlit applications.
Requires Hugging Face credentials to use the HugChat model.
Production Setting Considerations

In a production environment, it's crucial for the chatbot to respond solely based on your knowledge base. This involves converting your knowledge base into embeddings, storing them, and using them for efficient retrieval of relevant information. The article provides a link to another resource with more details on this process.

Additional Option: AWS CDK

The article also mentions AWS CDK as a robust solution for deploying chatbots on AWS infrastructure. It provides a comprehensive example with a React-based UI and supports multiple LLMs, conversation history management, and more.


