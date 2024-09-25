# RAG Chat with PDFs | Llama 3.1 & Groq API
## Overview
This project implements a Retrieval-Augmented Generation (RAG) application that allows users to query and retrieve information from large PDF documents. Powered by Llama 3.1 and Groq API, the system processes the documents, creates vector embeddings for efficient retrieval, and provides intelligent responses based on user queries.

The app supports various languages.

## Key Features
- PDF Document Loading: Ingest and process PDF files placed in a local folder.
- Dynamic Document Splitting: Split large documents into smaller chunks for better query handling and search.
- Vector Embeddings: Use Hugging Face's pre-trained sentence-transformers/all-MiniLM-L6-v2 model for vector embeddings.
- Llama 3.1 Integration: Utilize Llama 3.1's capabilities to generate accurate and context-aware responses based on the retrieved information.
- Document Similarity Search: Retrieve relevant document sections contributing to the generated answers.
- Interactive Streamlit Interface: A user-friendly interface for asking questions, viewing results, and exploring document similarities.
## Technologies Used
- LangChain: A framework for constructing chains of operations like document retrieval and text generation.
- Groq API: To leverage advanced deep learning capabilities for natural language processing.
- Hugging Face: For embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.
- FAISS: For efficient similarity search over document chunks.
- Streamlit: For building a responsive and interactive web interface.
## Prerequisites
Before running the project, ensure you have the following installed:

- Python 3.x
- Required Python libraries (listed in requirements.txt)
- Groq API key
- Hugging Face API token

## Environment Variables
You need to set up the following environment variables in a .env file:
 
    hf_token=YOUR_HUGGINGFACE_API_TOKEN
    groq_api_key=YOUR_GROQ_API_KEY


## Installation
1.Clone the repository:

    git clone https://github.com/AdbulrhmanEldeeb/Llama3.1-rag-pdf-chat.git 
    cd Llama3.1-rag-pdf-chat.git 
2.Install dependencies:

    pip install -r requirements.txt

3.Set up environment variables: Create a .env file in the root directory and add your API tokens:
    
    hf_token=YOUR_HUGGINGFACE_API_TOKEN
    groq_api_key=YOUR_GROQ_API_KEY

4.Prepare your PDF documents: Place the PDF files you want to query inside the data folder.

## Usage 
Run the Streamlit app:

    streamlit run app.py

Access the web interface: Open your web browser and go to http://localhost:8501.

Interact with the App:

Ask questions about the content of the PDFs.
View the answers and relevant document sections.

## How it Works
1.Document Ingestion: PDFs are loaded from the data folder.

2.Document Splitting: The documents are split into manageable chunks using the RecursiveCharacterTextSplitter.

3.Vector Embedding: The chunks are converted into vector embeddings using Hugging Face's pre-trained model (all-MiniLM-L6-v2).

4.Vector Search: FAISS is used to search for relevant chunks based on user queries.

5.Question Answering: Llama 3.1, via Groq API, generates answers from the retrieved document chunks.

## Example
Once the app is running, you can input a question like:

    What is the main argument presented in the introduction of the document?
The app will retrieve relevant chunks from the PDF and generate a response based on the content.