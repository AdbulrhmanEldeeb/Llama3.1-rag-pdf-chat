import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv


# from langchain.memory import ConversationBufferMemory

# memory = ConversationBufferMemory(input_key="input", output_key="answer", memory_key="history")  # Initialize memory

# Load environment variables (API keys, tokens, etc.)
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable certain TensorFlow optimizations

# Set up the Streamlit page configuration with a custom logo and title
my_logo = "app_data\\my_logo_2.png"
st.set_page_config(page_icon=my_logo, page_title="RAG Chat with pdf")

# Set the app title and sidebar
st.title("Chatgroq With Llama3.1")
lama_image = "app_data\\lama.jpeg"
st.sidebar.image(lama_image)
st.sidebar.header("RAG Project using Llama3.1 and groq API")
st.sidebar.markdown(
    'Make sure your PDFs are placed in the "data" folder in the same directory of this app.'
)

# Initialize the language model (LLM) using the Groq API with Llama3.1
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")

# Define the prompt template for question answering
prompt = ChatPromptTemplate.from_template(
    """
Answer the questions based on the provided context.If the provided context does not include the answer, provide an answer based on your knowledge.
{context}
<context>
Questions: {input}
"""
)


# Function to handle vector embedding of documents
def vector_embedding():
    # Check if vectors are already stored in session state
    if "vectors" not in st.session_state:
        start = time.time()  # Track start time

        # Initialize Hugging Face embeddings (pre-trained model)
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Check if the FAISS index file exists
        vector_store_path = "./faiss_vector_store"

        if os.path.exists(vector_store_path):
            # Load FAISS from disk
            st.session_state.vectors = FAISS.load_local(
                vector_store_path,
                st.session_state.embeddings,
                allow_dangerous_deserialization=True,
            )
            st.write("Loaded vector store from disk.")

        else:
            # Load all PDFs from the "./data" folder
            st.session_state.loader = PyPDFDirectoryLoader("./data")
            st.session_state.docs = st.session_state.loader.load()

            # Split the loaded documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            st.session_state.final_documents = (
                st.session_state.text_splitter.split_documents(st.session_state.docs)
            )

            # Create vector embeddings for the split documents
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )

            # Save FAISS index to disk
            st.session_state.vectors.save_local(vector_store_path)
            st.write("Vector store saved to disk.")

        # Display the total processing time
        end = time.time()
        total_time = end - start
        st.write(f"Total time to process documents: {round(total_time/60, 2)} minutes.")


# User input for asking questions
prompt1 = st.text_input("Add your question here and press enter")

# Display the initial status message to inform the user that documents are being processed
info = st.empty()
info.info("Your documents are being processed, please wait a second...⌛")

# Call the function to create vector embeddings for the documents
vector_embedding()

# Once the documents are processed, update the status message
info.info("Vector Store DB is ready. Ask any question from your documents.")

# If the user enters a question
if prompt1:
    # Create a chain to retrieve and process relevant documents using the LLM and prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retrieve the vector store created from the PDF documents
    retriever = st.session_state.vectors.as_retriever()

    # Clear the info box once the processing starts
    info.write("")

    # Create a retrieval chain that combines the document retriever and the language model
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time taken to generate the response
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time:", time.process_time() - start)

    # Display the response (answer to the user's question)
    st.write(response["answer"])

    # Use an expander to show relevant document chunks for similarity search
    with st.expander("Document Similarity Search"):
        # Display relevant document chunks that were used to generate the response
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
