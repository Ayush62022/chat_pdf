import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
import tempfile
# Import the HTML formatting module
from format import generate_custom_html_content

# Load environment variables
load_dotenv()

# Retrieve the OPENAI_API_KEY from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit page configuration
st.set_page_config(page_title="PDF Conversation", layout="wide")

# Title and introduction
st.title("Interact with a PDF Document")
st.markdown("""
This application allows you to upload a PDF document and ask questions based on its content. 
The answers are generated using AI models. Simply upload your PDF and start asking!
""")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def process_pdf(uploaded_file):
    # Temporarily save uploaded file to read it with PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        tmp_filename = tmpfile.name

    # Load a PDF document and split it into pages
    loader = PyPDFLoader(tmp_filename)
    pages = loader.load_and_split()

    return pages

if uploaded_file is not None:
    with st.spinner('Processing the PDF...'):
        pages = process_pdf(uploaded_file)
    st.success('PDF processed successfully!')

    # Use the custom HTML content for the header
    custom_html = generate_custom_html_content()
    st.markdown(custom_html, unsafe_allow_html=True)

    question = st.text_input("Enter your question here:", key="question")

    if question:
        # Initialize and set up the model, embeddings, and chain based on the model choice
        MODEL = "gpt-3.5-turbo"
        if MODEL.startswith("gpt"):
            model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
            embeddings = OpenAIEmbeddings()
        else:
            model = Ollama(model=MODEL)
            embeddings = OllamaEmbeddings(model=MODEL)

        parser = StrOutputParser()

        vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)

        chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question"),
            }
            | prompt
            | model
            | parser
        )

        try:
            with st.spinner('Generating the answer...'):
                answer = chain.invoke({'question': question})
                st.success("Answer generated successfully!")
                st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
