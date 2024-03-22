import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
# MODEL = "mixtral:8x7b"
# MODEL = "llama2"
from langchain_community.llms import Ollama
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

if MODEL.startswith("gpt"):
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
else:
    model = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings(model=MODEL)

model.invoke("Tell me a joke")

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = model | parser 
chain.invoke("Tell me a joke")

from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

chain = prompt | model | parser

chain.invoke({"context": "My parents named me Santiago", "question": "What's your name'?"})

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:\RAG_for_multiple_pdf\VisionTransformer.pdf")
pages = loader.load_and_split()
len(pages)


from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)

retriever = vectorstore.as_retriever()
retriever.invoke("how could do fine tuning in vision transformer")


from operator import itemgetter

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)



questions = [
    "what is positional embedding   "
    # "How many hours of live sessions?",
    # "How many coding assignments are there in the program?",
    # "Is there a program certificate upon completion?",
    # "What programming language will be used in the program?",
    # "How much does the program cost?",
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {chain.invoke({'question': question})}")
    print()





chain.batch([{"question": q} for q in questions])

for s in chain.stream({"question": "What is the vision transfromer"}):
    print(s, end="", flush=True)