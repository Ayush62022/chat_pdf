import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Langchain with OpenAI LLM
llm = OpenAI(api_key=openai_api_key)

def extract_text_from_pdf(pdf_path):
    """Extracts text from each page of the specified PDF."""
    doc = fitz.open(pdf_path)
    texts = [page.get_text() for page in doc]
    doc.close()
    return " ".join(texts)  # Combining text for simplicity

def chunk_text(text, max_length=4000):
    """Splits the text into chunks that are within the model's token limit."""
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1  # +1 for the space

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def ask_question(text, question):
    max_length = 800  # Adjust based on trial and error to fit within token limits
    focused_text = " ".join(text.split()[:max_length])  # Taking a chunk of text
    response = llm.generate(prompts=[f"Answer the following question based on the text: {focused_text}\n\nQuestion: {question}\nAnswer:"])
    # Assuming the response structure is similar to OpenAI's and there's a direct method or attribute for the text
    answer_text = response.choices[0].text if response.choices else "No answer found."
    return answer_text.strip()

def summarize_text(text):
    max_chunk_size = 800  # Adjust based on trial and error
    chunks = chunk_text(text, max_chunk_size)
    summaries = []
    for chunk in chunks:
        response = llm.generate(prompts=[f"Summarize the following text:\n\n{chunk}"])
        # Assuming a similar structure to the OpenAI response
        summary_text = response.choices[0].text if response.choices else "Summary not available."
        summaries.append(summary_text.strip())
    return " ".join(summaries)




def main(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)

    while True:
        print("\nOptions:\n1. Ask a Question\n2. Summarize Text\n3. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            question = input("Enter your question: ")
            print("Answer:", ask_question(pdf_text, question))
        elif choice == "2":
            print("Summary:", summarize_text(pdf_text))
        elif choice == "3":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_langchain_cli_modified.py <pdf_path>")
    else:
        main(sys.argv[1])
