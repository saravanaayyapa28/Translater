import streamlit as st
from googletrans import Translator
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Generative AI API
genai.configure(api_key=api_key)

# Initialize the Translator
translator = Translator()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("/tmp/faiss_index")

# Function to set up the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Provide a similar answer.
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
# Function to handle user input and provide an answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local("/tmp/faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve relevant documents using similarity search
    docs = faiss_index.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response.get("output_text", "No response generated")
    
    return answer

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF", page_icon="💁")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        answer = user_input(user_question)
        st.write("Reply:", answer)
        
        language_options = {
            "Tamil": "ta",
            "Spanish": "es",
            "Telugu": "te",
            "Hindi": "hi",
            "French": "fr"
        }

        selected_language = st.selectbox(" Select your Translate Language:", options=list(language_options.keys()))

        if st.button("Translate"):
            try:
                translated_text = translator.translate(answer, dest=language_options[selected_language]).text
                st.text_area(f"Translated {selected_language} Text", value=translated_text, height=200)
            except Exception as e:
                st.error(f"Translation error: {e}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete.")
                    else:
                        st.warning("No text extracted from PDF files.")
            else:
                st.warning("Please upload PDF files before processing.")

if __name__ == "__main__":
    main()
