from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import pandas as pd

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or st.secrets("GOOGLE_API_KEY"))

# Initialize lists to track questions and responses
if 'pdf_question_history' not in st.session_state:
    st.session_state['pdf_question_history'] = []

if 'pdf_response_history' not in st.session_state:
    st.session_state['pdf_response_history'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get QA chain for conversational model
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input for PDF-based Q&A
def user_input(user_question):
    # Check if the user question matches the special case
    if "who made you" in user_question.lower() or "who build you" in user_question.lower():
        response_text = "I am an AI model built by the Tech-Titans group."
        st.write("Reply: ", response_text)
        
        # Update history
        st.session_state['pdf_question_history'].append(user_question)
        st.session_state['pdf_response_history'].append(response_text)
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Attempt to load the FAISS index and handle errors
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        st.error(f"Error loading FAISS index: {e}")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    # Generate and display the response
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response_text = response["output_text"]
        st.write("Reply: ", response_text)
        
        # Update history
        st.session_state['pdf_question_history'].append(user_question)
        st.session_state['pdf_response_history'].append(response_text)
        
    except Exception as e:
        st.error(f"Error generating response: {e}")

# Function to get responses from the Gemini model for chatbot
def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Tech-Titans")
    st.header("Tech-Titans")

    # Sidebar for selecting the mode
    mode = st.sidebar.selectbox("Choose Mode", ["PDF-based Q&A", "Chatbot"])

    if mode == "PDF-based Q&A":
        st.sidebar.subheader("PDF Upload")
        pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.sidebar.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.sidebar.success("PDF processing complete.")

        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question)

        # Display history of questions and responses for PDF-based Q&A
        if st.session_state['pdf_question_history'] and st.session_state['pdf_response_history']:
            history_df = pd.DataFrame({
                "Question": st.session_state['pdf_question_history'],
                "Response": st.session_state['pdf_response_history']
            })
            st.write("### PDF-based Question and Response History")
            st.dataframe(history_df)

    elif mode == "Chatbot":
        st.sidebar.subheader("Chatbot")
        
        input_question = st.text_input("Enter your question:")
        submit_button = st.button("Ask the question")

        if submit_button and input_question:
            response = get_gemini_response(input_question)
            st.subheader("The Response is")
            response_text = ""
            for chunk in response:
                response_text += chunk.text
                st.write(chunk.text)
            
            # Add user query and response to session state chat history
            st.session_state['chat_history'].append(("User", input_question))
            st.session_state['chat_history'].append(("Tech-Titans", response_text))
        
        st.subheader("Chat History")
        if 'chat_history' in st.session_state:
            for role, text in st.session_state['chat_history']:
                st.write(f"{role}: {text}")

        # Button to save chat history
        if st.button("Save Chat History"):
            with open("chat_history.txt", "w") as file:
                for role, text in st.session_state['chat_history']:
                    file.write(f"{role}: {text}\n")
            st.success("Chat history saved!")

            # Display a download link for the saved chat history
            with open("chat_history.txt", "rb") as file:
                st.download_button(
                    label="Download Chat History",
                    data=file,
                    file_name="chat_history.txt",
                    mime="text/plain",
                )

if __name__ == "__main__":
    main()