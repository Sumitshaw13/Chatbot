# PDF-based Q&A and AI Chatbot Platform

This repository contains a project that allows users to interact with PDF documents through a Question-Answering (Q&A) system using LangChain and Google Gemini API. Additionally, it features an AI chatbot interface, also powered by the Google Gemini API. The platform is built using Python and Streamlit for the frontend and leverages LangChain for document processing and conversational AI.

## Features

- **PDF-based Q&A:**
  - Upload multiple PDF documents and ask questions based on the content of those documents.
  - Uses LangChain for processing the documents and splitting them into manageable text chunks.
  - Employs FAISS (Facebook AI Similarity Search) to perform similarity searches for relevant document sections.
  - Provides accurate answers from the PDF content via Google's Gemini language model.

- **AI Chatbot:**
  - Chat with an AI powered by the Google Gemini model.
  - Supports real-time conversational responses.
  - Displays chat history and allows users to save and download chat sessions.

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, LangChain, Google Gemini API
- **Database:** FAISS (for document vector search), Firebase (if needed for authentication or storage)
- **PDF Parsing:** PyPDF2
- **Embeddings & AI Models:** GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI (LangChain integration)
  
## Prerequisites

To run this project, you'll need the following:

- Python 3.9+
- A Google API key for the Gemini model
- FAISS for similarity search
- A `.env` file with your `GOOGLE_API_KEY` (Alternatively, you can set it up in Streamlit secrets)

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/tech-titans.git
    cd tech-titans
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/MacOS
    .\venv\Scripts\activate   # Windows
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory to store your Google API key:
    ```bash
    touch .env
    ```

   Add your Google API key:
    ```text
    GOOGLE_API_KEY=your-google-api-key
    ```

5. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. **PDF-based Q&A:**
   - In the sidebar, upload one or more PDF files.
   - Once the PDF is processed, you can ask questions based on the content of the PDFs.
   - A similarity search will be performed on the PDF content, and the most relevant sections will be used to answer the question.
   
2. **Chatbot Mode:**
   - Switch to the "Chatbot" mode in the sidebar.
   - Ask the chatbot any question, and it will provide an AI-generated response using the Google Gemini API.
   - You can also view and save your chat history.

## File Structure

