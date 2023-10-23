import streamlit as st
import pickle
import os
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

# Basic Layout of the Site
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered PDF chatbot which uses LLM model to answer the question from the uploaded PDF built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Flan-T5 by Google](https://huggingface.co/google/flan-t5-xxl) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made by Rishi Patel for  PearlThoughts')

load_dotenv()

#Defining the main Work Layout
def main():
    global VectorStore
    st.header("Chat with PDF")

    # upload PDF
    pdf = st.file_uploader("Upload the PDF", type='pdf')

    #Scarpping the PDF
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
    # Embedding
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

    # st.write(chunks)
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embedding = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding = embedding)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        # Defining LLM Model from Hugging Face
        repo_id =  "google/flan-t5-xxl"
        llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})

        # st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm_chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = llm_chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__=='__main__':
    main()