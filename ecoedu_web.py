import os
import glob
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "./data"
DB_PATH ="./db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "ecoedu"

st.set_page_config(page_title="Eco Edu Chatpog", page_icon="🌱")

@st.cache_resource
def initialize_rag_system():
    print("Initializing RAG System")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
    else:
        documents = []
        pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
        if not pdf_files:
            return None, "No PDFs found in 'data/' folder."
        for pdf_file  in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        chunks = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=DB_PATH
        )
    llm = ChatOllama(model=LLM_MODEL_NAME)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # 2. Simplified Prompt
    # We don't need to tell it "You are EcoEdu" anymore.
    # We just need to say "Here is the data, answer the question."
    rag_prompt = (
        "Context information is below:\n"
        "---------------------\n"
        "{context}\n"
        "---------------------\n"
        "Given this context, please answer the question: {input}"
    )

    prompt = ChatPromptTemplate.from_template(rag_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain, "System initialized successfully with custom 'EcoEdu' model!"


# --- UI Layout ---
st.title("🌱 EcoEdu: Custom RAG Model")

if "messages" not in st.session_state:
    st.session_state.messages = []

rag_chain, status_msg = initialize_rag_system()

if not rag_chain:
    st.error(status_msg)
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask EcoEdu..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consulting local documents..."):
            response = rag_chain.invoke({"input": prompt})
            with st.expander("See Retrieved Context"):
                st.write(response['context'])
            answer = response['answer']
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})