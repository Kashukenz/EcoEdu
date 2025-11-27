import os
import glob
import streamlit as st
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DATA_PATH = "./data"
DB_PATH = "./db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "ecoedu"

st.set_page_config(page_title="Eco Edu ChatBot", page_icon="🌱")


def is_environmental_query(query: str) -> bool:
    """
    More sophisticated environmental topic detection using semantic keywords.
    Returns True if the query is environmental, False otherwise.
    """
    query_lower = query.lower()

    # Environmental topic indicators
    environmental_keywords = [
        # Core topics
        "climate", "environment", "ecology", "ecosystem", "biodiversity",
        "conservation", "sustainability", "pollution", "emissions", "carbon",
        "renewable", "fossil fuel", "deforestation", "reforestation",

        # Philippine-specific
        "philippines", "manila bay", "boracay", "palawan", "coral reef",
        "typhoon", "monsoon", "denr", "environmental management bureau",

        # Waste & recycling
        "waste", "recycle", "recycling", "garbage", "landfill", "compost",
        "plastic", "single-use", "circular economy",

        # Nature & wildlife
        "species", "endangered", "wildlife", "forest", "marine", "ocean",
        "river", "water", "air quality", "habitat", "flora", "fauna",

        # Energy & resources
        "solar", "wind energy", "geothermal", "energy", "resource",
        "water conservation", "electricity",

        # Impact & policy
        "environmental law", "policy", "regulation", "impact", "assessment",
        "protection", "preservation", "restoration"
    ]

    # Non-environmental topic indicators (more specific)
    non_environmental_indicators = [
        "recipe", "cooking", "movie", "music", "celebrity", "sports team",
        "programming", "code", "javascript", "python tutorial", "css",
        "celebrity gossip", "tv show", "anime", "manga", "video game",
        "fashion", "makeup", "hairstyle", "workout routine", "diet plan",
        "politics", "president"
    ]

    # Check for explicit non-environmental queries
    for indicator in non_environmental_indicators:
        if indicator in query_lower:
            return False

    # Check for environmental content
    for keyword in environmental_keywords:
        if keyword in query_lower:
            return True

    # Ambiguous queries default to environmental (let the RAG system decide)
    return True


def initialize_rag_system():
    """Initialize RAG system with progress tracking"""

    # Create a status container for updates
    status_container = st.empty()
    progress_bar = st.progress(0)

    try:
        # Step 1: Initialize embeddings
        status_container.info("🔧 Step 1/5: Initializing embedding model...")
        progress_bar.progress(10)

        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        progress_bar.progress(20)

        # Step 2: Check if database exists
        if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
            status_container.info("📂 Step 2/5: Loading existing database...")
            progress_bar.progress(40)

            vector_store = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embedding_function
            )
            progress_bar.progress(60)
            status_msg = "✅ Loaded existing database successfully!"

        else:
            # Step 2: Load PDFs
            status_container.info("📂 Step 2/5: Scanning for PDF files...")
            pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))

            if not pdf_files:
                progress_bar.empty()
                status_container.empty()
                return None, "❌ No PDFs found in 'data/' folder. Please add some PDFs first."

            status_container.info(f"📄 Found {len(pdf_files)} PDF(s). Loading documents...")
            progress_bar.progress(30)

            documents = []
            for idx, pdf_file in enumerate(pdf_files):
                file_name = os.path.basename(pdf_file)
                status_container.info(f"📖 Loading {idx + 1}/{len(pdf_files)}: {file_name}")

                try:
                    loader = PyPDFLoader(pdf_file)
                    raw_docs = loader.load()

                    # Clean text
                    for doc in raw_docs:
                        doc.page_content = doc.page_content.replace('\n\n', '  ').replace('\n', ' ').strip()

                    documents.extend(raw_docs)

                except Exception as e:
                    st.warning(f"⚠️ Could not load {file_name}: {str(e)}")
                    continue

            if not documents:
                progress_bar.empty()
                status_container.empty()
                return None, "❌ No documents could be loaded. Check your PDF files."

            progress_bar.progress(45)

            # Step 3: Split into chunks
            status_container.info(f"✂️ Step 3/5: Splitting {len(documents)} pages into chunks...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,  # Increased to keep penalties with context
                chunk_overlap=200,  # More overlap to avoid splitting penalties
                separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)

            status_container.info(f"✂️ Created {len(chunks)} text chunks")
            progress_bar.progress(55)

            # Step 4: Create embeddings (THIS IS THE SLOW PART)
            status_container.warning(
                f"🔄 Step 4/5: Creating embeddings for {len(chunks)} chunks... This may take several minutes. Please wait.")
            progress_bar.progress(60)

            # Create vector store with batch processing
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                persist_directory=DB_PATH
            )

            progress_bar.progress(90)
            status_msg = f"✅ Created new database with {len(chunks)} chunks from {len(pdf_files)} PDFs!"

        # Step 5: Initialize LLM
        status_container.info("🤖 Step 5/5: Initializing language model...")

        llm = ChatOllama(model=LLM_MODEL_NAME)

        # Create retriever with better parameters for finding penalties/specifics
        retriever = vector_store.as_retriever(
            search_type="similarity",  # Changed from mmr - simpler and more direct
            search_kwargs={
                "k": 10,  # Retrieve more chunks to find penalties
            }
        )

        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("human",
             """Context:
{context}

Question: {input}

Provide a clear, direct answer. Never mention "context", "documents", "knowledge base", or "database". Answer naturally as if you simply know the information.""")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        progress_bar.progress(100)
        status_container.success(status_msg)

        # Clear progress indicators after 2 seconds
        import time
        time.sleep(2)
        progress_bar.empty()
        status_container.empty()

        return rag_chain, status_msg

    except Exception as e:
        progress_bar.empty()
        status_container.empty()
        return None, f"❌ Error during initialization: {str(e)}"


st.title("🌱 EcoEdu: Philippines Environment Bot")
st.caption("Ask me about climate change, biodiversity, recycling, and sustainability in the Philippines!")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
    st.session_state.status_msg = None

# Initialize RAG system only once
if st.session_state.rag_chain is None:
    rag_chain, status_msg = initialize_rag_system()
    st.session_state.rag_chain = rag_chain
    st.session_state.status_msg = status_msg

if not st.session_state.rag_chain:
    st.error(st.session_state.status_msg)
    st.info("💡 Tip: Make sure you have PDF files in the './data' folder")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask EcoEdu about environmental topics..."):
    # Basic input validation
    if len(prompt.strip()) < 2:
        # Detect language for error message
        with st.chat_message("assistant"):
            st.markdown(
                "I didn't understand that. Could you ask me about an environmental topic? / Hindi ko po maintindihan. Pwede po bang magtanong tungkol sa kalikasan?")
        st.stop()

    # Handle basic identity questions without RAG (English and Tagalog)
    prompt_lower = prompt.lower().strip()
    identity_questions_en = ["who are you", "what are you", "who r u", "what is ecoedu", "what's ecoedu"]
    identity_questions_tl = ["sino ka", "ano ka", "sino si ecoedu", "ano ang ecoedu"]

    if any(q in prompt_lower for q in identity_questions_en):
        identity_response = "I'm EcoEdu, an environmental assistant that helps answer questions about Philippine environmental laws, regulations, and best practices. I can help you with topics like waste management, air quality, recycling, and environmental penalties."
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            st.markdown(identity_response)
        st.session_state.messages.append({"role": "assistant", "content": identity_response})
        st.stop()

    if any(q in prompt_lower for q in identity_questions_tl):
        identity_response = "Ako si EcoEdu, tumutulong ako sa mga tanong tungkol sa kapaligiran ng Pilipinas. Makakatulong ako sa mga paksa tulad ng pamamahala ng basura, kalidad ng hangin, pag-recycle, at mga parusa sa paglabag sa kalikasan."
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            st.markdown(identity_response)
        st.session_state.messages.append({"role": "assistant", "content": identity_response})
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if query is environmental
    if not is_environmental_query(prompt):
        # Detect if query is in Tagalog
        tagalog_indicators = ["ano", "paano", "saan", "sino", "kailan", "bakit", "pwede", "pano", "gawa", "mag"]
        is_tagalog = any(indicator in prompt_lower for indicator in tagalog_indicators)

        if is_tagalog:
            refusal_text = "Espesyalista ako sa mga paksa tungkol sa kapaligiran ng Pilipinas. Makakatulong ako sa mga tanong tungkol sa pagbabago ng klima, biodiversity, pag-recycle, sustainability, polusyon, at mga batas pangkapaligiran. Pwede po bang magtanong tungkol sa mga paksang ito?"
        else:
            refusal_text = "I specialize in environmental topics related to the Philippines. I can help you with questions about climate change, biodiversity, recycling, sustainability, pollution, and environmental policies. Could you ask me something about these topics?"

        with st.chat_message("assistant"):
            st.markdown(refusal_text)

        st.session_state.messages.append({"role": "assistant", "content": refusal_text})
        st.stop()

    # Process environmental queries
    with st.chat_message("assistant"):
        with st.spinner("Searching through environmental documents..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": prompt})

                # Show sources in expander
                if response.get('context'):
                    with st.expander("📚 View Sources"):
                        sources_seen = set()
                        for i, doc in enumerate(response['context']):
                            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                            if source_name not in sources_seen:
                                st.caption(f"📄 {source_name}")
                                sources_seen.add(source_name)
                            # Show snippet
                            st.text(doc.page_content[:200] + "...")
                            st.divider()

                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"I encountered an error while searching: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with info
with st.sidebar:
    st.header("About EcoEdu")
    st.write("EcoEdu is your environmental assistant focused on Philippine environmental issues.")

    st.subheader("Topics I can help with:")
    st.write("✅ Climate Change / Pagbabago ng Klima")
    st.write("✅ Biodiversity & Conservation / Biodiversity at Konserbasyon")
    st.write("✅ Recycling & Waste Management / Pag-recycle at Pamamahala ng Basura")
    st.write("✅ Sustainability / Sustainability")
    st.write("✅ Environmental Policies / Mga Batas Pangkapaligiran")
    st.write("✅ Pollution & Air Quality / Polusyon at Kalidad ng Hangin")

    st.divider()

    # Debug info
    if st.session_state.rag_chain:
        st.success("✅ System Ready")
    else:
        st.error("❌ System Not Initialized")

    # Add button to reinitialize
    if st.button("🔄 Reinitialize System"):
        st.session_state.rag_chain = None
        st.session_state.status_msg = None
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()