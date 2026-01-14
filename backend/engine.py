import os
import arxiv
from dotenv import load_dotenv

# Cloud-compatible Components
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class ArxivEngine:
    def __init__(self):
        # Swapped Ollama for HuggingFace - This makes the app deployable to the web!
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def search_arxiv(self, query: str, max_results=6):
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        return [{
            "id": p.entry_id.split('/')[-1], 
            "title": p.title, 
            "summary": p.summary[:300] + "...",
            "authors": [a.name for a in p.authors],
            "year": p.published.year
        } for p in search.results()]

    def fetch_and_index(self, paper_id: str):
        # 1. Download
        search = arxiv.Search(id_list=[paper_id])
        paper = next(search.results())
        if not os.path.exists("./downloads"): os.makedirs("./downloads")
        path = paper.download_pdf(dirpath="./downloads")

        # 2. Process
        loader = PyPDFLoader(path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(pages)

        # 3. Vectorize (Using Chroma + HuggingFace)
        # Persistent directory ensures we don't re-index if paper was already processed
        self.vector_db = Chroma.from_documents(
            documents=splits, 
            embedding=self.embeddings,
            persist_directory=f"./db/{paper_id}"
        )
        return paper.title

    def ask_question(self, paper_id: str, question: str):
        # Load the existing DB for that specific paper
        db = Chroma(persist_directory=f"./db/{paper_id}", embedding_function=self.embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # High-End Research Prompt
        template = """You are ARXAST, a PhD-level research assistant. Use the following context from the research paper to answer the user's inquiry. 
        If you don't know the answer based on the context, state that, but provide related insights from the text.
        
        CONTEXT: {context}
        
        INQUIRY: {question}
        
        ANALYSIS:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # The LCEL Chain - Reliable and fast
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(question)