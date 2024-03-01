from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from vector_store.CustomFaissRetriever import CustomFaissRetriever
import os

class Vector_Store:
    def __init__(self, db_path=None, input_docs_path=None):
        # Relative paths
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(__file__), 'faiss_index')
        if input_docs_path is None:
            self.input_docs_path = os.path.join(os.path.dirname(__file__), 'input_docs')
        self.db = None
        self.retriever = None
        self.load_database()
        self.create_retriever()

    def load_database(self):
        # Load embedding model (https://huggingface.co/spaces/mteb/leaderboard) to convert text to vectors
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cuda"}
        encode_kwargs = {"normalize_embeddings": True}

        embedding = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # Create or load database
        if os.path.exists(self.db_path):
            print("Loading database...")
            self.db = FAISS.load_local(self.db_path, embedding)
        else:
            print("No data found. Creating new database based on input folder...")
            self.create_database(embedding)

    def create_retriever(self):
        # Retriever (Custom class using similarity_search_with_score as langchain currently does not support it)
        self.retriever = CustomFaissRetriever(
            vectorstore=self.db,
            search_type="similarity_score_threshold", # also used by Microsoft
            search_kwargs={"score_threshold": 0.5, "k": 3} # https://js.langchain.com/docs/modules/data_connection/retrievers/similarity-score-threshold-retriever
        )

    def get_retriever(self):
        return self.retriever
    
    def create_database(self, embedding):

        # General idea: https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter
        # Text Splitter used: https://python.langchain.com/docs/modules/data_connection/document_transformers/split_by_token

        # Load texts
        loader = DirectoryLoader(
            path=self.input_docs_path,
            glob="./*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True,
            loader_kwargs={'autodetect_encoding': True}
        )
        raw_docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200, chunk_overlap=0)
        split_docs = text_splitter.split_documents(raw_docs)

        # Save documents to database (chunks are converted to vectors with embedding model)
        self.db = FAISS.from_documents(documents=split_docs, embedding=embedding)

        # Save db to disk (this is also called the "index")
        self.db.save_local(self.db_path)