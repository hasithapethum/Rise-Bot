from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from utils import embeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS



def split_documents(documents):
    text_splitter = SemanticChunker(embeddings=embeddings)

    text_splits = text_splitter.split_documents(documents=documents)
    
    return text_splits


def feed_vectorstore(text_splits):
    index = faiss.IndexFlatL2(embeddings.embed_query("Hello world"))
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    
    vector_store.add_documents(text_splits)
    
    
    
    








