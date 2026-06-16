from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = ["Large Language Models are trained on massive datasets",
        "Large Language Models (LLMs) are particularly trained using transformers",
        "Chroma is a light-weight vector store used in Langchain",
        "Embeddings convert text into numerical representation"]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")

vector_store = Chroma.from_texts(texts=text, embedding=embedding, collection_name="langchain_chroma_demo")

query = "tell me more about LLMs"

result = vector_store.similarity_search(query, k=2)
print(result)
