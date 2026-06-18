from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

documents = [Document(page_content="Langchain makes it easy to work with LLMs."),
             Document(page_content="Langchain helps developers build LLM applications easily."),
             Document(page_content="Chroma is a vector database optimized for LLM-based search."),
             Document(page_content="Embeddings convert text into high-dimensional vectors."),
             Document(page_content="MMR helps you get diverse results when doing similarity search"),
             Document(page_content="OpenAI provides powerful embedding models.")]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(documents=documents, embedding=embedding)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k":3, "lambda_mult":0.25})
# lambda_mult = 0 means full diversity and ignores relevance completely
# lambda_mult = 0.25 means 25% focus on relevance and 75% focus on diversity

query = "What is Langchain?"

result = retriever.invoke(query)
print(result)
