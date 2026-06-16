from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "New York is a busy city with a lot of population"

result = embedding.embed_query(text) # embed_docs for documents input
print(result)
