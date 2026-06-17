from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

text = ["Cricket is one of the most popular sports in the world and is played between two teams of eleven players each.",
        "The game originated in England and is now especially popular in countries such as India, Australia, England, South Africa, Pakistan, New Zealand, Sri Lanka, Bangladesh, and the West Indies.",
        "A cricket match is played on a circular field with a rectangular pitch at the center.",
        "The pitch is 22 yards long.",
        "One team bats while the other team bowls and fields.",
        "The batting team's objective is to score runs, while the bowling team aims to dismiss batters.",
        "Sachin Tendulkar is known as the God of Cricket and scored 100 international centuries."]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")

vector_store = FAISS.from_texts(texts=text, embedding=embedding)

query = "Who is the God of Cricket?"

result = vector_store.similarity_search(query, k=1)
print(result)
