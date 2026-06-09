from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

print(docs)
print(type(docs))
print(docs[0])
print(docs[0].metadata)
print(docs[0].page_content)
