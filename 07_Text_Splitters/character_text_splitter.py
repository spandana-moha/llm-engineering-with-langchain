from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator=" ")

text = "Cricket is one of the most popular sports in the world and is played between two teams of eleven players each. The game originated in England and is now especially popular in countries such as India, Australia, England, South Africa, Pakistan, New Zealand, Sri Lanka, Bangladesh, and the West Indies."

result_doc = splitter.split_documents(docs)
result_text = splitter.split_text(text)

print(result_doc[1].page_content)
print(result_doc[1].metadata)
print(result_text)
