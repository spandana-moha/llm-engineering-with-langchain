from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Rent Agreement - Balaji Whitefield - B 402_compressed.pdf")
docs = loader.load()

print(docs[1])
