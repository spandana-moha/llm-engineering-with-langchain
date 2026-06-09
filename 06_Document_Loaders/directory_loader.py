from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="docs", glob="*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

print(len(docs))
print(docs[0].metadata)

# lazy loading
docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
