from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(template="answer the question {question} from the following text: {text}",
                        input_variables=["question", "text"])

url = "https://en.wikipedia.org/wiki/El_Ni%C3%B1o%E2%80%93Southern_Oscillation"

loader = WebBaseLoader(url)
docs = loader.load()

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"question":"Which phenomenon is the article talking about?", "text":docs[0].page_content})
print(response)
