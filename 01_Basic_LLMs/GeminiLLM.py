from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() # loads credentials from .env file
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = llm.invoke("Hello, Langchain! Explain yourself in one sentence.")
print(response.content)
