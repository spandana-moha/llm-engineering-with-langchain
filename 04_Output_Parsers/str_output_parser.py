from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template1 = PromptTemplate(template="write detailed report on {topic}.",
                           input_variables=["topic"])

template2 = PromptTemplate(template="write 5 line summary on {text}.",
                           input_variables=["text"])

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic":"Black Hole"})
print(response)
