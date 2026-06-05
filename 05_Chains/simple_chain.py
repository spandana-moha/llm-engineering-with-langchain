from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate(template="generate 3 facts about a topic {topic}",
                        input_variables=["topic"])

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({"topic":"Aliens"})
print(response)
