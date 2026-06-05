from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(template="generate detailed report on {topic}",
                         input_variables=["topic"])

prompt2 = PromptTemplate(template="generate 3 point summary of {text}",
                         input_variables=["text"])

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

response = chain.invoke({"topic":"Aliens"})
print(response)
