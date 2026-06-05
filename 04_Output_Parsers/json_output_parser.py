from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = JsonOutputParser()

template = PromptTemplate(template="give name, age and city of a fictional person and the name and city must be of indian origin. {format_instruction}",
                          input_variables=[],
                          partial_variables={"format_instruction":parser.get_format_instructions})

chain = template | model | parser

response = chain.invoke({})
print(response)
