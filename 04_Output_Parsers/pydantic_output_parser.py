from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Person(BaseModel):
    name: str = Field(description="person's full name")
    age: int = Field(ge=18, lt=60, description="person's age, must not be less than 18 and greater than 60")
    city: str = Field(description="city where the person lives in")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(template=("give name, age and city of a fictional {place} person\n"
                                    "make sure the age is greater than equal to 18 and less than 60\n"
                                    "return response in the following format:\n\n"
                                    "{format_instruction}\n\n"),
                                    input_variables=["place"],
                                    partial_variables={"format_instruction":parser.get_format_instructions()})

prompt = template.invoke({"place":"Austria"})

chain = template | model | parser 

response = chain.invoke({"place":"Austria"})
print(response)
