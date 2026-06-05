from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["Positive", "Negative"] = Field(description="sentiment of the feedback, must be either positive or negative")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(template="classify the sentiment of the text as Positive or Negative: {feedback}, {format_instruction}",
                         input_variables=["feedback"],
                         partial_variables={"format_instruction":parser2.get_format_instructions()})

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(template="write an appropriate response to positive feedback: {feedback}",
                         input_variables=["feedback"])

prompt3 = PromptTemplate(template="write an appropriate response to negative feedback: {feedback}",
                         input_variables=["feedback"])

branch_chain = RunnableBranch((lambda x: x.sentiment=="Positive", prompt2 | model | parser1),
                              (lambda x: x.sentiment=="Negative", prompt3 | model | parser1),
                              RunnableLambda(lambda x: "No valid sentiment found"))

chain = classifier_chain | branch_chain

response = chain.invoke({"feedback":"This is a beautiful place"})
print(response)
