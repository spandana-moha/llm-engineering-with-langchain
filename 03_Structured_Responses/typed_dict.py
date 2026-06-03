from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# schema
class Review(TypedDict):
    summary: str
    sentiment: str

prompt = """The hardware is great but the software feels bloated.
            There are too many pre-installed apps that I never use and can't uninstall.
            The battery life is decent, but it could be better.
            Also the UI looks outdated compared to other brands.
            Hoping for a software update to fix this. Overall, it's an average phone with some good features but also some drawbacks."""

structured_model = model.with_structured_output(Review)
response = structured_model.invoke(prompt)
print(response)
