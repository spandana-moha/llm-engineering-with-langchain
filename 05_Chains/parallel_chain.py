from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser = StrOutputParser()

prompt1 = PromptTemplate(template="generate short and simple notes from following text: {text}",
                         input_variables=["text"])

prompt2 = PromptTemplate(template="generate 5 short answer questions from following text: {text}",
                         input_variables=["text"])

prompt3 = PromptTemplate(template="merge the notes and questions with answers into single document {notes}, {qna}",
                         input_variables=["notes", "qna"])

runnable_chain = RunnableParallel({"notes": prompt1 | model1 | parser,
                                  "qna": prompt2 | model2 | parser})

merge_chain = prompt3 | model1 | parser

chain = runnable_chain | merge_chain

text = """Support Vector Machine (SVM) is a supervised machine learning algorithm used for both classification and regression tasks, although it is primarily used for classification problems. The main objective of SVM is to find the optimal decision boundary, known as a hyperplane, that separates data points belonging to different classes with the maximum possible margin.

In a two-dimensional space, the hyperplane is simply a line that divides the dataset into different classes. In higher-dimensional spaces, the hyperplane becomes a plane or a higher-dimensional surface. SVM attempts to maximize the distance between the hyperplane and the nearest data points from each class. These nearest data points are called support vectors, and they play a crucial role in determining the position and orientation of the hyperplane.

One of the major advantages of SVM is its ability to handle high-dimensional data effectively. It performs well even when the number of features is greater than the number of samples. SVM is also effective in cases where there is a clear margin of separation between classes."""

response = chain.invoke(text)
print(response)
