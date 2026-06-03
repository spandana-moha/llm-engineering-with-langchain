from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

print("Blog Post Generator")
print("Provide ideas or instructions for the blog post. Type 'exit' to quit.")

topic = input("Enter blog post topic:")

chat_prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a creative blog post writer. Generate informative, engaging, and well-structured blog posts about {topic}."),
        HumanMessagePromptTemplate.from_template("Write a detailed blog post about {topic}.")
    ])

# Initialize chat history
chat_history = []

while True:
    user_input = input("\nIdeas/Instructions for blog post or type 'exit': ")

    if user_input.lower() == "exit":
        print("Exiting Blog Post Generator...")
        break

    # Create base prompt messages
    messages = chat_prompt_template.format_messages(topic=topic)

    # Add previous conversation history
    for prev in chat_history:
        messages.append(prev)

    # Create current user message
    current_user_message = HumanMessagePromptTemplate.from_template("{user_input}").format_messages(user_input=user_input)[0]

    messages.append(current_user_message)

    # Get model response
    response = chat_model.invoke(messages)

    print("\nBlog Post Content:\n")
    print(response.content)

    # Store conversation history
    chat_history.append(current_user_message)
    chat_history.append(response)
