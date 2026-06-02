from langchain_core.prompts import PromptTemplate

dynamic_prompt = PromptTemplate(template="Provide a brief summary about {topic} in {language} language.",
                                input_variables=["topic", "language"])

prompt_text = dynamic_prompt.format(topic="AI", language="English")
print(prompt_text)

prompt_text_01 = dynamic_prompt.format(topic="Blockchain & IoT", language="English")
print(prompt_text_01)
