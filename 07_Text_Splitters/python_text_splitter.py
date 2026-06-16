from langchain_text_splitters import PythonCodeTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade 
    
    def get_details(self):
        return self.name
    
    def is_passing(self):
        return self.grade >= 6.0

# Example
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("Student is passing")
else:
    print("Student is not passing")
    """

splitter = PythonCodeTextSplitter(chunk_size=300, chunk_overlap=100)
chunks = splitter.split_text(text)
print(chunks)
