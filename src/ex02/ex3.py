from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Define the prompt template with placeholders
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {topic}. Always respond in Brazilian Portuguese."),
    ("human", "{question}"),
])

# Fill in the placeholders
prompt = template.invoke({
    "topic": "Python programming",
    "question": "What are list comprehensions?",
})

response = llm.invoke(prompt)
print(response.content)