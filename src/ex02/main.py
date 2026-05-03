from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

messages = [
    SystemMessage(content="You are a helpful assistant that always responds in Brazilian Portuguese."),
    HumanMessage(content="What is machine learning?"),
]

response = llm.invoke(messages)
print(f"\nModel used: {response.response_metadata}")