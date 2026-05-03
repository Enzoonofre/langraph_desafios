from dotenv import load_dotenv
from langchain_groq import ChatGroq
# Load environment variables from .env file
load_dotenv()

# Initialize the Gemini model
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Send a simple message
response = llm.invoke("Diga 'Ambiente configurado com sucesso!' em inglês.")

print(response.content)