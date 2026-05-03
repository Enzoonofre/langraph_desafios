from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# The chat history starts with the system instruction
history = [
    SystemMessage(content="You are a helpful and concise assistant. Always respond in Brazilian Portuguese."),
]

print("Chat com Groq (digite 'sair' para encerrar)\n")

while True:
    user_input = input("Você: ")
    
    if user_input.lower() in ["sair", "exit", "quit"]:
        print("Encerrando o chat. Até mais!")
        break
    
    # Add user message to history
    history.append(HumanMessage(content=user_input))
    
    # Call the LLM with the full history
    response = llm.invoke(history)
    
    # Add AI response to history (important for context!)
    history.append(AIMessage(content=response.content))
    
    print(f"\n Groq: {response.content}\n")