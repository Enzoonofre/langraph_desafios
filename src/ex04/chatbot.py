from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# System instruction
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant. Always respond in Brazilian Portuguese."
)

def call_llm(state: MessagesState) -> dict:
    # Prepend the system message to the history
    messages = [SYSTEM_MESSAGE] + state["messages"]
    response = llm.invoke(messages)
    # Return the AI response to be added to the messages list
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("llm", call_llm)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

app = graph_builder.compile()

# Interactive chat loop
print("Chatbot com LangGraph + Groq (digite 'sair' para encerrar)\n")

# The conversation history is stored here
conversation_messages = []

while True:
    user_input = input("Você: ")
    
    if user_input.lower() in ["sair", "exit", "quit"]:
        print("Encerrando. Até mais!")
        break
    
    # Add user message to history
    conversation_messages.append(HumanMessage(content=user_input))
    
    # Invoke the graph with the full history
    result = app.invoke({"messages": conversation_messages})
    
    # Update history with the full result (includes AI response)
    conversation_messages = result["messages"]
    
    # Get and print the last message (AI response)
    last_message = result["messages"][-1]
    print(f"\nGroq: {last_message.content}\n")