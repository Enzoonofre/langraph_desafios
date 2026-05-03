from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class State(TypedDict):
    # Without add_messages: would REPLACE the list
    # messages: list[BaseMessage]
    
    # With add_messages: APPENDS to the list
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

def classify_question(state: MessagesState) -> Literal["tech_expert", "general_assistant"]:
    """Classifies whether the question is about programming/tech."""
    last_message = state["messages"][-1].content
    
    classification_prompt = [
        SystemMessage(content="""You classify questions. Answer ONLY with one word:
        - 'tech' if the question is about programming, software, or technology
        - 'general' for anything else"""),
        HumanMessage(content=last_message)
    ]
    
    response = llm.invoke(classification_prompt)
    category = response.content.strip().lower()
    
    if "tech" in category:
        return "tech_expert"
    return "general_assistant"

def tech_expert(state: MessagesState) -> dict:
    """Responds as a technical expert."""
    messages = [
        SystemMessage(content="You are a senior software engineer. Give detailed, technical answers in Brazilian Portuguese."),
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def general_assistant(state: MessagesState) -> dict:
    """Responds as a general assistant."""
    messages = [
        SystemMessage(content="You are a friendly, helpful assistant. Answer in Brazilian Portuguese."),
    ] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("tech_expert", tech_expert)
graph_builder.add_node("general_assistant", general_assistant)

graph_builder.add_conditional_edges(START, classify_question)
graph_builder.add_edge("tech_expert", END)
graph_builder.add_edge("general_assistant", END)

app = graph_builder.compile()

# Test
questions = [
    "Como funciona um decorator em Python?",
    "Qual é a capital da Austrália?",
]

for question in questions:
    print(f"\n{'='*50}")
    print(f"Pergunta: {question}")
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Resposta: {result['messages'][-1].content}")