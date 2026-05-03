from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- Define Tools ---

@tool
def get_weather(city: str) -> str:
    """
    Returns the current weather for a given city.
    Use when the user asks about weather or temperature in a city.
    """
    weather_data = {
        "São Paulo": "25°C, partly cloudy",
        "Rio de Janeiro": "30°C, sunny",
        "Brasília": "22°C, clear sky",
        "Curitiba": "18°C, rainy",
    }
    return weather_data.get(city, f"Weather data not available for '{city}'")

@tool
def calculate(expression: str) -> str:
    """
    Evaluates a mathematical expression.
    Use for any math calculations like addition, multiplication, etc.
    Example expressions: '2 + 2', '100 * 0.15', '(10 + 5) * 2'
    """
    try:
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def get_current_date() -> str:
    """Returns today's date. Use when the user asks what day or date it is."""
    from datetime import date
    return f"Today is {date.today().strftime('%B %d, %Y')}"

# --- Setup LLM with Tools ---
tools = [get_weather, calculate, get_current_date]

llm = ChatGroq(model="llama-3.3-70b-versatile")
llm_with_tools = llm.bind_tools(tools)  # Tell the LLM which tools it can use

# --- Define Nodes ---
SYSTEM = SystemMessage(
    content="You are a helpful assistant. Use the available tools when needed. Always respond in Brazilian Portuguese."
)

def call_llm(state: MessagesState) -> dict:
    """Node: calls the LLM (with tools bound)."""
    messages = [SYSTEM] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ToolNode is a prebuilt node that automatically executes tool calls
tool_node = ToolNode(tools)

# tools_condition is a prebuilt conditional edge function:
# - returns "tools" if the LLM wants to call a tool
# - returns END if the LLM gave a final answer
def route(state: MessagesState):
    return tools_condition(state)

# --- Build the Graph ---
graph_builder = StateGraph(MessagesState)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", route)  # LLM → tools or END
graph_builder.add_edge("tools", "llm")             # After tools → back to LLM

app = graph_builder.compile()

# --- Test the Agent ---
test_questions = [
    "Qual é o clima em Curitiba?",
    "Quanto é 15% de 240?",
    "Que dia é hoje e como está o tempo em São Paulo?",  # uses 2 tools!
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"Pergunta: {question}")
    result = app.invoke({"messages": [HumanMessage(content=question)]})
    print(f"Resposta: {result['messages'][-1].content}")