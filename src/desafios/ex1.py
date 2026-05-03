# - `celsius_to_fahrenheit(celsius: float)` → converte temperatura
# - `get_bmi(weight_kg: float, height_m: float)` → calcula o IMC e retorna a classificação
# - `convert_currency(amount: float, from_currency: str, to_currency: str)` → converte valores entre moedas (pode ser com taxas fixas/simuladas)

from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """
    Converte uma temperatura de graus Celsius para Fahrenheit.
    Args:
        celsius: O valor numérico da temperatura em graus Celsius.  
    Returns:
        O valor convertido da temperatura em graus Fahrenheit.
    """
    return (celsius * 1.8) + 32

@tool
def get_bmi(weight_kg: float, height_m: float) -> float:
    """
    Calcula o Índice de Massa Corporal (IMC).
    Args:
        weight_kg: O peso da pessoa em quilogramas (ex: 70.5).
        height_m: A altura da pessoa em metros (ex: 1.75).   
    Returns:
        O valor numérico do IMC calculado.
    """
    return weight_kg / (height_m ** 2)

@tool
def convert_currency(valor: float) -> float:
    """
    Converte um valor monetário de Reais (BRL) para Dólares (USD).
    Args:
        valor: O montante em Reais (BRL) a ser convertido para Dólares. 
    Returns:
        Uma string contendo o valor convertido em USD e a taxa de câmbio utilizada.
    """
    response = requests.get("https://open.er-api.com/v6/latest/BRL")
    data = response.json()
    taxa = data["rates"]["USD"]
    resultado = valor * taxa
    return resultado

tools = [celsius_to_fahrenheit,get_bmi,convert_currency]

llm = ChatGroq(model="llama-3.3-70b-versatile")

llm_with_tools = llm.bind_tools(tools)



SYSTEM = SystemMessage(
    content="You are a personal assistant that always respond in Brazilian portuguese in a clear and concise way with a friendly tone")


def call_llm(state : MessagesState) -> dict:
    messages = [SYSTEM] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

def route(state: MessagesState):
    return tools_condition(state)

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("llm", call_llm)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges("llm", route)
graph_builder.add_edge("tools","llm")

app = graph_builder.compile()


def print_result(result):
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "HumanMessage":
            print(f"[Human]: {msg.content}")
        elif msg_type == "AIMessage":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[AI calls tool]: {tc['name']}({tc['args']})")
            else:
                print(f"[AI final answer]: {msg.content}")
        elif msg_type == "ToolMessage":
            print(f"[Tool result]: {msg.content}")


if __name__ == "__main__":
    print("=== Assistente Pessoal ===")
    print("Digite 'sair' para encerrar.\n")

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in ("sair", "exit", "quit"):
            print("Até logo!")
            break
        if not user_input:
            continue

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        print()
        print_result(result)
        print()
