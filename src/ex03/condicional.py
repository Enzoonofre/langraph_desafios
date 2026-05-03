from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    number: int
    result: str


def check_number(state: State) -> Literal["positive_node", "negative_node"]:
    """Decides which node to go to based on the number."""
    if state["number"] > 0:
        return "positive_node"
    else:
        return "negative_node"
    

def positive_node(state: State) -> dict:
    return {"result": f"{state['number']} is a positive number!"}

def negative_node(state: State) -> dict:
    return {"result": f"{state['number']} is a negative number (or zero)!"}


graph_builder = StateGraph(State)
graph_builder.add_node("positive_node", positive_node)
graph_builder.add_node("negative_node", negative_node)

graph_builder.add_conditional_edges(
    START,                      # from this node
    check_number,               # using this function to decide
    {
        "positive_node": "positive_node",   # if function returns "positive_node" → go to "positive_node"
        "negative_node": "negative_node",   # if function returns "negative_node" → go to "negative_node"
    }
)

graph_builder.add_edge("positive_node", END)
graph_builder.add_edge("negative_node", END)

app = graph_builder.compile()

print(app.invoke({"number": -42})["result"])