from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    text: str
    processed: str
    final_output: str


def process_text(state: State) -> dict:
    result = state["text"].upper()
    print(f"[process_text] Input: '{state["text"]}'-> Output: '{result}'")
    return {"processed":result}

def format_output(state: State) -> dict:
    result = f"===RESULT: {state['processed']}==="
    print(f"[format_output] Formatted: '{result}'")
    return {"final_output": result}

graph_builder = StateGraph(State)

graph_builder.add_node("process_text", process_text)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "process_text")

graph_builder.add_edge("process_text","format_output")
graph_builder.add_edge("format_output",END)

app = graph_builder.compile()

result = app.invoke({"text":"Hello from langraph"})

print(f"\nFinal state: {result}")
print(f"\nOutput: {result['final_output']}")