from tkinter.messagebox import QUESTION
from utils import llm, State, config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser




graph_builder = StateGraph(State)

def agent(state: State):
    message = state["messages"]


    prompt_template = f"""You are an intelligent and engaging AI chatbot representing Rise Digital, 
    a pioneering AI solutions provider under CodeGen's Rise Tech Village. Your role is to assist users by providing insightful, helpful, and professional responses 
    about our cutting-edge AI solutions, innovations, and contributions to industries and sustainability.
        Your tone should be:
            Knowledgeable but approachable
            Professional yet engaging
            Innovative with a focus on ethical AI and sustainability
        
        Capabilities:
            Provide information about Rise Digital's AI solutions and services.
            Explain how AI is shaping industries and enhancing human lives.
            Highlight our commitment to ethical AI and sustainable technology.
            Assist with inquiries about collaborations, partnerships, or use cases.
    Ensure responses are clear, concise, and future-focused, reinforcing Rise Digital’s role as a leader in AI-driven transformation.
    Questions : {QUESTION}
    
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"QUESTION": RunnablePassthrough()} |
        prompt |
        llm
    )

    response = chain.invoke({
        "QUESTION": message,
    })

    return {"messages": [response]}




graph_builder.add_node("agent", agent)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)


memory = MemorySaver()


graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    for event in graph.astream({"messages": [{"role": "user", "content": user_input}]},
                              config=config):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)



def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END








