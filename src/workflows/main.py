from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from pydantic import Field
from typing import TypedDict, List, Literal, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch

load_dotenv()


# Models
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

query_rewrite_model = ChatOllama(model="gemma4:31b-cloud")
judge_model = ChatOllama(model="gemma4:31b-cloud")
critic_model = ChatOllama(model="gemma4:31b-cloud")
main_model = ChatOllama(model="gemma4:31b-cloud")

# vector_store
vector_store = FAISS.load_local(
    "./data/constitution_and_ipc.faiss",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

tavily_tool = TavilySearch(max_results=3)


# schema
class schema(TypedDict):
    # Not including retriever because it is inserilisable so it is giving error with inmemorystore
    # retriever:BaseRetriever
    retrieval_required: Literal["retrieval", "web_search", "None"]
    web_searched: bool
    user_query: str
    retrieved_contexts: List[str]
    relevant_contexts: List[str]
    answer_for_query: str
    generated_response: str
    is_grounded: bool
    is_supported: bool
    is_answer_useful: bool
    evidence: str
    k: Optional[int] = Field(default=3)
    max_retry_for_revise_answer: Optional[int] = Field(default=3)
    max_retry_for_rewrite_query: Optional[int] = Field(default=2)


# Nodes
class schema_for_retrieval_decider_node(BaseModel):
    retrieval_required: Literal["retrieval", "web_search", "None"] = Field(...)


parser_for_retrieval_decider_node = PydanticOutputParser(
    pydantic_object=schema_for_retrieval_decider_node
)

sys_prompt_for_retrieval_decider_node = f"You are an expert legal AI Assistant specializing in the Indian Penal Code (IPC) and the Constitution of India. Your task is to analyze the user's query and determine the most appropriate retrieval method. \n\n- Choose 'retrieval' if the answer is likely to be found in the official legal documents (IPC or Constitution) stored in the internal vector database.\n- Choose 'web_search' if the query requires current events, recent legal precedents, or general information not likely to be in the static legal documents.\n- Choose 'None' if the query is a greeting or does not require external information to be answered.\n\nOutput Format -{parser_for_retrieval_decider_node.get_format_instructions()}"


def retrieval_decider_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retrieval_decider_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_retrieval_decider_node.invoke(
        main_model.invoke(inp).content
    ).retrieval_required
    return {"retrieval_required": res}


def retrieve_node(state: schema):
    global retriever
    retrieved_contexts = retriever.invoke(state["user_query"], k=state["k"])
    return {"retrieved_contexts": [i.page_content for i in retrieved_contexts]}


def direct_generation_node(state: schema):
    res = main_model.invoke(state["user_query"]).content
    return {"generated_response": res}


class schema_for_is_relevant_node(BaseModel):
    is_relevant_context: bool


parser_for_is_relevant_node = PydanticOutputParser(
    pydantic_object=schema_for_is_relevant_node
)


def is_relevant_node(state: schema):
    contexts = state["retrieved_contexts"]
    sys_prompt = SystemMessage(
        content=f"""You are a legal analyst. Your task is to determine if the provided retrieved context is relevant to answer the user's query. \n\nAnalyze if the context contains information that is required to answer user's query. \n\nOutput format - {parser_for_is_relevant_node.get_format_instructions()}"""
    )
    hmn_prompt = f"Query - {state['user_query']}"
    lst = []
    for context in contexts:
        hmn_prompt_dash = hmn_prompt + f"\n Context - {context}"
        res = parser_for_is_relevant_node.invoke(
            main_model.invoke(
                [sys_prompt, HumanMessage(content=hmn_prompt_dash)]
            ).content
        )

        lst.append(res.is_relevant_context)

    return {"relevant_contexts": [contexts[i] for i in range(len(contexts)) if lst[i]]}


class schema_for_answer_from_context_node(BaseModel):
    response: str = Field(..., description="Response for given query")


parser_for_answer_from_context_node = PydanticOutputParser(
    pydantic_object=schema_for_answer_from_context_node
)


def answer_from_context_node(state: schema):
    contexts = state["relevant_contexts"]
    sys_prompt_for_answer_from_context_node = SystemMessage(
        content=f"""You are an expert legal AI Assistant. Your task is to answer the user's query accurately using the provided contexts. \n\nCRITICAL INSTRUCTIONS:\n1. Use ONLY the provided context to answer the query.\n2. If the answer is not present in the context, explicitly state that the information is not available in the provided documents.\n3. You MUST cite your sources. For every claim or piece of information, quote the relevant part of the context or provide the URL if it's a web search result (e.g., \"According to [Source/URL], ...\").\n4. Maintain a professional and objective legal tone.\n\nOutput format - {parser_for_answer_from_context_node.get_format_instructions()}"""
    )

    context = ""
    for i in contexts:
        context += i
        context += "\n"

    hmn_prompt = HumanMessage(
        content=f"Query - {state['user_query']} \n\n Contexts - {context}"
    )
    inp = [sys_prompt_for_answer_from_context_node, hmn_prompt]

    res = parser_for_answer_from_context_node.invoke(
        main_model.invoke(inp).content
    ).response
    return {"generated_response": res}


class schema_for_check_answer_grounded_node(BaseModel):
    is_grounded: Literal["fully_supported", "not_fully_supported"]
    evidence: str = Field(
        ..., description="Proof that answer is not supported by given contexts"
    )


parser_for_schema_for_check_answer_grounded_node = PydanticOutputParser(
    pydantic_object=schema_for_check_answer_grounded_node
)


def check_answer_grounded_node(state: schema):
    contexts = state["relevant_contexts"]
    sys_prompt = SystemMessage(
        content=f"You are a legal auditor. Your task is to verify if the generated answer is fully supported by the provided context. \n\nCheck for any hallucinations, inaccuracies, or information added that is not present in the context. If the answer is not fully supported, identify the specific part of the answer that lacks evidence and explain why. \n\nOutput format - {parser_for_schema_for_check_answer_grounded_node.get_format_instructions()}"
    )
    context = ""
    for i in contexts:
        context += i
        context += "\n"

    human_pr = HumanMessage(
        content=f"Answer - {state['generated_response']} \n Contexts - {context}"
    )

    res = parser_for_schema_for_check_answer_grounded_node.invoke(
        main_model.invoke([sys_prompt, human_pr]).content
    )

    return {"is_grounded": res.is_grounded, "evidence": res.evidence}


class schema_for_revise_answer_node(BaseModel):
    revised_response: str = Field(..., description="Response for given query")


parser_for_revise_answer_node = PydanticOutputParser(
    pydantic_object=schema_for_revise_answer_node
)


def revise_answer_node(state: schema):
    contexts = state["relevent_contexts"]
    context = ""
    for i in contexts:
        context += i
        context += "\n"
    generated_response = state["generated_response"]
    user_query = state["user_query"]
    evidence = state["evidence"]

    sys_prompt = SystemMessage(
        content=f"You are a legal editor. You will be provided with a user query, a generated answer, the relevant contexts, and evidence showing why the current answer is not fully supported by the contexts. \n\nYour task is to revise the answer so that it is completely grounded in the provided contexts. Ensure that all claims are supported by the evidence and that you maintain the requirement to quote sources/URLs in the final response.\n\nOutput format - {parser_for_revise_answer_node.get_format_instructions()}"
    )

    human_pr = HumanMessage(
        content=f"Query - {user_query} \n\n Generated Response - {generated_response} \n\n Contexts - {x} \n\n Evidence - {evidence}"
    )

    revised_answer = parser_for_revise_answer_node.invoke(
        critic_model.invoke([sys_prompt, human_pr]).content
    )

    return {
        "generated_response": res.revised_response,
        "max_retry_for_revise_answer": state["max_retry_for_revise_answer"] - 1,
    }


class schema_for_is_answer_useful_node(BaseModel):
    is_useful: bool = Field(
        ..., description="Boolean response whether answer is useful or not"
    )


parser_for_is_answer_useful_node = PydanticOutputParser(
    pydantic_object=schema_for_is_answer_useful_node
)


def is_answer_useful_node(state: schema):
    user_query = state["user_query"]
    generated_response = state["generated_response"]

    sys_prompt = SystemMessage(
        content=f"You are a legal quality assurance judge. Your task is to evaluate whether the generated response successfully and accurately solves the user's query based on the legal contexts provided. \n\nConsider if the answer is complete, accurate, and directly addresses the user's core question. \n\nOutput format - {parser_for_is_answer_useful_node.get_format_instructions()}"
    )
    human_pr = HumanMessage(
        content=f"Query - {user_query} \n\n Generated Response - {generated_response}"
    )

    res = parser_for_is_answer_useful_node.invoke(
        judge_model.invoke([sys_prompt, human_pr]).content
    )
    return {"is_answer_useful": res.is_useful}


class schema_for_rewrite_query_node(BaseModel):
    updated_query: str


parser_for_rewrite_query_node = PydanticOutputParser(
    pydantic_object=schema_for_rewrite_query_node
)


def rewrite_query_node(state: schema):
    sys_prompt = SystemMessage(
        content=f"You are a legal search expert. Your task is to rewrite a user's query to make it highly optimized for vector database retrieval. \n\nThe database contains legal documents from the Indian Constitution and other official Acts. If the user's query is vague or colloquial, expand it using legal terminology and specify the likely sections or themes it relates to, while maintaining the original intent. \n\nOutput Format - {parser_for_rewrite_query_node.get_format_instructions()}"
    )
    human_pr = state["user_query"]

    res = parser_for_rewrite_query_node.invoke(
        query_rewrite_model.invoke([sys_prompt, human_pr]).content
    )
    return {
        "user_query": res.updated_query,
        "max_retry_for_rewrite_query": state["max_retry_for_rewrite_query"] - 1,
    }


def web_search_node(state: schema):
    x = tavily_tool.invoke(state["user_query"])
    res = []
    for r in x["results"]:
        p = f"Source - {r['url']} \n title - {r['title']} \n {r['content']}"
        res.append(p)
    return {"relevant_contexts": res, "web_searched": True}


def retrieval_decider_condition(state: schema):
    return state["retrieval_required"]


def is_relevant_condition(state: schema):
    return len(state["relevant_contexts"]) > 0


def is_grounded_condition(state: schema):
    return state["is_grounded"] and state["max_retry_for_revise_answer"] > 0


def is_answer_useful_condition(state: schema):
    return state["is_answer_useful"] and state["max_retry_for_rewrite_query"] > 0


conn=sqlite3.connect("statedb.db", check_same_thread=False)
ck_ptr = SqliteSaver(conn=conn)

graph = StateGraph(state_schema=schema)

# adding nodes
# TODO - add a websearch node if none of generated contexts are relevant
graph.add_node("retrieval_decider_node", retrieval_decider_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("direct_generation_node", direct_generation_node)
graph.add_node("is_relevant_node", is_relevant_node)
graph.add_node("answer_from_context_node", answer_from_context_node)
graph.add_node("check_answer_grounded_node", check_answer_grounded_node)
graph.add_node("revise_answer_node", revise_answer_node)
graph.add_node("is_answer_useful_node", is_answer_useful_node)
graph.add_node("rewrite_query_node", rewrite_query_node)
graph.add_node("web_search_node", web_search_node)


graph.add_edge(START, "retrieval_decider_node")
graph.add_conditional_edges(
    "retrieval_decider_node",
    retrieval_decider_condition,
    {
        "retrieval": "retrieve_node",
        "None": "direct_generation_node",
        "web_search": "web_search_node",
    },
)
graph.add_edge("direct_generation_node", END)
graph.add_edge("retrieve_node", "is_relevant_node")
graph.add_conditional_edges(
    "is_relevant_node",
    is_relevant_condition,
    {True: "answer_from_context_node", False: "web_search_node"},
)
graph.add_edge("web_search_node", "answer_from_context_node")
graph.add_edge("answer_from_context_node", "check_answer_grounded_node")
graph.add_conditional_edges(
    "check_answer_grounded_node",
    is_grounded_condition,
    {True: "is_answer_useful_node", False: "revise_answer_node"},
)
graph.add_edge("revise_answer_node", "is_answer_useful_node")
graph.add_conditional_edges(
    "is_answer_useful_node",
    is_answer_useful_condition,
    {True: END, False: "rewrite_query_node"},
)
graph.add_edge("rewrite_query_node", "retrieval_decider_node")


workflow = graph.compile(checkpointer=ck_ptr)

if __name__ == "__main__":
    while True:
        human = input("\n\nHuman- ")
        if human == "exit":
            break
        print()
        initial_state = {
            # 'retriever':retriever,
            "user_query": human,
            "k": 3,
            "max_retry_for_revise_answer": 3,
            "max_retry_for_rewrite_query": 2,
        }

        print("RUNNING WORKFLOW")
        for chunk in workflow.stream(
            initial_state, {"configurable": {"thread_id": "2"}},stream_mode="updates"
        ):
            print(f"{list(chunk.keys())[0]} is completed")

        response=workflow.get_state(config={"configurable": {"thread_id": "2"}})
        print("\nAI- ", response.values["generated_response"])
