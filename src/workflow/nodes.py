from langchain_core.messages import HumanMessage, SystemMessage
from workflow.state import schema
from workflow.config import (
    main_model, critic_model, judge_model, query_rewrite_model,
    retriever, tavily_tool
)
from workflow.schemas import (
    parser_for_retrieval_decider_node,
    parser_for_is_relevant_node,
    parser_for_answer_from_context_node,
    parser_for_schema_for_check_answer_grounded_node,
    parser_for_revise_answer_node,
    parser_for_is_answer_useful_node,
    parser_for_rewrite_query_node,
    parser_for_retriever_query_node,
    parser_for_web_search_query_node
)
from workflow.prompts import (
    sys_prompt_for_retrieval_decider_node,
    sys_prompt_for_is_relevant_node,
    sys_prompt_for_answer_from_context_node,
    sys_prompt_for_check_answer_grounded_node,
    sys_prompt_for_revise_answer_node,
    sys_prompt_for_is_answer_useful_node,
    sys_prompt_for_rewrite_query_node,
    sys_prompt_for_retriever_query_node,
    sys_prompt_for_web_search_query_node
)
from langgraph.types import Send

def retrieval_decider_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retrieval_decider_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_retrieval_decider_node.invoke(
        main_model.invoke(inp).content
    ).retrieval_required
    return {"retrieval_required": res}

def generate_retriever_query_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_retriever_query_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_retriever_query_node.invoke(
        main_model.invoke(inp).content
    ).retriever_query
    return {"retriever_query": res}

def retrieve_node(state: schema):
    global retriever
    query = state.get("retriever_query") or state["user_query"]
    retrieved_contexts = retriever.invoke(query, k=state["k"])
    return {"retrieved_contexts": [i.page_content for i in retrieved_contexts]}

def direct_generation_node(state: schema):
    res = main_model.invoke(state["user_query"]).content
    return {"generated_response": res}

def fanout_relevant_node(state:schema):
    contexts = state["retrieved_contexts"]
    lst = []

    for context in contexts:
        lst.append(Send("is_relevant_node",{"context":context,"user_query":state['user_query']}))
    
    return lst

def is_relevant_node(inp):
    sys_prompt = SystemMessage(content=sys_prompt_for_is_relevant_node)
    hmn_prompt = f"Query - {inp['user_query']}" + f"\n Context - \n {inp['context']}"

    res = parser_for_is_relevant_node.invoke(
            main_model.invoke(
                [sys_prompt, HumanMessage(content=hmn_prompt)]
            ).content
        )
    
    if res.is_relevant_context :
        return {"relevant_contexts": [inp['context']] }
    else:
        return {}

def aggregate_relevance(state):
    # plain pass-through node, just a sync point
    return {}

def answer_from_context_node(state: schema):
    contexts = [x.content for x  in state["relevant_contexts"]]
    sys_prompt = SystemMessage(content=sys_prompt_for_answer_from_context_node)

    context = ""
    for i in contexts:
        context += i
        context += "\n"

    hmn_prompt = HumanMessage(
        content=f"Query - {state['user_query']} \n\n Contexts - \n {context}"
    )
    inp = [sys_prompt,hmn_prompt]

    res = parser_for_answer_from_context_node.invoke(
        main_model.invoke(inp).content
    ).response
    return {"generated_response": res}

def check_answer_grounded_node(state: schema):
    contexts = [x.content for x  in state["relevant_contexts"]]
    sys_prompt = SystemMessage(content=sys_prompt_for_check_answer_grounded_node)
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

def revise_answer_node(state: schema):
    contexts =[x.content for x  in state["relevant_contexts"]]
    context = ""
    for i in contexts:
        context += i
        context += "\n"
    generated_response = state["generated_response"]
    user_query = state["user_query"]
    evidence = state["evidence"]

    sys_prompt = SystemMessage(content=sys_prompt_for_revise_answer_node)

    human_pr = HumanMessage(
        content=f"""Query - {user_query} \n\n Generated Response - {generated_response} \n\n Contexts - {context} \n\n Evidence - {evidence}"""
    )

    revised_answer = parser_for_revise_answer_node.invoke(
        critic_model.invoke([sys_prompt, human_pr]).content
    )

    return {
        "generated_response": revised_answer.revised_response,
        "max_retry_for_revise_answer": state["max_retry_for_revise_answer"] - 1,
    }

def is_answer_useful_node(state: schema):
    user_query = state["user_query"]
    generated_response = state["generated_response"]

    sys_prompt = SystemMessage(content=sys_prompt_for_is_answer_useful_node)
    human_pr = HumanMessage(
        content=f"Query - {user_query} \n\n Generated Response - {generated_response}"
    )

    res = parser_for_is_answer_useful_node.invoke(
        judge_model.invoke([sys_prompt, human_pr]).content
    )
    return {"is_answer_useful": res.is_useful}

def rewrite_query_node(state: schema):
    sys_prompt = SystemMessage(content=sys_prompt_for_rewrite_query_node)
    human_pr = state["user_query"]

    res = parser_for_rewrite_query_node.invoke(
        query_rewrite_model.invoke([sys_prompt, human_pr]).content
    )
    return {
        "user_query": res.updated_query,
        "max_retry_for_rewrite_query": state["max_retry_for_rewrite_query"] - 1,
    }

def generate_web_search_query_node(state: schema):
    inp = [
        SystemMessage(content=sys_prompt_for_web_search_query_node),
        HumanMessage(content=f"User Query - {state['user_query']}"),
    ]
    res = parser_for_web_search_query_node.invoke(
        main_model.invoke(inp).content
    ).web_search_query
    return {"web_search_query": res}

def web_search_node(state: schema):
    query = state.get("web_search_query") or state["user_query"]
    x = tavily_tool.invoke(query)
    res = []
    for r in x["results"]:
        p = f"Source - {r['url']} \n title - {r['title']} \n {r['content']}"
        res.append(p)
    return {"relevant_contexts": res, "web_searched": True}
