from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel,Field
from pydantic import Field
from typing import TypedDict,List,Literal,Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
import json
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
import os
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch

load_dotenv()


# Models
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

query_rewrite_model=ChatOllama(model='gemma4:31b-cloud')
judge_model=ChatOllama(model='gemma4:31b-cloud')
critic_model=ChatOllama(model='gemma4:31b-cloud')
main_model=ChatOllama(model='gemma4:31b-cloud')

# vector_store
vector_store=FAISS.load_local("./data/constitution_and_ipc.faiss",embeddings=embeddings,allow_dangerous_deserialization=True)
retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":3})

tavily_tool=TavilySearch(max_results=3)

# schema
class schema(TypedDict):
    # Not including retriever because it is inserilisable so it is giving error with inmemorystore
    # retriever:BaseRetriever
    retrieval_required:Literal['retrieval','web_search','None']
    web_searched:bool
    user_query:str
    retrieved_contexts:List[str]
    relevant_contexts:List[str]
    answer_for_query:str
    generated_response:str
    is_grounded:bool
    is_supported:bool
    is_answer_useful:bool
    evidence:str
    k:Optional[int]=Field(default=3)
    max_retry_for_revise_answer:Optional[int]=Field(default=3)
    max_retry_for_rewrite_query:Optional[int]=Field(default=2)
    
    
# Nodes
class schema_for_retrieval_decider_node(BaseModel):
    retrieval_required:Literal['retrieval','web_search','None']=Field(...)

parser_for_retrieval_decider_node=PydanticOutputParser(pydantic_object=schema_for_retrieval_decider_node)

sys_prompt_for_retrieval_decider_node=f"You are an AI Assistant . In retriever Indian Penal Code and Consititution of India is present . User will give you some query your task is to determine whether retrieval is required or websearch is required or none of them is required  \n Output Format -{parser_for_retrieval_decider_node.get_format_instructions()}"


def retrieval_decider_node(state:schema):
    inp=[SystemMessage(content=sys_prompt_for_retrieval_decider_node),HumanMessage(content=f"User Query - {state['user_query']}")]
    res=parser_for_retrieval_decider_node.invoke(main_model.invoke(inp).content).retrieval_required
    return {'retrieval_required':res}

def retrieve_node(state:schema):
    global retriever
    retrieved_contexts=retriever.invoke(state['user_query'],k=state['k'])
    return {'retrieved_contexts':[i.page_content for i in retrieved_contexts]}

def direct_generation_node(state:schema):
    res=main_model.invoke(state['user_query']).content
    return {"generated_response":res}

class schema_for_is_relevant_node(BaseModel):
    is_relevant_context:bool
parser_for_is_relevant_node=PydanticOutputParser(pydantic_object=schema_for_is_relevant_node)

def is_relevant_node(state:schema):
    contexts=state['retrieved_contexts']
    sys_prompt=SystemMessage(content=f"""User will feed you a query  and a retrieved context for that query your task is to tell whether retrieved context is relevent for answering the given query or not \n Output format - {parser_for_is_relevant_node.get_format_instructions()}""")
    hmn_prompt=f"Query - {state['user_query']}"
    lst=[]
    for context in contexts:
        hmn_prompt_dash=hmn_prompt+f"\n Context - {context}"
        res=parser_for_is_relevant_node.invoke(main_model.invoke([sys_prompt,HumanMessage(content=hmn_prompt_dash)]).content)

        lst.append(res.is_relevant_context)

    return {'relevant_contexts':[contexts[i] for i in range(len(contexts)) if lst[i]]}


class schema_for_answer_from_context_node(BaseModel):
    response:str=Field(...,description="Response for given query")
    
parser_for_answer_from_context_node=PydanticOutputParser(pydantic_object=schema_for_answer_from_context_node)

def answer_from_context_node(state:schema):
    contexts=state['relevant_contexts']
    sys_prompt_for_answer_from_context_node=SystemMessage(content=f"""User will give you a query and context for query your work is to give answer user query based on the context provided. \n Output format - {parser_for_answer_from_context_node.get_format_instructions()}""")
    
    context=""
    for i in contexts:
        context+=i
        context+="\n"
        
    hmn_prompt=HumanMessage(content=f"Query - {state['user_query']} \n\n Contexts - {context}")
    inp=[sys_prompt_for_answer_from_context_node,hmn_prompt]

    res=parser_for_answer_from_context_node.invoke(main_model.invoke(inp).content).response
    return {"generated_response":res}

class schema_for_check_answer_grounded_node(BaseModel):
    is_grounded:Literal['fully_supported','not_fully_supported']
    evidence:str=Field(...,description="Proof that answer is not supported by given contexts")

parser_for_schema_for_check_answer_grounded_node=PydanticOutputParser(pydantic_object=schema_for_check_answer_grounded_node)

def check_answer_grounded_node(state:schema):
    contexts=state['relevant_contexts']
    sys_prompt=SystemMessage(content=f"You are verifying whether the ANSWER is supported by the CONTEXT.\n Output format - {parser_for_schema_for_check_answer_grounded_node.get_format_instructions()}")
    context=""
    for i in contexts:
        context+=i
        context+="\n"

    human_pr=HumanMessage(content=f"Answer - {state['generated_response']} \n Contexts - {context}")

    res=parser_for_schema_for_check_answer_grounded_node.invoke(main_model.invoke([sys_prompt,human_pr]).content)
    
    return {'is_grounded':res.is_grounded,'evidence':res.evidence}


class schema_for_revise_answer_node(BaseModel):
    revised_response:str=Field(...,description="Response for given query")
    
parser_for_revise_answer_node=PydanticOutputParser(pydantic_object=schema_for_revise_answer_node)


def revise_answer_node(state:schema):
    contexts=state['relevent_contexts']
    context=""
    for i in contexts:
        context+=i
        context+="\n"
    generated_response=state['generated_response']
    user_query=state['user_query']
    evidence=state['evidence']
    
    sys_prompt=SystemMessage(content=f"User will give you a query ,an answer for the given query, context for the given query, response generated by llm and a evidence that the response is not fully supported by the given contexts Your task is to revise the answer such that revised answer is fully suported by the given contexts\n Output  format- {parser_for_revise_answer_node.get_format_instructions()}")
    
    
    human_pr=HumanMessage(content=f"Query - {user_query} \n\n Generated Response - {generated_response} \n\n Contexts - {x} \n\n Evidence - {evidence}")

    revised_answer=parser_for_revise_answer_node.invoke(critic_model.invoke([sys_prompt,human_pr]).content)
    
    return {'generated_response':res.revised_response,'max_retry_for_revise_answer':state['max_retry_for_revise_answer']-1}


class schema_for_is_answer_useful_node(BaseModel):
    is_useful:bool=Field(...,description="Boolean response whether answer is useful or not")

parser_for_is_answer_useful_node=PydanticOutputParser(pydantic_object=schema_for_is_answer_useful_node)

def is_answer_useful_node(state:schema):
    user_query=state['user_query']
    generated_response=state['generated_response']

    sys_prompt=SystemMessage(content=f"User will give you a query and a response of the query generated by llm your task is to tell whether the response solves users query or not. Output format - {parser_for_is_answer_useful_node.get_format_instructions()}")
    human_pr=HumanMessage(content=f"Query - {user_query} \n\n Generated Response - {generated_response}")

    res=parser_for_is_answer_useful_node.invoke(judge_model.invoke([sys_prompt,human_pr]).content)
    return {'is_answer_useful':res.is_useful}


class schema_for_rewrite_query_node(BaseModel):
    updated_query:str

parser_for_rewrite_query_node=PydanticOutputParser(pydantic_object=schema_for_rewrite_query_node)

def rewrite_query_node(state:schema):
    sys_prompt=SystemMessage(content=f"You are a assistant whose work is to modify user query. Our vector database contains documents related to indian constitution and other acts . Users query is too vague to retrieve relevent context to answer the query your task is to rewrite the query such that it is optimised for retrieving correct context. \n Output Format - {parser_for_rewrite_query_node.get_format_instructions()}")
    human_pr=state['user_query']

    res=parser_for_rewrite_query_node.invoke(query_rewrite_model.invoke([sys_prompt,human_pr]).content)
    return {'user_query':res.updated_query,'max_retry_for_rewrite_query':state['max_retry_for_rewrite_query']-1}

def web_search_node(state:schema):
    x=tavily_tool.invoke(state['user_query'])
    res=[]
    for r in x['results']:
        p=f"Source - {r['url']} \n title - {r['title']} \n {r['content']}"
        res.append(p)
    return {"relevant_contexts":res,"web_searched":True}

def retrieval_decider_condition(state:schema):
    return state['retrieval_required']

def is_relevant_condition(state:schema):
    return len(state['relevant_contexts'])>0

def is_grounded_condition(state:schema):
    return state['is_grounded'] and state['max_retry_for_revise_answer']>0

def is_answer_useful_condition(state:schema):
    return state['is_answer_useful'] and state['max_retry_for_rewrite_query']>0


ck_ptr=InMemorySaver()

graph=StateGraph(state_schema=schema)

# adding nodes
# TODO - add a websearch node if none of generated contexts are relevant 
graph.add_node('retrieval_decider_node',retrieval_decider_node)
graph.add_node('retrieve_node',retrieve_node)
graph.add_node('direct_generation_node',direct_generation_node)
graph.add_node('is_relevant_node',is_relevant_node)
graph.add_node('answer_from_context_node',answer_from_context_node)
graph.add_node('check_answer_grounded_node',check_answer_grounded_node)
graph.add_node('revise_answer_node',revise_answer_node)
graph.add_node('is_answer_useful_node',is_answer_useful_node)
graph.add_node('rewrite_query_node',rewrite_query_node)
graph.add_node('web_search_node',web_search_node)


graph.add_edge(START,'retrieval_decider_node')
graph.add_conditional_edges('retrieval_decider_node',retrieval_decider_condition,{"retrieval":'retrieve_node',"None":'direct_generation_node',"web_search":"web_search_node"})
graph.add_edge('direct_generation_node',END)
graph.add_edge('retrieve_node','is_relevant_node')
graph.add_conditional_edges('is_relevant_node',is_relevant_condition,{True:'answer_from_context_node',False:'web_search_node'})
graph.add_edge('web_search_node','answer_from_context_node')
graph.add_edge('answer_from_context_node','check_answer_grounded_node')
graph.add_conditional_edges('check_answer_grounded_node',is_grounded_condition,{True:'is_answer_useful_node',False:'revise_answer_node'})
graph.add_edge('revise_answer_node','is_answer_useful_node')
graph.add_conditional_edges('is_answer_useful_node',is_answer_useful_condition,{True:END,False:'rewrite_query_node'})
graph.add_edge('rewrite_query_node','retrieval_decider_node')


workflow=graph.compile(checkpointer=ck_ptr)

if __name__=="__main__":
    while True:
        human=input("Human- ")
        if human=="exit":
            break
        print()
        initial_state={
            # 'retriever':retriever,
            'user_query':human,
            'k':3,
            'max_retry_for_revise_answer':3,
            'max_retry_for_rewrite_query':2
        }
        
        response=workflow.invoke(initial_state,{"configurable": {"thread_id": "thread-1"}})
        
        print("AI- ",response['generated_response'])
    