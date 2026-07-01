from pydantic import BaseModel, Field
from typing import Literal, List
from langchain_core.output_parsers import PydanticOutputParser

class schema_for_retrieval_decider_node(BaseModel):
    retrieval_required: Literal["retrieval", "web_search", "None"] = Field(...)

parser_for_retrieval_decider_node = PydanticOutputParser(
    pydantic_object=schema_for_retrieval_decider_node
)

class schema_for_is_relevant_node(BaseModel):
    is_relevant_context: bool

parser_for_is_relevant_node = PydanticOutputParser(
    pydantic_object=schema_for_is_relevant_node
)

class schema_for_answer_from_context_node(BaseModel):
    response: str = Field(..., description="Response for given query")

parser_for_answer_from_context_node = PydanticOutputParser(
    pydantic_object=schema_for_answer_from_context_node
)

class schema_for_check_answer_grounded_node(BaseModel):
    is_grounded: Literal["fully_supported", "not_fully_supported"]
    evidence: str = Field(
        ..., description="Proof that answer is not supported by given contexts"
    )

parser_for_schema_for_check_answer_grounded_node = PydanticOutputParser(
    pydantic_object=schema_for_check_answer_grounded_node
)

class schema_for_revise_answer_node(BaseModel):
    revised_response: str = Field(..., description="Response for given query")

parser_for_revise_answer_node = PydanticOutputParser(
    pydantic_object=schema_for_revise_answer_node
)

class schema_for_is_answer_useful_node(BaseModel):
    is_useful: bool = Field(
        ..., description="Boolean response whether answer is useful or not"
    )

parser_for_is_answer_useful_node = PydanticOutputParser(
    pydantic_object=schema_for_is_answer_useful_node
)

class schema_for_rewrite_query_node(BaseModel):
    updated_query: str

parser_for_rewrite_query_node = PydanticOutputParser(
    pydantic_object=schema_for_rewrite_query_node
)

class schema_for_retriever_query_node(BaseModel):
    retriever_queries: List[str] = Field(..., description="Optimized search queries for the internal vector database retrieval. Generate at most 3 queries.")

parser_for_retriever_query_node = PydanticOutputParser(
    pydantic_object=schema_for_retriever_query_node
)

class schema_for_web_search_query_node(BaseModel):
    web_search_queries: List[str] = Field(..., description="Optimized search queries for the web search engine. Generate at most 3 queries.")

parser_for_web_search_query_node = PydanticOutputParser(
    pydantic_object=schema_for_web_search_query_node
)
