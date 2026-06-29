from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

if not os.environ.get('OLLAMA_API_KEY'):
    raise ValueError("No ollama api key")
else:
    OLLAMA_API_KEY = os.environ['OLLAMA_API_KEY']

# Models
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

query_rewrite_model = ChatOllama(
    model="nemotron-3-ultra:cloud",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)
judge_model = ChatOllama(
    model="nemotron-3-ultra:cloud",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)
critic_model = ChatOllama(
    model="nemotron-3-ultra:cloud",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)
main_model = ChatOllama(
    model="nemotron-3-ultra:cloud",
    base_url="https://ollama.com",
    client_kwargs={"headers": {"Authorization": f"Bearer {OLLAMA_API_KEY}"}},
)

# vector_store
vector_store = FAISS.load_local(
    "C:\\Users\\panka\\genai_project\\constitution_rag\\data\\constitution_and_ipc.faiss",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# search tools
tavily_tool = TavilySearch(max_results=3)
