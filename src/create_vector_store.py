import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

data={}
with open('../data/articles.json','r') as f:
    data=json.load(f)
    
    
documents=[]
for key,value in zip(data.keys(),data.values()):
    documents.append(Document(metadata={"Article":key},page_content=f"{key} \n: {value}"))


data={}
with open('../data/penal_code_sections.json','r') as f:
    data=json.load(f)


for key,value in data.items():
    documents.append(Document(metadata={"Section":key},page_content=f"{key} \n{value}"))
    
vector_store=FAISS.from_documents(documents,embedding=embeddings)
vector_store.save_local("../data/constitution_and_ipc.faiss")