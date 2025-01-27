
from langchain_chroma import Chroma
from preprocessing import * 

retriever_list = []
for i in range(len(Documets)) : 
    vectorstore = Chroma(persist_directory=f"data\database\chroma_db_{i}") 
    retriever_list.append(vectorstore.as_retriever()) 
