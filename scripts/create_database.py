from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os 
from preprocessing import * 
from langchain_huggingface import HuggingFaceEmbeddings

# chunking 
text_splitter1 = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=100,separators = ['#','\n\n','\n'])
#text_splitter2 = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap=100,separators = ['#','\n\n','\n'])
list_splits = []
for i in range(len(Documets)) :
    # list_splits.append(text_splitter1.create_documents([document[i]]) + text_splitter2.create_documents([document[i]]))
    list_splits.append(text_splitter1.create_documents([Documets[i]]))

#embedding 
model_name = "hiieu/halong_embedding"  
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device' :"cpu"},
    encode_kwargs=encode_kwargs
)
retriever_list = []

for i in range(len(list_splits)) :
    if not os.path.exists(f"data\database\chroma_db_{i}") : 
        vectorstore = Chroma.from_documents(documents=list_splits[i],
                                    embedding=hf,persist_directory=f"data\database\chroma_db_{i}")
    else : 
        vectorstore = Chroma(persist_directory=f"data\database\chroma_db_{i}",  embedding_function=hf)     
    retriever_list.append(vectorstore.as_retriever()) 





