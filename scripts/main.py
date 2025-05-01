import sys  # Add this import
from route import route_layer
from create_database import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
# os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
# os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def generate(question):
    topic = route_layer(question).name
    ind = int(topic)
    print(ind)
    template = """Hãy hiểu context bên dưới và hãy hiểu câu hỏi. Xem thông tin chỉ có trong context có thể dùng để trả lời một cách đầy đủ cho câu hỏi hay không.
    - Nếu không, thì trả lời: "Xin lỗi, tôi không có thông tin về câu hỏi này" và đưa ra 2 link trên internet có khả năng trả lời câu hỏi
    - Nếu có, thì hãy dùng nó để trả lời câu hỏi.
    Context: {context}
    Câu hỏi: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Post-processing
    def format_docs(docs):
        return '<' + "\n".join(doc.page_content for doc in docs) + '>'

    # Chain
    rag_chain = (
        {"context": retriever_list[ind] | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(question)
    return answer

# Check if a command-line argument is provided
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <question>")
    else:
        question = sys.argv[1]  # Get the question from the command line
        answer = generate(question)
        print(answer)