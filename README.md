# Introduce
Ứng dụng RAG trong việc thiết kế Chatbot hỏi đáp về những vấn đề trong Machine learning 

Dữ liệu là một cuốn sách - trong project này là cuốn machinelearningcoban của Vũ Hữu Tiệp
Thiết kế dựa trên llamaindex, langchain và semantic router 

## Getting Started
To get started with this project, follow the instructions below.
### Note
- The template is simple, so I don't use it. You will interact by CLI
- If you use CPU, Chatbot will respond a little slower from 5 - 20s, because I use an open-source model
- You have to have API keys: Cohere, Langchain and OpenAI
### How to run ? 

1. Clone the repository:

   git clone https://github.com/hoanglvuit/Chatbot-Q-A-about-Machine-learning.git
2. Navigate to the project directory:

   cd Chatbot-Q-A-about-Machine-learning
3. Install dependencies:
 
   pip install -r requirement.txt
   
   cd projectname
   
   pip install .
   
   cd ..
   
4. Create .env file:
 
   echo "" > .env
5. Open .env file and write:
 
   LANGCHAIN_API_KEY = VALUE
    
   OPENAI_API_KEY = VALUE
   
   COHERE_API_KEY  = VALUE
   
   VALUE not string and save as UTF-8
6. Done. You can run like: python scripts/main.py 'svm là gì?'

   

