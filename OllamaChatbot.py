from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
#from langchain_ollama import OllamaLLM
import streamlit as st
import os 

from dotenv import load_dotenv

load_dotenv()

#Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With Ollama"


##Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.Please response to the user question"),
        ("user", "Question:{question}"),
    ]
)

def generate_response(question,llm,temperature,max_tokens):
    llm=Ollama(model=llm)
    output_parser=StrOutputParser()
    chain=prompt | llm | output_parser
    answer=chain.invoke({"question":question})
    return answer


## Streamlit 
## Titel of the app
st.title("Q&A Chatbot with Ollama")

## Sidebar for settings
st.sidebar.title("Settings")
#api_key=st.sidebar.text_input("Enter your OpenAI API Key:",type="password")

## Drop down to select various Open AI models
llm=st.sidebar.selectbox("Select Ollama model",["deepseek-r1","llama3.2","mistral"])

temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.5)
max_tokens=st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)


## Main interface for user input
st.write("Go ahead and ask any question!")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please enter a question to get a response.")