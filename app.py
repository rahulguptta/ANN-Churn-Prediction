import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()

# langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatboat"


models = [
    "meta-llama/llama-3.3-70b-versatile",
    "meta-llama/llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an assistance to answer user's query."),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, api_key,llm,temperature,max_tokens):
  api_key = api_key
  llm = ChatGroq(model = llm,
                 groq_api_key = api_key)
  output_parser = StrOutputParser()
  chain = prompt|llm|output_parser
  answer = chain.invoke({'question':question})
  return answer

# Title of the app
st.title("Simple Q&A Chatbots")

## sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("key", type="password")

# select the model
engine = st.sidebar.selectbox("model", models)

## Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## MAin interface for user input
st.write("Goe ahead and ask any question")
user_input=st.text_input("You:")

if user_input and api_key:
    response=generate_response(user_input,api_key,engine,temperature,max_tokens)
    st.write(response)

elif user_input:
    st.warning("Please enter api Key in the sider bar")
else:
    st.write("Please provide the user input")
