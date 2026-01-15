import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

st.title('Hello from Colab!')
st.write('This is Streamlit running via Cloudflare Tunnel.')
name = st.text_input('Your name')
if name:
    st.success(f'Welcome, {name}!')
