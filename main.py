import json
import os

# exercise 15
import sqlite3
import tempfile
from datetime import datetime

import lancedb
import matplotlib.pyplot as plt
import openai
import pandas as pd
import streamlit as st

# Exercise 17
# exercise 16
from langchain.agents import AgentExecutor, ConversationalChatAgent, tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# exercise 13
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings

# exercise 11
from langchain.llms import OpenAI

# exercise 12
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import LanceDB

# Exercise 18
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

from part1 import (
    ch1,
    ch4,
    ch4_display,
    ch6,
    ch8,
    ch10,
    ex1,
    ex2,
    ex3,
    ex4a,
    ex4b,
    ex5,
    ex6,
    ex8,
    ex9_basebot,
    ex10_basebot,
)
from part2 import ch11, ch12, ex11a, ex11b, ex12, ex13, ex14_basebot
from part3 import ch15, ex15, ex16_agent_bot, ex17_agent_bot
from part4 import ex18_pandas_AI

# Global Ex 7
os.environ["OPENAI_API_KEY"] = st.secrets["openapi_key"]
openai.api_key = st.secrets["openapi_key"]

# Global ex 13
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)
# ex15
DB_NAME = os.path.join(WORKING_DIRECTORY, "default_db")

st.title("GenAI codecraft workshop")


def main():
    # initialize session state, from ch4
    if "name" not in st.session_state:
        st.session_state.name = "Yoda"

    if "age" not in st.session_state:
        st.session_state.age = 999

    if "gender" not in st.session_state:
        st.session_state.gender = "male"

    if "prompt_template" not in st.session_state:
        st.session_state.prompt_template = "Speak like Yoda from Star Wars for every question that was asked, do not give a direct answer but ask more questions in the style of wise Yoda from Star Wars"

    st.write("Hello world!")
    # ex1()
    # ch1()
    # ex2()
    # ex3()
    # ex4a()
    # ex4b()
    # ch4()
    # ch4_display()
    # ex5()
    # ex6()
    # ch6()
    # ex8()
    # ch8()
    # ex9_basebot()
    # ex10_basebot()
    # ch10()
    # ex11a()
    # ex11b()
    # ch11()
    # ch12()
    # ex12()
    # ex13()
    # ex14_basebot()
    # ex15()
    # ch15()
    # ex16_agent_bot()
    # ex17_agent_bot()
    ex18_pandas_AI()


if __name__ == "__main__":
    main()
