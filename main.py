import streamlit as st
import openai

# exercise 11
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# exercise 12
from langchain.memory import ConversationBufferWindowMemory

# exercise 13
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import LanceDB
import lancedb
import os
import tempfile

# exercise 15
import sqlite3
import pandas as pd
from datetime import datetime

# exercise 16
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun

# Exercise 17
from langchain.agents import tool
import json

# Exercise 18
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt

from part1 import ex1, ex2, ex3, ex4a, ex4b, ex5, ex6, ex8, ex9, ex10
from part2 import ex11a, ex11b, ex12, ex13, ex14
from part3 import ex15, ex16, ex17
from part4 import ex18

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
	# ex2()
	# ex3()
	# ex4a()
	# ex4b()
	# ex5()
	# ex6()
	# ex8()
	# ex9()
	# ex10()
	# ex11a()
	# ex11b()
	# ex12()
	# ex13()
	# ex14()
	# ex15()
	# ex16()
	# ex17()
	# ex18()

if __name__ == "__main__":
	main()