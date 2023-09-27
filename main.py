import streamlit as st
import openai

# exercise 11
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# exercis 12
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

# Global ex 13
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
	os.makedirs(WORKING_DIRECTORY)
# ex15
DB_NAME = os.path.join(WORKING_DIRECTORY, "default_db")


# Exercise 1 : Hello World and Input
def ex1():
	st.write("Hello World")
	name = st.text_input("Enter your name")
	# only prints the Hello {name} if input box is not empty
	if name:
		st.write("Hello " + name)

st.title("ITD workshop file")

st.write("Hello World!")