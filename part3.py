import json
import os

# exercise 15
import sqlite3
import tempfile
from datetime import datetime

import lancedb
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
from PIL import Image

# Import from part2
from part2 import chat_completion_stream_prompt, prompt_inputs_form

# os.environ["OPENAI_API_KEY"] = st.secrets["openapi_key"]
# openai.api_key = st.secrets["openapi_key"]

# Global ex 13
cwd = os.getcwd()
WORKING_DIRECTORY = os.path.join(cwd, "database")

if not os.path.exists(WORKING_DIRECTORY):
    os.makedirs(WORKING_DIRECTORY)
# ex15
DB_NAME = os.path.join(WORKING_DIRECTORY, "default_db")


def ex15_initialise():
    # Create or check for the 'database' directory in the current working directory
    # Set DB_NAME to be within the 'database' directory at the top of main.py
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Conversation data table
    cursor.execute(
        """
		CREATE TABLE IF NOT EXISTS data_table (
			id INTEGER PRIMARY KEY,
			date TEXT NOT NULL UNIQUE,
			username TEXT NOT NULL,
			chatbot_ans TEXT NOT NULL,
			user_prompt TEXT NOT NULL,
			tokens TEXT
		)
	"""
    )
    conn.commit()
    conn.close()


def ex15_collect(username, chatbot_response, prompt):
    # collect data from bot
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now()  # Using ISO format for date
    tokens = len(chatbot_response) * 1.3
    cursor.execute(
        """
		INSERT INTO data_table (date, username,chatbot_ans, user_prompt, tokens)
		VALUES (?, ?, ?, ?, ?)
	""",
        (now, username, chatbot_response, prompt, tokens),
    )
    conn.commit()
    conn.close()


# implementing data collection and displaying
def ex15():
    # initialise database first
    ex15_initialise()
    # collect some data
    ex15_collect("yoda", "I am Yoda. The Force is strong with you", "Who are you?")
    # display data
    # Connect to the specified database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Fetch all data from data_table
    cursor.execute("SELECT * FROM data_table")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    df = pd.DataFrame(rows, columns=column_names)
    st.dataframe(df)
    conn.close()


def ch15_initialise():
    # Create or check for the 'database' directory in the current working directory
    # Set DB_NAME to be within the 'database' directory at the top of main.py
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Conversation data table
    cursor.execute(
        """
		CREATE TABLE IF NOT EXISTS ch15_data_table (
			id INTEGER PRIMARY KEY,
			date TEXT NOT NULL UNIQUE,
			username TEXT NOT NULL,
			user_prompt TEXT NOT NULL,
            chatbot_ans TEXT NOT NULL,
            vs_prompt TEXT NOT NULL,
			prompt_tokens TEXT,
            response_tokens TEXT,
            vs_tokens TEXT
		)
	"""
    )
    conn.commit()
    conn.close()


def ch15_collect(username, prompt, chatbot_response, vs_prompt):
    # collect data from bot
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    now = datetime.now()  # Using ISO format for date
    token_multiplier = 1.3
    prompt_tokens = len(prompt) * token_multiplier
    response_tokens = len(chatbot_response) * token_multiplier
    vs_tokens = len(vs_prompt) * token_multiplier
    cursor.execute(
        """
		INSERT INTO ch15_data_table (date, username, user_prompt, chatbot_ans, vs_prompt, prompt_tokens, response_tokens, vs_tokens)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	""",
        (
            now,
            username,
            prompt,
            chatbot_response,
            vs_prompt,
            prompt_tokens,
            response_tokens,
            vs_tokens,
        ),
    )
    conn.commit()
    conn.close()


def ch15_display():
    # display data
    # Connect to the specified database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Fetch all data from data_table
    cursor.execute("SELECT * FROM ch15_data_table")
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    df = pd.DataFrame(rows, columns=column_names)
    st.dataframe(df)
    conn.close()


# Exercise 14 with collection
def ch15():
    # initialise database first
    ch15_initialise()

    # Prompt_template form from ex14
    prompt_template = PromptTemplate(
        input_variables=["occupation", "topic", "age"],
        template="""Imagine you are a {occupation} who is an expert on the  topic of {topic} , you are going to help , teach and provide information
						to the person who is {age} years old, if you do not not know the answer, you must tell the person , do not make any answer up""",
    )
    dict_inputs = prompt_inputs_form()
    if dict_inputs:
        input_prompt = prompt_template.format(
            occupation=dict_inputs["occupation"],
            topic=dict_inputs["topic"],
            age=dict_inputs["age"],
        )
        st.session_state.input_prompt = input_prompt

    if "input_prompt" not in st.session_state:
        st.session_state.input_prompt = "Speak like Yoda from Star Wars"

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=5)

    # step 1 save the memory from your chatbot
    # step 2 integrate the memory in the prompt_template (st.session_state.prompt_template) show a hint
    memory_data = st.session_state.memory.load_memory_variables({})
    st.write(memory_data)
    st.session_state.prompt_template = f"""
st.session_state.input_prompt: {st.session_state.input_prompt}

This is the last conversation history
{memory_data}

"""
    st.write("new prompt template: ", st.session_state.prompt_template)

    st.session_state.vectorstore = vectorstore_creator()

    # Initialize chat history
    if "msg" not in st.session_state:
        st.session_state.msg = []

    # Showing Chat history
    for message in st.session_state.msg:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    try:
        #
        if prompt := st.chat_input("What is up?"):
            # query information
            if st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(prompt)
                docs = docs[0].page_content
                # add your query prompt
                vs_prompt = f"""You should reference this search result to help your answer,
								{docs}
								if the search result does not anwer the query, please say you are unable to answer, do not make up an answer"""
            else:
                vs_prompt = ""
            # add query prompt to your memory prompt and send it to LLM
            st.session_state.prompt_template = (
                st.session_state.prompt_template + vs_prompt
            )
            # set user prompt in chat history
            st.session_state.msg.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # streaming function
                for response in chat_completion_stream_prompt(prompt):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.msg.append({"role": "assistant", "content": full_response})
            st.session_state.memory.save_context(
                {"input": prompt}, {"output": full_response}
            )

            # collect data from bot
            ch15_collect("yoda", prompt, full_response, vs_prompt)

            # display data from bot
            ch15_display()

    except Exception as e:
        st.error(e)


# smart agents accessing the internet for free
# https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/search_and_chat.py
def ex16_agent_bot():
    st.title("🦜 LangChain: Chat with internet search")

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="chat_history",
        output_key="output",
    )
    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(
                    f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
                ):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

    if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
        st.chat_message("user").write(prompt)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, streaming=True
        )
        tools = [DuckDuckGoSearchRun(name="Search")]
        chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = executor(prompt, callbacks=[st_cb])
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response[
                "intermediate_steps"
            ]


def upload_file_streamlit():
    def get_file_extension(file_name):
        return os.path.splitext(file_name)[1]

    st.subheader("Upload your docs")

    # Streamlit file uploader to accept file input
    uploaded_file = st.file_uploader("Choose a file", type=["docx", "txt", "pdf"])

    if uploaded_file:
        # Reading file content
        file_content = uploaded_file.read()

        # Determine the suffix based on uploaded file's name
        file_suffix = get_file_extension(uploaded_file.name)

        # Saving the uploaded file temporarily to process it
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
            temp_file.write(file_content)
            temp_file.flush()  # Ensure the data is written to the file
            temp_file_path = temp_file.name
        return temp_file_path


# exercise 13 - split and chunk, embeddings and storing in vectorstores for reference
def vectorstore_creator():
    # WORKING_DIRECTORY set above in the main.py
    # Process the temporary file using UnstructuredFileLoader (or any other method you need)
    embeddings = OpenAIEmbeddings()
    db = lancedb.connect(WORKING_DIRECTORY)
    table = db.create_table(
        "my_table",
        data=[
            {
                "vector": embeddings.embed_query("Query unsuccessful"),
                "text": "Query unsuccessful",
                "id": "1",
            }
        ],
        mode="overwrite",
    )
    # st.write(temp_file_path)
    temp_file_path = upload_file_streamlit()
    if temp_file_path:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load_and_split()
        db = LanceDB.from_documents(documents, embeddings, connection=table)
        return db


# agents ,vectorstores, wiki
# https://python.langchain.com/docs/modules/agents/how_to/custom_agent_with_tool_retrieval
# note tool
@tool("Document search")
def document_search(query: str) -> str:
    # this is the prompt to the tool itself
    "Use this function first to search for documents pertaining to the query before going into the internet"
    docs = st.session_state.vectorstore.similarity_search(query)
    docs = docs[0].page_content
    json_string = json.dumps(docs, ensure_ascii=False, indent=4)
    return json_string


# combine vector store and internet search
def ex17_agent_bot():
    st.title("🦜 LangChain: Chat with internet search")

    st.session_state.vectorstore = vectorstore_creator()

    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=msgs,
        return_messages=True,
        memory_key="chat_history",
        output_key="output",
    )
    if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")
        st.session_state.steps = {}

    avatars = {"human": "user", "ai": "assistant"}
    for idx, msg in enumerate(msgs.messages):
        with st.chat_message(avatars[msg.type]):
            # Render intermediate steps if any were saved
            for step in st.session_state.steps.get(str(idx), []):
                if step[0].tool == "_Exception":
                    continue
                with st.status(
                    f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
                ):
                    st.write(step[0].log)
                    st.write(step[1])
            st.write(msg.content)

    if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
        st.chat_message("user").write(prompt)

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, streaming=True
        )
        tools = [document_search, DuckDuckGoSearchRun(name="Internet Search")]
        chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = executor(prompt, callbacks=[st_cb])
            st.write(response["output"])
            st.session_state.steps[str(len(msgs.messages) - 1)] = response[
                "intermediate_steps"
            ]
