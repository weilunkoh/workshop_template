import streamlit as st
import openai
import pandas as pd
import os

# Exercise 18
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt


# PandasAI- A smart agent that can do visual analytics
def ex18_pandas_AI():
	st.title("pandas-ai streamlit interface")

	# Upload CSV file using st.file_uploader
	uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
	if "openai_key" not in st.session_state:
		st.session_state.openai_key = st.secrets["openapi_key"]
		st.session_state.prompt_history = []
		st.session_state.df = None

	if uploaded_file is not None:
		try:
			df = pd.read_csv(uploaded_file)
			st.session_state.df = df
		except Exception as e:
			st.write("There was an error processing the CSV file.")
			st.write(e)

	else:
		st.session_state.df = pd.DataFrame(
			{
				"country": [
					"United States",
					"United Kingdom",
					"France",
					"Germany",
					"Italy",
					"Spain",
					"Canada",
					"Australia",
					"Japan",
					"China",
				],
				"gdp": [
					19294482071552,
					2891615567872,
					2411255037952,
					3435817336832,
					1745433788416,
					1181205135360,
					1607402389504,
					1490967855104,
					4380756541440,
					14631844184064,
				],
				"happiness_index": [
					6.94,
					7.16,
					6.66,
					7.07,
					6.38,
					6.4,
					7.23,
					7.22,
					5.87,
					5.12,
				],
			}
		)
	chart_path = os.path.join("exports/charts")
	with st.form("Question"):
		question = st.text_input("Question", value="", type="default")
		submitted = st.form_submit_button("Submit")
		if submitted:
			with st.spinner():
				llm = OpenAI(api_token=st.session_state.openai_key)
				df = SmartDataframe(
					st.session_state.df,
					config={
						"llm": llm,
						"save_charts_path": chart_path,
						"save_charts": True,
						"verbose": True,
					},
				)
				response = df.chat(
					question
				)  # Using 'chat' method based on your context

				# Display the textual response (if any):
				if response:
					st.write(response)
				chart_path = os.path.join("exports/charts", "temp_chart.png")
				if os.path.exists(chart_path):
					st.image(
						chart_path, caption="Generated Chart", use_column_width=True
					)
				# Append the question to the history:
				st.session_state.prompt_history.append(question)

	if st.session_state.df is not None:
		st.subheader("Current dataframe:")
		st.write(st.session_state.df)

	st.subheader("Prompt history:")
	st.write(st.session_state.prompt_history)

	if st.button("Clear"):
		st.session_state.prompt_history = []
		st.session_state.df = None
