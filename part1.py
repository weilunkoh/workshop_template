import openai
import streamlit as st

# openai.api_key = st.secrets["openapi_key"]


def ex1():
    # Exercise 1 : Functions
    st.write("Hello World")
    # only prints the Hello {name} if input box is not empty
    name = st.text_input("Enter your name")
    if name:
        st.write("Hello " + name)


def ch1():
    name = st.text_input("Enter your name", "John")
    age = st.number_input("Enter your age", 30)
    gender = st.selectbox("State your gender", ["Male", "Female"])
    if name:
        st.write("Hello " + name)


def ex2():
    gender = st.selectbox("State your gender", ["Male", "Female"])
    age = int(st.text_input("State your age", 18))
    photo = st.camera_input("Smile! take a picture here.")

    # conditional logic to run different statements
    if age >= 21 and gender == "Male":
        st.write("You are a male adult")
    elif age < 21 and gender == "Male":
        st.write("You are a young boy")
    elif age >= 21 and gender == "Female":
        st.write("You are a female adult")
    elif age < 21 and gender == "Female":
        st.write("You are a young girl")

    if photo:
        st.write("Here is your photo: ")
        st.image(photo)
    else:
        st.write("No photo taken")


def ex3():
    # Data list
    fruits = ["apple", "banana", "orange"]

    # For loop to show list
    for fruit in fruits:
        st.write(fruit)

    # Dictionary
    person = {"name": "John", "age": 30, "gender": "Male", "city": "New York"}

    # Print out the items in the dictionary
    st.write("Here is your *person* dictionary: ")
    st.write(person)

    # for loop to show dictionary list
    st.write("You can also show individual items in the dictionary like this: ")
    for key, value in person.items():
        st.write(key + ": " + str(value))

    # get user input to update the dictionary
    name = st.text_input("Enter your name", "John")
    age = st.text_input("State your age", 30)
    gender = st.selectbox("State your gender", ["Male", "Female"])
    city = st.text_input("State your city", "New York")
    person["name"] = name
    person["age"] = age
    person["gender"] = gender
    person["city"] = city

    st.write("Here is your updated *person* dictionary: ")
    st.write(person)


def ex4a():
    st.subheader("Session Data:")
    if "session_data" not in st.session_state:
        st.session_state.session_data = ["alpha", "omega"]

    if "name" not in st.session_state:
        st.session_state.name = ""

    if "age" not in st.session_state:
        st.session_state.age = ""

    if "gender" not in st.session_state:
        st.session_state.gender = ""

    # For loop to show list
    for data in st.session_state.session_data:
        st.write("session_data: ", data)

    # WL Code
    # st.write("session_data: ", st.session_state.session_data)

    st.write("name: ", st.session_state.name)
    st.write("age: ", st.session_state.age)
    st.write("gender: ", st.session_state.gender)


def ex4b():
    st.subheader("Session Data:")
    user_name = st.text_input("Enter your name")
    user_age = st.text_input("State your age")
    user_gender = st.selectbox("State your gender", ["", "Male", "Female"])

    if user_name:
        st.session_state.name = user_name
        st.write("name: ", st.session_state.name)
    if user_age:
        st.session_state.age = int(user_age)
        st.write("age: ", st.session_state.age)
    if user_gender:
        st.session_state.gender = user_gender
        st.write("gender: ", st.session_state.gender)


def ch4():
    st.subheader("Session Data:")
    user_name = st.text_input("Enter your name")
    user_age = st.text_input("State your age")
    user_gender = st.selectbox("State your gender", ["", "Male", "Female"])
    prompt_template = st.text_area("Enter your prompt template")

    if user_name:
        st.session_state.name = user_name
        st.write("name: ", st.session_state.name)
    if user_age:
        st.session_state.age = int(user_age)
        st.write("age: ", st.session_state.age)
    if user_gender:
        st.session_state.gender = user_gender
        st.write("gender: ", st.session_state.gender)
    if prompt_template:
        st.session_state.prompt_template = prompt_template
        st.write("prompt_template: ", st.session_state.prompt_template)


def ch4_display():
    # Display session state
    st.write("name: ", st.session_state.name)
    st.write("age: ", st.session_state.age)
    st.write("gender: ", st.session_state.gender)
    st.write("prompt_template: ", st.session_state.prompt_template)


def ex5():
    st.title("My first chatbot")

    if "store_msg" not in st.session_state:
        st.session_state.store_msg = []

    prompt = st.chat_input("Say something")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")
        st.session_state.store_msg.append(prompt)
        for message in st.session_state.store_msg:
            with st.chat_message("user"):
                st.write(message)
            with st.chat_message("assistant"):
                st.write("Hello human, what can I do for you?")


def ex6():
    st.markdown("**Echo Bot**")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def ch6():
    st.markdown("**Challenge 6: Echo Bot**")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if prompt == "Hello":
            response = "Hi there what can I do for you?"
        elif prompt == "What is your name?":
            response = "My name is EAI , an electronic artificial being"
        elif prompt == "How old are you?":
            response = "Today is my birthday!"
        else:
            response = "I am sorry, I am unable to help you with your query"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def ex8():
    st.title("Api Call")
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Tell me about Singapore in the 1970s in 50 words.",
            },
        ],
        temperature=0,
    )
    st.markdown("**This is the raw response:**")
    st.write(response)
    st.markdown("**This is the extracted response:**")
    st.write(response["choices"][0]["message"]["content"].strip())
    s = str(response["usage"]["total_tokens"])
    st.markdown("**Total tokens used:**")
    st.write(s)


def ch8():
    st.title("Api Call")
    MODEL = "gpt-3.5-turbo"

    if query := st.text_input("Enter your query"):
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": query,
                },
            ],
            temperature=0,
        )
        st.markdown("**This is the raw response:**")
        st.write(response)
        st.markdown("**This is the extracted response:**")
        st.write(response["choices"][0]["message"]["content"].strip())
        s = str(response["usage"]["total_tokens"])
        st.markdown("**Total tokens used:**")
        st.write(s)


def chat_completion_stream(prompt, template="You are a helpful assistant"):
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # temperature
        stream=True,  # stream option
    )
    return response


def ex9_basebot():
    # Initialize chat history
    if "chat_msg" not in st.session_state:
        st.session_state.chat_msg = []

    # Showing Chat history
    for message in st.session_state.chat_msg:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    try:
        #
        if prompt := st.chat_input("What is up?"):
            # set user prompt in chat history
            st.session_state.chat_msg.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # streaming function
                for response in chat_completion_stream(prompt):
                    full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            st.session_state.chat_msg.append(
                {"role": "assistant", "content": full_response}
            )
    except Exception as e:
        st.error(e)


def ex10_basebot():
    # prompt_template in session state already set in main()
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": st.session_state.prompt_template},
            {
                "role": "user",
                "content": "Tell me about Singapore in the 1970s in 50 words",
            },
        ],
        temperature=0,
    )
    st.markdown("**LLM Response:**")
    st.write(response["choices"][0]["message"]["content"].strip())
    st.markdown("**Total tokens:**")
    st.write(str(response["usage"]["total_tokens"]))


# WL Self Attempt
# def ch10(prompt_template=None, store_memory=False):
#     if prompt_template is None:
#         if my_prompt_template := st.text_input(
#             "Enter a system prompt template. E.g. Speak like Yoda from Star Wars."
#         ):
#             st.session_state.prompt_template = my_prompt_template
#             st.write("new prompt template set! ", st.session_state.prompt_template)
#     else:
#         my_prompt_template = prompt_template

#     # Initialize chat history
#     if "chat_msg" not in st.session_state:
#         st.session_state.chat_msg = []

#     # Showing Chat history
#     for message in st.session_state.chat_msg:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     if my_query := st.chat_input("What is up?"):
#         # set user prompt in chat history
#         st.session_state.chat_msg.append({"role": "user", "content": my_query})
#         with st.chat_message("user"):
#             st.markdown(my_query)

#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""
#             # streaming function
#             for response in chat_completion_stream(
#                 my_query, template=my_prompt_template
#             ):
#                 full_response += response.choices[0].delta.get("content", "")
#                 message_placeholder.markdown(full_response + "▌")
#             message_placeholder.markdown(full_response)
#         st.session_state.chat_msg.append(
#             {"role": "assistant", "content": full_response}
#         )
#         if store_memory:
#             st.session_state.memory.save_context(
#                 {"input": my_query}, {"output": full_response}
#             )


# Challenge 10
# mod chat complete stream function by replacing system content to session_state prompt template
def chat_completion_stream_prompt(prompt):
    MODEL = "gpt-3.5-turbo"  # consider changing this to session_state
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": st.session_state.prompt_template},
            {"role": "user", "content": prompt},
        ],
        temperature=0,  # temperature
        stream=True,  # stream option
    )
    return response


# Challenge 10: Make the bot speak like someone you know
def ch10():
    # call the function in your base bot
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

    except Exception as e:
        st.error(e)
