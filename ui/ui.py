import streamlit as st
from response_generator.Response_Generator import Response_Generator
import os
from langchain_community.callbacks import StreamlitCallbackHandler

# Streamlit runs this code with every button click, input sent, etc.

def chat():
    st.title("🤖 Agent")
    st.text("Currently only has the RAG tool at it's disposal")

    with st.chat_message("assistant"):
        st.markdown("Hello, how can I help you today?")

    # Check if the generator is already created in the session state, otherwise create it
    # This way we can keep it loaded in memory and dont recreate it on every ui change
    if "generator" not in st.session_state:
        print("Creating generator...")
        st.session_state.generator = Response_Generator()

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Type away..."):
        # Display user message and store it
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call LLM
        with st.chat_message("assistant"):
            
            st_callback = StreamlitCallbackHandler(
                parent_container=st.container(),
                expand_new_thoughts=False
            )
            generator = st.session_state.generator
            response = generator.generate_response(prompt, st_callback)
            st.write(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
            })

chat()