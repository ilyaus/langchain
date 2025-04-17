import os

import streamlit as st

from backend.core import run_llm


if ("user_prompt_history" not in st.session_state and 
    "chat_answer_history" not in st.session_state and 
    "chat_history" not in st.session_state):
    
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_answer_history"] = []
    st.session_state["chat_history"] = []

def format_sources(sources: list) -> str:
    if not sources:
        return ""
    
    source_list = list(sources)
    source_list.sort()

    source_string = "Sources:\n"

    for i, source in enumerate(source_list):
        source_string += f"{i + 1}. {source}\n"

    return source_string


st.header("LangChain Doc Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here.")

if prompt:
    with st.spinner("Generating Response ..."):
        response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        sources = set([doc.metadata["source"] for doc in response["source_documents"]])

        formatted_response = f"{response["result"]}\n\n{format_sources(sources)}"
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", response["result"]))

        prompt = None


if st.session_state["chat_answer_history"]:
    for user_query, generated_response in zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"]):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)