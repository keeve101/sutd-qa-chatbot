from config import SERVER_URL

import streamlit as st
import requests

server_url = SERVER_URL


def get_llm_response(query, model_choice, use_rag):
    model_map = {
        "Base": "llm_base",
        "Finetuned": "llm_finetune",
    }

    payload = {
        "query": query,
        "model": model_map[model_choice],
        "use_rag": use_rag,
    }

    try:
        with requests.post(server_url, json=payload, stream=True) as resp:
            if resp.status_code == 200:
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode("utf-8").removeprefix("data: ")
                        yield decoded
            else:
                yield f"Error: {resp.text}"
    except Exception as e:
        yield f"Exception occurred: {str(e)}"


st.title("SUTD Chatbot")

if "model_choice" not in st.session_state:
    st.session_state["model_choice"] = "Base"
if "use_rag" not in st.session_state:
    st.session_state["use_rag"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "pending_user_input" not in st.session_state:
    st.session_state["pending_user_input"] = ""

with st.sidebar:
    model_choice = st.selectbox(
        "Select Model",
        ["Base", "Finetuned"],
        index=["Base", "Finetuned"].index(st.session_state["model_choice"]),
    )
    use_rag = st.checkbox("Use RAG", value=st.session_state["use_rag"])

    if st.button("Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()

if (
    model_choice != st.session_state["model_choice"]
    or use_rag != st.session_state["use_rag"]
):
    st.session_state["chat_history"] = []
    st.session_state["model_choice"] = model_choice
    st.session_state["use_rag"] = use_rag
    st.rerun()

for entry in st.session_state["chat_history"]:
    st.text(f"{entry['role'].capitalize()}: {entry['message']}")

response_box = st.empty()


def handle_submit():
    user_message = st.session_state["pending_user_input"]
    if user_message:
        st.session_state["chat_history"].append(
            {"role": "user", "message": user_message}
        )

        st.session_state["pending_user_input"] = ""

        full_response = ""
        for token in get_llm_response(
            user_message, st.session_state["model_choice"], st.session_state["use_rag"]
        ):
            full_response += token
            response_box.text(f"Assistant: {full_response}")

        st.session_state["chat_history"].append(
            {"role": "assistant", "message": full_response}
        )


st.text_input("Ask something...", key="pending_user_input", on_change=handle_submit)
