# importing libraries
import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(page_title="CohereLLM", page_icon=":rocket:", layout="wide")

# Initialize the chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.markdown("<h1 style='text-align:center'>Llama 3 LLM</h1>", unsafe_allow_html=True)
st.write("<h3 style='text-align:center'>This app uses a pre-trained Llama model to generate text.</h3>", unsafe_allow_html=True)


user_input = st.text_input("Enter your message:", key="user_input", placeholder="Type your message here...")
if user_input:
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    response = model.generate(input_ids, max_length=2500, temperature=0.3, num_return_sequences=1)
    response_text = tokenizer.decode(response[0], skip_special_tokens=True)
    st.write(response_text)
