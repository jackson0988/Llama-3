import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mount the Amazon EFS file system
efs_mount_point = "/mnt/efs"
os.system(f"mount -t efs -o tls {EFS_FILE_SYSTEM_ID}:/ {efs_mount_point}")

# Define the model path
model_path = os.path.join(efs_mount_point, "cached_model")

# Check if the cached model exists
if os.path.exists(model_path):
    st.write("Loading cached model...")
    # Load the cached model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    st.write("Caching model...")
    # Load the original model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-3b-hf", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-3b-hf")

    # Save the model and tokenizer to the EFS file system
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Streamlit app
st.title("LLaMA 3 Inference")

input_text = st.text_area("Enter text for inference", height=200)

if st.button("Generate"):
    with st.spinner("Generating text..."):
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate text with the model
        output_ids = model.generate(input_ids, max_length=500, num_return_sequences=1, early_stopping=True)

        # Decode the output text
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        st.write(f"Output: {output_text}")

# Unmount the EFS file system when the app is stopped
def on_stop():
    os.system(f"umount {efs_mount_point}")

# Register the on_stop function to be called when the app is stopped
st.on_exit(on_stop)
