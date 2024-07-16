from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
genai.configure(api_key=api_key)

def get_gemini_response(input_prompt, image_data, input_text):
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        
        def stream_callback(response):
            return response.text

        response = model.generate_content(
            contents=[input_prompt, image_data[0], input_text],
            stream=True
        )

        full_response = ""
        for chunk in response:
            chunk_text = stream_callback(chunk)
            full_response += chunk_text if chunk_text else ""

        return full_response if full_response else "No response generated."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Streamlit app
st.set_page_config(page_title="Invoice Extractor")
st.header("Multi Language Invoice Extractor With Gemini")

input_prompt = """
You are an expert in understanding invoices.
You will receive input images as invoices &
you will have to answer questions based on the input image
"""

input_text = st.text_input("Input Prompt: ", key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

submit = st.button("Tell me about the image")

if submit:
    if uploaded_file is None:
        st.error("Please upload an image first.")
    else:
        with st.spinner("Analyzing the image..."):
            image_data = input_image_setup(uploaded_file)
            response_text = get_gemini_response(input_prompt, image_data, input_text)
            
            st.subheader("The Response is")
            st.write(response_text)