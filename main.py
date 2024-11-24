import os
import shutil
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import pandas as pd
import google.generativeai as genai
import streamlit as st
import zipfile
import tempfile


sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

api_key = "removed for security"
GOOGLE_API_KEY = api_key  
genai.configure(api_key=GOOGLE_API_KEY)


generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}
google_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = " ".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def extract_text_from_xlsx(file_path):
    df = pd.read_excel(file_path, sheet_name=0)
    text = " ".join(df.astype(str).values.flatten())
    return text

def image_directly_to_folder(file_path, image_folder="organized_documents/images"):
    os.makedirs(image_folder, exist_ok=True)
    shutil.move(file_path, os.path.join(image_folder, os.path.basename(file_path)))


def classify_topic(text, topics):
    """Classifies the topic of a document using Google Generative AI."""
    try:
        chat_session = google_model.start_chat(history=[])
        category_prompt = f"""
        Classify the following text into one of these categories: {', '.join(topics)}.
        Text: {text}
        Please provide only the category label.
        """
        response = chat_session.send_message(category_prompt)
        predicted_category = response.text.strip()
        return predicted_category if predicted_category in topics else "Unclassified"
    except Exception as e:
        st.error(f"Error classifying text: {e}")
        return "Error in classification"

def organize_files(file_paths, topics, base_folder="organized_documents"):
    os.makedirs(base_folder, exist_ok=True)
    for file_path, topic in zip(file_paths, topics):
        topic_folder = os.path.join(base_folder, topic)
        os.makedirs(topic_folder, exist_ok=True)
        shutil.move(file_path, os.path.join(topic_folder, os.path.basename(file_path)))


def process_and_organize_files(uploaded_files, user_defined_topics):
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    topics = []
    non_image_files = []
    for file_path in file_paths:
        if file_path.endswith((".png", ".jpg", ".jpeg")):
            image_directly_to_folder(file_path)
        else:
            if file_path.endswith(".pdf"):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                text = extract_text_from_docx(file_path)
            elif file_path.endswith(".xlsx"):
                text = extract_text_from_xlsx(file_path)

            topic = classify_topic(text, user_defined_topics)
            topics.append(topic)
            non_image_files.append(file_path)
    
    organize_files(non_image_files, topics)

    zip_file_path = tempfile.mktemp(suffix=".zip")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("organized_documents"):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), "organized_documents"))

    return zip_file_path

# Streamlit app layout
st.title("File Organizer")


custom_topics = st.text_area("Enter topic names (comma-separated)", " ", height=150)
user_defined_topics = [topic.strip() for topic in custom_topics.split(",") if topic.strip()]

# File uploader widget in Streamlit
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)


if st.button("Start Organizing"):
    if uploaded_files and user_defined_topics:
        with st.spinner("Organizing files..."):
            zip_file_path = process_and_organize_files(uploaded_files, user_defined_topics)
            with open(zip_file_path, "rb") as f:
                st.download_button(
                    label="Download Organized Files",
                    data=f,
                    file_name="organized_documents.zip",
                    mime="application/zip"
                )
    else:
        st.warning("Please upload files and enter at least one topic.")
