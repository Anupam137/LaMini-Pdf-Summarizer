import streamlit as st 
import base64
import torch
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import pypdf

# Page configuration
st.set_page_config(page_title="PDF Summarizer", layout="wide")

# Model and tokenizer loading
@st.cache_resource
def load_model():
    try:
        checkpoint = "google/flan-t5-base"  # Using a larger model for better summarization
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        base_model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, 
            device_map='auto', 
            torch_dtype=torch.float32
        )
        return tokenizer, base_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Improved PDF text extraction
def extract_pdf_text(file_path):
    try:
        # Use PyPDF2 for more robust text extraction
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Extract text from all pages
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                
                # Clean and preprocess text
                page_text = re.sub(r'\s+', ' ', page_text)  # Remove extra whitespaces
                page_text = page_text.strip()
                
                if page_text:
                    text_content.append(page_text)
        
        # Join all page texts
        full_text = " ".join(text_content)
        
        return full_text if full_text else None
    
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return None

# Improved summarization pipeline
@st.cache_data
def llm_pipeline(filepath, _tokenizer, _base_model):
    try:
        # Extract full text from PDF
        full_text = extract_pdf_text(filepath)
        
        if not full_text:
            return "Unable to extract text from the PDF."
        
        # Determine appropriate chunking based on text length
        max_input_length = 1024  # Adjust based on model's max input length
        
        # If text is too long, split into chunks
        if len(full_text) > max_input_length:
            # Split text into manageable chunks
            chunks = [full_text[i:i+max_input_length] for i in range(0, len(full_text), max_input_length)]
        else:
            chunks = [full_text]
        
        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            # Dynamically calculate max_length based on input length
            input_length = len(chunk)
            dynamic_max_length = max(50, min(input_length // 2, 500))  # Adjust summary length dynamically
            
            pipe_sum = pipeline(
                'summarization',
                model=_base_model,
                tokenizer=_tokenizer,
                max_length=dynamic_max_length,  # Dynamically set max length
                min_length=max(20, dynamic_max_length // 3),  # Proportional min length
                do_sample=True,   # Enable sampling for more varied output
            )
            
            # Add context to generate a more comprehensive summary
            prompt = f"Provide a detailed, structured summary of the following text. Include key points, main ideas, and important details:\n\n{chunk}"
            
            result = pipe_sum(prompt, max_length=dynamic_max_length, min_length=max(20, dynamic_max_length // 3))
            summaries.append(result[0]['summary_text'])
        
        # Combine and refine summaries
        final_summary = " ".join(summaries)
        
        # Post-process summary to add structure
        return format_summary(final_summary)
    
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return f"Summarization failed: {e}"

# Format summary with bullet points and structure
def format_summary(summary):
    # Remove random character sequences and non-meaningful text
    summary = re.sub(r'[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-D]\s*[A-B]\s*[A-D]', '', summary)
    
    # Split summary into sentences with more robust regex
    sentences = re.split(r'(?<=[.!?])\s+', summary)
    
    # Advanced filtering and cleaning
    def is_meaningful_sentence(sentence):
        # Remove very short or non-informative sentences
        if len(sentence.strip()) < 30:
            return False
        
        # Remove sentences with excessive random characters
        if len(re.findall(r'[A-Z]', sentence)) > len(sentence) / 2:
            return False
        
        # Remove purely numeric or symbol-based sentences
        if re.match(r'^[\d\W]+$', sentence):
            return False
        
        return True
    
    # Filter and clean sentences
    meaningful_sentences = [
        sentence.strip() 
        for sentence in sentences 
        if is_meaningful_sentence(sentence)
    ]
    
    # Remove exact duplicates while preserving order
    unique_sentences = []
    for sentence in meaningful_sentences:
        if sentence not in unique_sentences:
            unique_sentences.append(sentence)
    
    # If no meaningful sentences found, return original summary
    if not unique_sentences:
        return summary
    
    # Limit to top 10 sentences to prevent overly long summaries
    unique_sentences = unique_sentences[:10]
    
    # Format as markdown with structured bullet points
    formatted_summary = "## Key Insights\n\n"
    formatted_summary += "\n".join([f"• {sentence}" for sentence in unique_sentences])
    
    return formatted_summary

# Display PDF function
@st.cache_data
def display_pdf(file):
    try:
        with open(file, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        return pdf_display
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        return None

def main():
    # Title and description
    st.title("📄 Advanced PDF Summarization Assistant")
    st.markdown("Upload a PDF and get a comprehensive, structured summary using advanced language models.")

    # Sidebar for additional information
    st.sidebar.header("About")
    st.sidebar.info(
        "This app uses the Flan-T5 model to generate detailed, structured summaries "
        "from uploaded PDF files. Simply upload a PDF and click 'Summarize'."
    )

    # Load model
    tokenizer, base_model = load_model()

    # Check if model loading was successful
    if tokenizer is None or base_model is None:
        st.error("Failed to load the language model. Please check your internet connection.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=['pdf'], 
        help="Upload a PDF file to generate a comprehensive summary"
    )

    if uploaded_file is not None:
        # Save uploaded file
        filepath = f"data/{uploaded_file.name}"
        
        # Ensure data directory exists
        import os
        os.makedirs("data", exist_ok=True)
        
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Create columns for PDF view and summary
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Uploaded PDF")
            pdf_view = display_pdf(filepath)
            if pdf_view:
                st.markdown(pdf_view, unsafe_allow_html=True)

        with col2:
            st.subheader("📝 Comprehensive Summary")
            if st.button("Generate Summary", type="primary"):
                with st.spinner("Generating comprehensive summary..."):
                    summary = llm_pipeline(filepath, tokenizer, base_model)
                    
                    if summary:
                        st.markdown(summary)
                    else:
                        st.error("Failed to generate summary. Please try again.")

if __name__ == "__main__":
    main()
