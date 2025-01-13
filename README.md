# 📄 LaMini PDF Summarizer

## 🚀 Advanced PDF Summarization Tool

### 🌟 Key Features
- **Intelligent PDF Text Extraction**: Robust text extraction from various PDF formats
- **Advanced Summarization**: Uses Flan-T5 language model for comprehensive summaries
- **Smart Summary Formatting**: 
  * Removes random characters and noise
  * Filters out non-meaningful sentences
  * Generates clean, structured bullet-point summaries
- **Dynamic Length Adaptation**: Adjusts summary length based on input document

### 🔧 Technical Highlights
- Utilizes `transformers` library for summarization
- Implements advanced text preprocessing
- Handles PDFs of varying lengths and complexities
- Provides markdown-formatted summary output

### 🛠 Requirements
- Python 3.8+
- Transformers
- Streamlit
- PyPDF2
- Torch

### 🚦 Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/LaMini-Pdf-Summarizer.git
```

2. Create a virtual environment
```bash
python -m venv pdf_env
source pdf_env/bin/activate  # On Windows use `pdf_env\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

### 🏃 Running the Application
```bash
streamlit run stream.py
```

### 📝 Usage
1. Upload a PDF file
2. Click "Generate Summary"
3. View comprehensive, structured summary

### 🔍 How It Works
- Extracts text from uploaded PDF
- Chunks long documents for processing
- Applies advanced NLP techniques to generate summaries
- Cleans and structures summary output

### 🌈 Recent Improvements
- Enhanced text extraction algorithm
- Improved summary formatting
- Better handling of complex PDF documents
- Reduced noise in summary generation

### 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

### 💡 Powered By
- Streamlit
- Hugging Face Transformers
- Flan-T5 Language Model
