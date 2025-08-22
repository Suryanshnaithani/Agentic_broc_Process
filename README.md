# 🏠 Real Estate Brochure Analyzer & Q&A Assistant

Transform **static real estate brochures (PDFs)** into **interactive, queryable assistants**.  
Our tool extracts key project information, images (floorplans, amenities, masterplans, maps), and enables a **Q&A chatbot** where users can ask natural language questions about the project.

---

## ✨ Features

- 📄 **PDF Upload & Extraction**
  - Extracts structured data (JSON) from brochures using predefined schema
  - Auto-generates project summary: name, builder, RERA, address, floorplans, amenities, etc.

- 🖼️ **Image Extraction**
  - Extracts and organizes brochure visuals:
    - Floor plans
    - Amenities
    - Masterplan
    - Location map
    - Builder logo  

- 🤖 **Q&A Chatbot**
  - Ask questions in natural language
  - Powered by Gemini + FAISS Vector DB for retrieval-augmented responses
  - Answers strictly from extracted project context

- 💻 **Web UI (Gradio)**
  - **Left panel:** Upload brochures → view extracted data & image gallery  
  - **Right panel:** Interactive chatbot for project Q&A  

---

## 🚀 Getting Started

### 1. Clone repo
- git clone https://github.com/Suryanshnaithani/Agentic_broc_Process.git


### 2. Install dependencies
- pip install -r requirements.txt
- Also run once in Python to download nltk tokenizer:
  import nltk
  nltk.download('punkt')


### 3. Set Environment Variables
- Set your **Gemini API Key** and **Vision API Key**
Links to get the keys:
1. https://aistudio.google.com/apikey
2. https://va.landing.ai/settings/personal/api-key

### 4. RUN gradio UI
- python ui.py

---

## 🖥️ Demo Walkthrough

1. Upload a sample brochure PDF
2. Extracted summary & data instantly display (project info, floorplans, amenities)
3. Gallery shows floorplans, masterplans, location maps, etc.
4. Ask chatbot:  
   - *“What are the BHK configurations?”*  
   - *“List all amenities in the project”*  
   - *“Where is the project located?”*  

---

## ⚙️ Tech Stack

- **Backend**
  - [PyMuPDF (fitz)] – PDF parsing
  - [Pillow] – Image processing
  - Custom Schema-driven extractor (`schema.py`)
- **AI**
  - [Agentic Dpcument Extraction] - Agent to extract data from documents created by LandingAI.
  - [Google Gemini API] – Text embeddings & content generation
  - [FAISS] – Vector database for efficient semantic retrieval
- **Frontend**
  - [Gradio] – Minimalist web UI


## Library Support for seamless integration between modules
 - **Clone Library from this link**
 -  https://github.com/Suryanshnaithani/Brochure-analyzer.git
 - **Install it using PIP**
 - **Command** :
   pip install .
