import gradio as gr
import os
import json
import tempfile
import shutil
import pandas as pd
import traceback
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any
from elements_breakdown import BrochureProcessor
from main import process_brochure_pdf
from genai import MarkdownVectorDB

# Globals
genai_db = None
RESPONSES_DIR = 'Responses'
DATA_DIR = 'Data'
current_project_data = None

def format_extracted_data(data: Dict[str, Any]) -> str:
    """Format extracted brochure data into a concise, business-friendly summary"""
    if not data:
        return "No data extracted"

    lines = [
        f"Project: {data.get('projectName', 'N/A')}",
        f"Builder: {data.get('builder', {}).get('name', 'N/A')}",
        f"Location: {data.get('projectAddress', {}).get('Locality', 'N/A')}, "
        f"{data.get('projectAddress', {}).get('City', 'N/A')}",
        f"RERA: {data.get('rera', 'N/A')}",
        "",
        "Floor Plans:",
    ]

    # Floor plans
    floorplans = data.get('floorplanConfigs', [])
    if floorplans:
        for fp in floorplans:
            bhk = fp.get('bhkType', 'N/A')
            carpet = fp.get('carpetArea', 'N/A')
            total = fp.get('totalArea', 'N/A')
            lines.append(f"• {bhk} | Carpet: {carpet} | Total: {total}")
    else:
        lines.append("• No floor plans found")

    # Amenities
    amenities = data.get('amenities', [])
    if amenities and amenities != ["Not Present"] and amenities[0] != "Not Present":
        valid_amenities = [a for a in amenities if a and a != "Not Present"]
        if valid_amenities:
            lines.append("")
            lines.append(f"Amenities ({len(valid_amenities)}): {', '.join(valid_amenities[:7])}")
            if len(valid_amenities) > 7:
                lines.append(f"... and {len(valid_amenities) - 7} more")

    # Location highlights
    locations = data.get('location_highlights', [])
    if locations and locations != ["Not Present"]:
        valid_locations = [l for l in locations if l != "Not Present" and isinstance(l, dict)]
        if valid_locations:
            lines.append("")
            location_names = [l.get('location_name', '') for l in valid_locations[:4] if l.get('location_name')]
            if location_names:
                lines.append(f"Nearby: {', '.join(location_names)}")
                if len(valid_locations) > 4:
                    lines.append(f"... and {len(valid_locations) - 4} more locations")

    return "\n".join(lines)

def load_images_from_directory(images_dir: str) -> List[str]:
    """Load image file paths from a directory for gallery display."""
    image_paths = []
    if os.path.exists(images_dir):
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def process_brochure_handler(pdf_file):
    """Handle brochure PDF upload and processing with comprehensive error handling"""
    global genai_db, current_project_data
    
    print("\n" + "="*50)
    print("PROCESSING BROCHURE")
    print("="*50)
    
    if pdf_file is None:
        return "❌ Please upload a PDF file first.", [], None

    try:
        # Store temp copy of uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copy2(pdf_file.name, tmp_file.name)
            temp_pdf_path = tmp_file.name

        print(f"📄 Processing PDF: {pdf_file.name}")
        print(f"📁 Temp path: {temp_pdf_path}")
        
        # Process brochure
        result = process_brochure_pdf(temp_pdf_path)
        
        print(f"📊 Processing result status: {result.get('status', 'Unknown')}")
        
        if result.get('status') != 200:
            print(f"❌ Processing failed with status: {result.get('status')}")
            return "❌ Processing failed. Please check the PDF and try again.", [], None

        # Load extracted data
        json_file = result.get('json_file')
        if not json_file or not os.path.exists(json_file):
            print(f"❌ JSON file not found: {json_file}")
            return "❌ Failed to extract data from brochure.", [], None
            
        print(f"📋 Loading data from: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
        
        extraction_data = full_data.get('extraction', {})
        if not extraction_data:
            print("❌ No extraction data found in JSON")
            return "❌ No data could be extracted from the brochure.", [], None
            
        current_project_data = extraction_data
        project_name = extraction_data.get('projectName', 'Unknown')
        print(f"✅ Extracted project: {project_name}")

        # Format summary for UI
        summary = format_extracted_data(extraction_data)
        print(f"📝 Generated summary ({len(summary)} chars)")

        # Load images for gallery
        images = []
        project_data_dir = result.get('project_data_dir')
        if project_data_dir and os.path.exists(project_data_dir):
            images_dir = os.path.join(project_data_dir, 'images')
            images = load_images_from_directory(images_dir)
            print(f"🖼️ Found {len(images)} images")
        else:
            print("⚠️ No project data directory found")

        # Initialize Vector DB with comprehensive error handling
        genai_db = None  # Reset first
        qa_status = "❌ Q&A Not Available"
        
        md_file = result.get('response_path')
        print(f"📄 Checking markdown file: {md_file}")
        
        if md_file and os.path.exists(md_file):
            try:
                # Check file size
                file_size = os.path.getsize(md_file)
                print(f"📄 Markdown file size: {file_size} bytes")
                
                if file_size < 50:
                    print("⚠️ Markdown file too small")
                    qa_status = "⚠️ Q&A Unavailable (Insufficient content)"
                else:
                    # Check API key
                    api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyBKMTlb3t0Yg6j85ynT3TEsz_ZQhV1zlO4')
                    
                    if not api_key or api_key == 'DUMMY_KEY':
                        print("⚠️ No valid API key found")
                        qa_status = "⚠️ Q&A Unavailable (No API key)"
                    else:
                        print(f"🔑 Using API key: {api_key[:20]}...")
                        
                        # Check markdown content
                        with open(md_file, 'r', encoding='utf-8') as f:
                            md_content = f.read().strip()
                        
                        print(f"📄 Markdown content preview: {md_content[:200]}...")
                        
                        if len(md_content) < 100:
                            print("⚠️ Markdown content too short")
                            qa_status = "⚠️ Q&A Unavailable (Content too short)"
                        else:
                            # Initialize GenAI DB
                            print("🤖 Initializing GenAI DB...")
                            
                            genai_db = MarkdownVectorDB(
                                api_key=api_key,
                                markdown_path=md_file
                            )
                            
                            print("✅ GenAI DB initialized successfully")
                            
                            # Test with a simple query
                            try:
                                print("🧪 Testing GenAI DB...")
                                test_response = genai_db.query("What is this document about?")
                                print(f"✅ Test query successful: {len(str(test_response))} chars")
                                qa_status = "✅ Q&A Ready"
                            except Exception as test_e:
                                print(f"❌ Test query failed: {test_e}")
                                genai_db = None
                                qa_status = f"❌ Q&A Error: {str(test_e)}"
                        
            except Exception as e:
                print(f"❌ GenAI DB initialization failed: {e}")
                print(f"Error type: {type(e).__name__}")
                traceback.print_exc()
                genai_db = None
                qa_status = f"❌ Q&A Error: {str(e)}"
        else:
            print(f"❌ Markdown file not found: {md_file}")
            qa_status = "❌ Q&A Unavailable (No markdown file)"

        # Clean up temp file
        if os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
            print("🗑️ Cleaned up temp file")
        
        # Final summary with status
        final_summary = f"{summary}\n\n" + "="*40 + f"\n{qa_status}"
        
        print("✅ Processing completed successfully")
        print("="*50)
        
        return final_summary, images, extraction_data

    except Exception as e:
        print(f"❌ Critical processing error: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        return f"❌ Critical error: {str(e)}", [], None

def ask_question_handler(history: List, question: str, extraction_data: Dict[str, Any]) -> List:
    """Handle user questions with comprehensive error handling and debugging"""
    global genai_db
    
    print(f"\n{'='*30}")
    print(f"Q&A REQUEST")
    print(f"{'='*30}")
    print(f"Question: '{question}'")
    print(f"History length: {len(history) if history else 0}")
    print(f"Extraction data exists: {extraction_data is not None}")
    print(f"GenAI DB exists: {genai_db is not None}")
    print(f"GenAI DB type: {type(genai_db).__name__ if genai_db else 'None'}")
    
    # Ensure history is a list
    if history is None:
        history = []
    
    # Check empty question
    if not question or not question.strip():
        print("⚠️ Empty question")
        return history + [("user", ""), ("assistant", "Please type a question.")]
    
    question = question.strip()
    
    # Check if brochure data exists
    if extraction_data is None:
        print("❌ No extraction data")
        return history + [("user", question), ("assistant", "❌ Please upload and process a brochure first.")]
    
    # Check if GenAI DB is available
    if genai_db is None:
        print("❌ GenAI DB not available")
        return history + [("user", question), ("assistant", "❌ Q&A system is not available. Please try re-uploading the brochure or check your API key.")]
    
    try:
        print(f"🤖 Querying GenAI DB...")
        answer = genai_db.query(question)
        
        # Ensure answer is a string
        answer_str = str(answer) if answer else "I couldn't find an answer to your question."
        
        print(f"✅ Got answer ({len(answer_str)} chars): {answer_str[:100]}{'...' if len(answer_str) > 100 else ''}")
        
        new_history = history + [("user", question), ("assistant", answer_str)]
        print(f"✅ Updated history length: {len(new_history)}")
        
        return new_history
        
    except AttributeError as e:
        print(f"❌ AttributeError: {e}")
        error_msg = "❌ There's an issue with the Q&A system. Please try re-uploading the brochure."
        return history + [("user", question), ("assistant", error_msg)]
    
    except Exception as e:
        print(f"❌ Q&A Error: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
        return history + [("user", question), ("assistant", error_msg)]

def clear_all():
    """Clear all data and reset the interface"""
    global genai_db, current_project_data
    
    print("\n🗑️ Clearing all data...")
    genai_db = None
    current_project_data = None
    
    return "🔄 Ready for new brochure...", [], None, []

def debug_status():
    """Debug function to check system status"""
    global genai_db, current_project_data
    
    status_lines = [
        "=== SYSTEM DEBUG STATUS ===",
        f"Current Time: {pd.Timestamp.now() if 'pd' in globals() else 'N/A'}",
        f"GenAI DB Status: {'✅ Available' if genai_db else '❌ Not Available'}",
        f"Project Data: {'✅ Available' if current_project_data else '❌ Not Available'}",
        f"API Key Set: {'✅ Yes' if os.getenv('GEMINI_API_KEY') else '❌ No'}",
        ""
    ]
    
    if genai_db:
        status_lines.extend([
            f"GenAI DB Type: {type(genai_db).__name__}",
            f"Markdown Path: {getattr(genai_db, 'markdown_path', 'Unknown')}",
            f"Chunks Count: {len(getattr(genai_db, 'chunks', []))}",
        ])
    
    if current_project_data:
        status_lines.extend([
            "",
            "=== PROJECT DATA ===",
            f"Project Name: {current_project_data.get('projectName', 'Unknown')}",
            f"Builder: {current_project_data.get('builder', {}).get('name', 'Unknown')}",
            f"Floor Plans: {len(current_project_data.get('floorplanConfigs', []))}",
            f"Amenities: {len([a for a in current_project_data.get('amenities', []) if a != 'Not Present'])}",
        ])
    
    status_text = "\n".join(status_lines)
    print(status_text)
    return status_text

# Enhanced CSS for better visual appeal
css = """
.gradio-container {
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 1300px;
}
.main-heading {
    font-size: 28px;
    font-weight: 600;
    text-align: center;
    margin-bottom: 10px;
    color: #2c3e50;
}
.divider {
    border-bottom: 2px solid #3498db;
    margin-bottom: 25px;
}
.section-title {
    font-size: 18px;
    font-weight: 500;
    margin: 18px 0 10px;
    color: #34495e;
}
.summary-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    border-left: 4px solid #3498db;
    font-family: 'Consolas', 'Monaco', monospace;
    line-height: 1.6;
    font-size: 13px;
}
.gallery {
    background: #fafbfc;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e1e5e9;
}
.chatbox {
    background: #fff;
    border-radius: 8px;
    border: 1px solid #e1e5e8;
    font-size: 14px;
}
.status-good {
    color: #27ae60;
    font-weight: 500;
}
.status-error {
    color: #e74c3c;
    font-weight: 500;
}
"""

with gr.Blocks(css=css, title="Brochure Analyzer") as demo:

    # Header
    gr.Markdown("## 🏠 Real Estate Brochure Analyzer", elem_classes=["main-heading"])
    gr.HTML("<div class='divider'></div>")

    with gr.Row():
        # LEFT PANEL: Upload & Data Display
        with gr.Column(scale=5):
            gr.Markdown("### 📄 Upload Brochure")
            
            upload_pdf = gr.File(
                label="Select PDF Brochure", 
                file_count="single",
                file_types=['.pdf']
            )
            
            with gr.Row():
                upload_btn = gr.Button("🔄 Process Brochure", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                debug_btn = gr.Button("🔍 Debug", variant="secondary", scale=1)
            
            # Status and summary display
            status_text = gr.Textbox(
                label="Project Summary & Status",
                value="🔄 Ready to process brochure...\n\nUpload a PDF file and click 'Process Brochure' to begin.",
                interactive=False,
                elem_classes=["summary-box"],
                lines=12,
                show_copy_button=True
            )
            
            # Images gallery
            gr.Markdown("### 🖼️ Extracted Images")
            images_gallery = gr.Gallery(
                label=None,
                columns=2,
                rows=2,
                object_fit="contain",
                height=280,
                elem_classes=["gallery"]
            )

        # RIGHT PANEL: Q&A Chat
        with gr.Column(scale=7):
            gr.Markdown("### 🤖 Project Q&A Assistant")
            
            # Chat interface
            chatbot = gr.Chatbot(
                label=None,
                elem_classes=["chatbox"],
                bubble_full_width=False,
                height=450,
                show_copy_button=True
            )
            
            # Question input and controls
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question about the project",
                    placeholder="e.g., What are the BHK types? What amenities are available? Where is it located?",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1):
                    q_btn = gr.Button("Send", variant="primary")
                    clear_chat_btn = gr.Button("Clear Chat", variant="secondary", size="sm")
            
            # Sample questions for quick access
            gr.Markdown("#### 💡 Quick Questions:")
            with gr.Row():
                with gr.Column():
                    sample_q1 = gr.Button("BHK Types", size="sm")
                    sample_q2 = gr.Button("Amenities", size="sm")
                with gr.Column():
                    sample_q3 = gr.Button("Location", size="sm")
                    sample_q4 = gr.Button("Builder Info", size="sm")

    # State management
    data_state = gr.State(value=None)
    chat_state = gr.State(value=[])

    # Event handlers
    upload_btn.click(
        fn=process_brochure_handler,
        inputs=[upload_pdf],
        outputs=[status_text, images_gallery, data_state],
        show_progress=True
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[status_text, images_gallery, data_state, chat_state]
    )
    
    debug_btn.click(
        fn=debug_status,
        inputs=[],
        outputs=[status_text]
    )

    # Q&A handlers
    q_btn.click(
        fn=ask_question_handler,
        inputs=[chat_state, question_input, data_state],
        outputs=[chat_state],
        show_progress=True
    ).then(
        lambda: "",  # Clear input after sending
        outputs=[question_input]
    )

    question_input.submit(
        fn=ask_question_handler,
        inputs=[chat_state, question_input, data_state],
        outputs=[chat_state],
        show_progress=True
    ).then(
        lambda: "",  # Clear input after sending
        outputs=[question_input]
    )
    
    # Clear chat only
    clear_chat_btn.click(
        lambda: [],
        outputs=[chat_state]
    )

    # Sample question handlers
    sample_q1.click(lambda: "What are the different BHK configurations available?", outputs=[question_input])
    sample_q2.click(lambda: "List all the amenities available in this project.", outputs=[question_input])
    sample_q3.click(lambda: "Where is this project located? What are nearby landmarks?", outputs=[question_input])
    sample_q4.click(lambda: "Who is the builder and what are the project details?", outputs=[question_input])

    # Update chat display
    chat_state.change(
        lambda x: x,
        inputs=[chat_state],
        outputs=[chatbot]
    )

# Launch configuration
if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("🚀 Starting Brochure Analyzer...")
    print(f"📁 Responses Directory: {RESPONSES_DIR}")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"🔑 API Key Set: {'Yes' if os.getenv('GEMINI_API_KEY') else 'No'}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True
    )
