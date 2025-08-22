import os
import json
import glob
import logging
from typing import Dict, Any
from agentic_doc.parse import parse
from elements_breakdown import BrochureProcessor
from schema import schema
import sys

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_DIR = os.path.join("Responses")
DATA_DIR = os.path.join("Data")
JSON_DIR = os.path.join("JSON_DIR")


def ensure_directories() -> None:
    """Ensures that the main output directories exist."""
    os.makedirs(RESPONSES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)


def _find_latest_json_for_project(project_name: str) -> str:
    pattern = os.path.join(JSON_DIR, f"{project_name}_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No JSON files found matching pattern: {pattern}")
    return max(files, key=os.path.getmtime)


def process_brochure_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    Extract data and process a single brochure PDF.

    Returns:
        dict: {
            "status": 200,
            "project_name": str,
            "json_file": str,
            "project_data_dir": str,
            "response_path": str
        }

    Raises:
        Exception on failure.
    """
    ensure_directories()

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    project_name = os.path.splitext(os.path.basename(pdf_path))[0]
    response_path = os.path.join(RESPONSES_DIR, f"{project_name}.md")

    if os.path.exists(response_path):
        logger.info(f"[SKIP] Already processed: {project_name}")
        try:
            json_file = _find_latest_json_for_project(project_name)
        except FileNotFoundError:
            json_file = None
        return {
            "status": 200,
            "project_name": project_name,
            "json_file": json_file,
            "project_data_dir": None,
            "response_path": response_path
        }

    logger.info(f"[START] Processing brochure: {project_name}")

    # Run parsing (saves JSON to JSON_DIR)
    try:
        parse(pdf_path, extraction_schema=schema, result_save_dir=JSON_DIR)
    except Exception as e:
        raise RuntimeError(f"Parsing PDF failed: {e}")

    # Find the matching JSON (starts with project_name + "_")
    json_file = _find_latest_json_for_project(project_name)

    # Read JSON
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file {json_file}: {e}")

    # Save markdown if present
    markdown_content = data.get("markdown")
    if markdown_content:
        with open(response_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_content)
        logger.info(f"[INFO] Markdown saved to {response_path}")
    else:
        logger.warning(f"[WARNING] No 'markdown' key found in {json_file}")
        

    # Get extraction data
    extracted_data = data.get("extraction")
    if not extracted_data:
        raise RuntimeError(f"No 'extraction' key found in {json_file}")

    # Prepare output directory
    project_data_dir = os.path.join(DATA_DIR, project_name)
    os.makedirs(project_data_dir, exist_ok=True)

    # Process brochure assets
    try:
        processor = BrochureProcessor(
            source_pdf_path=pdf_path,
            extracted_json_data=extracted_data,
            output_dir=project_data_dir
        )
        processor.process_all()
    except Exception as e:
        logger.exception("BrochureProcessor failed")
        raise RuntimeError(f"BrochureProcessor failed: {e}")

    logger.info(f"[DONE] Finished processing: {project_name}")
    return {
        "status": 200,
        "project_name": project_name,
        "json_file": json_file,
        "project_data_dir": project_data_dir,
        "response_path": response_path,
    }


if __name__ == "__main__":

    pdf_path = "Brochure/r413082.pdf"
    try:
        result = process_brochure_pdf(pdf_path)
        print(result)
    except Exception as e:
        logger.exception("Processing failed")
        print(f"[ERROR] {e}")
