import os
import json
from typing import Dict, Any
from PIL import Image
import re
import fitz

# Cleaning bounding box responses for mapping
def parse_bbox(bbox_str: str):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", bbox_str)
    if len(nums) != 4:
        raise ValueError(f"Invalid bbox format: {bbox_str}")
    return tuple(map(float, nums))


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


class BrochureProcessor:
    def __init__(self, source_pdf_path: str, extracted_json_data: Dict[str, Any], output_dir: str = "output") -> None:
        self.source_pdf_path = source_pdf_path
        self.data = extracted_json_data
        self.output_dir = output_dir
        self.image_dir = os.path.join(self.output_dir, "images")

        self.subdirs = {
            "floorplan": os.path.join(self.image_dir, "floorplan"),
            "amenities": os.path.join(self.image_dir, "amenities"),
            "masterplan": os.path.join(self.image_dir, "masterplan"),
            "location": os.path.join(self.image_dir, "location"),
            "builder": os.path.join(self.image_dir, "builder"),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def _ensure_dir(self, key: str) -> None:
        os.makedirs(self.subdirs[key], exist_ok=True)

    def _crop_bbox(self, image: Image.Image, bbox_str: str) -> Image.Image:
        try:
            width, height = image.size
            left, top, right, bottom = parse_bbox(bbox_str)
            return image.crop((left * width, top * height, right * width, bottom * height))
        except Exception as e:
            raise ValueError(f"Invalid bounding box format: '{bbox_str}' - {e}")

    def _is_hd(self, image: Image.Image, min_width: int = 1280, min_height: int = 720) -> bool:
        width, height = image.size
        return width >= min_width and height >= min_height

    def save_cleaned_json(self, filename: str = "extracted_data.json") -> None:
        cleaned_data = dict(self.data)
        cleaned_data.pop("amenitiesImages", None)
        cleaned_data.pop("masterplanImage", None)
        cleaned_data.pop("locationMapImage", None)

        for config in cleaned_data.get("floorplanConfigs", []):
            for key in ("imageId", "boundingBoxLTRB", "pageNumber"):
                config.pop(key, None)

        for key in ("imageId", "boundingBoxLTRB", "pageNumber"):
            cleaned_data.get("builder", {}).pop(key, None)
            
        json_path = os.path.join(self.output_dir, filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Cleaned JSON saved to: {json_path}")

    def _extract_floorplans(self, pages: Dict[int, Image.Image]) -> None:
        for config in self.data.get("floorplanConfigs", []):
            bbox = config.get("boundingBoxLTRB")
            page_idx = config.get("pageNumber")

            if not bbox or page_idx is None or str(page_idx).strip() == "":
                print(f"[WARNING] Skipping floorplan due to invalid bbox/page: bbox='{bbox}' page_idx='{page_idx}'")
                continue

            try:
                page_image = pages[int(page_idx)]
                cropped = self._crop_bbox(page_image, bbox)
                bhk = config.get("bhkType", "Unit").replace(" ", "").replace("+", "_")
                filename = sanitize_filename(f"{bhk}.jpg")
                self._ensure_dir("floorplan")
                save_path = os.path.join(self.subdirs["floorplan"], filename)

                counter = 1
                while os.path.exists(save_path):
                    filename = sanitize_filename(f"{bhk}_{counter}.jpg")
                    save_path = os.path.join(self.subdirs["floorplan"], filename)
                    counter += 1

                cropped.save(save_path)
                print(f"[INFO] Floorplan saved: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to extract floorplan: {e}")

    def _extract_amenities(self, pages: Dict[int, Image.Image]) -> None:
        amenities = self.data.get("amenitiesImages", [])
        for amenity in amenities:
            bbox = amenity.get("boundingBoxLTRB")
            page_idx = amenity.get("pageNumber")
            label = amenity.get("amenityLabel", "Amenity")

            if not bbox or page_idx is None or str(page_idx).strip() == "":
                print(f"[WARNING] Skipping amenity due to invalid bbox/page: {label}")
                continue

            try:
                cropped = self._crop_bbox(pages[int(page_idx)], bbox)
                filename = sanitize_filename(label.replace("&", "and") + ".jpg")

                if self._is_hd(cropped):
                    self._ensure_dir("amenities")
                    cropped.save(os.path.join(self.subdirs["amenities"], filename))
                    print(f"[INFO] Amenity saved: {filename}")
                else:
                    print(f"[INFO] Skipped non-HD amenity: {filename}")
            except Exception as e:
                print(f"[ERROR] Failed to extract amenity '{label}': {e}")

    def _extract_masterplan(self, pages: Dict[int, Image.Image]) -> None:
        master = self.data.get("masterplanImage")
        if not master:
            return

        bbox = master.get("boundingBoxLTRB")
        page_idx = master.get("pageNumber")
        if not bbox or page_idx is None or str(page_idx).strip() == "":
            print("[WARNING] Skipping masterplan due to invalid bbox/page")
            return

        try:
            cropped = self._crop_bbox(pages[int(page_idx)], bbox)
            self._ensure_dir("masterplan")
            cropped.save(os.path.join(self.subdirs["masterplan"], "masterplan.jpg"))
            print("[INFO] Masterplan image saved.")
        except Exception as e:
            print(f"[ERROR] Failed to extract masterplan: {e}")

    def _extract_location_map(self, pages: Dict[int, Image.Image]) -> None:
        location = self.data.get("locationMapImage")
        if not location:
            return

        bbox = location.get("boundingBoxLTRB")
        page_idx = location.get("pageNumber")
        if not bbox or page_idx is None or str(page_idx).strip() == "":
            print("[WARNING] Skipping location map due to invalid bbox/page")
            return

        try:
            cropped = self._crop_bbox(pages[int(page_idx)], bbox)
            self._ensure_dir("location")
            cropped.save(os.path.join(self.subdirs["location"], "location_map.jpg"))
            print("[INFO] Location map image saved.")
        except Exception as e:
            print(f"[ERROR] Failed to extract location map: {e}")

    def extract_builder_logo(self, pages: Dict[int, Image.Image]) -> None:
        builder = self.data.get("builder", {})
        bbox = builder.get("boundingBoxLTRB")
        page_idx = builder.get("pageNumber")
        if not bbox or page_idx is None or str(page_idx).strip() == "":
            print("[WARNING] Skipping builder logo due to invalid bbox/page")
            return

        try:
            cropped = self._crop_bbox(pages[int(page_idx)], bbox)
            self._ensure_dir("builder")
            cropped.save(os.path.join(self.subdirs["builder"], "logo.jpg"))
            print("[INFO] Builder logo image saved.")
        except Exception as e:
            print(f"[ERROR] Failed to extract builder logo: {e}")

    def _load_pages_with_fitz(self, required_pages):
        page_images = {}
        with fitz.open(self.source_pdf_path) as pdf:
            for page_num in required_pages:
                page = pdf[page_num]
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_images[page_num] = img
        return page_images

    def extract_images(self, pages: Dict[int, Image.Image]) -> None:
        self._extract_floorplans(pages)
        self._extract_amenities(pages)
        self._extract_masterplan(pages)
        self._extract_location_map(pages)

    def process_all(self) -> None:
        print(f"[START] Processing brochure: {os.path.basename(self.source_pdf_path)}")
        try:
            required_pages = set()
            for cfg in self.data.get("floorplanConfigs", []):
                page = cfg.get("pageNumber")
                if page is not None and str(page).strip().isdigit():
                    required_pages.add(int(page))

            for section in ["amenitiesImages", "masterplanImage", "locationMapImage", "builder"]:
                value = self.data.get(section)
                if isinstance(value, dict):
                    page = value.get("pageNumber")
                    if page is not None and str(page).strip().isdigit():
                        required_pages.add(int(page))
                elif isinstance(value, list):
                    for item in value:
                        page = item.get("pageNumber")
                        if page is not None and str(page).strip().isdigit():
                            required_pages.add(int(page))

            required_pages = sorted(required_pages)
            page_images = self._load_pages_with_fitz(required_pages)

            self.extract_images(page_images)
            self.extract_builder_logo(page_images)
            self.save_cleaned_json()

            print(f"[DONE] Completed processing: {os.path.basename(self.source_pdf_path)}")
        except Exception as e:
            print(f"[FATAL] Brochure processing failed: {e}")
