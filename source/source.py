# source/source.py
# OCR and PDF text extraction utilities with language parameter and basic post-processing.

import cv2
import numpy as np
import pytesseract  # Optical Character Recognition
import os         # Path checking
from pathlib import Path
import re
import logging      # Logging errors
import time # For progress updates

try:
    import fitz # PyMuPDF for PDF handling
except ImportError:
    logging.error("PyMuPDF (fitz) not installed. PDF functionality will be disabled. Run: pip install PyMuPDF")
    fitz = None

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
MIN_TEXT_HEIGHT_FOR_OCR = 10
TARGET_TEXT_HEIGHT_FOR_OCR = 35
MAX_UPSCALE_FACTOR = 3.0
ROI_PADDING_OCR = 10
PDF_RENDER_DPI = 300 # DPI for PDF rendering

# --- Image Preprocessing Helpers ---
def upscale_roi(image, target_height, current_height, max_factor=MAX_UPSCALE_FACTOR):
    """Upscales an image ROI to a target height."""
    if current_height <= 0 or target_height <= current_height: return image
    scale_factor = min(max_factor, target_height / current_height)
    if scale_factor <= 1.01: return image
    new_w = int(image.shape[1] * scale_factor)
    new_h = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

def sharpen_image(image):
    """Sharpens the image using a kernel."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

# --- Binarization ---
def sauvola_binarization(gray_image, window_size=21, k=0.2, R=128):
    """Applies Sauvola local thresholding (text white)."""
    if gray_image is None: logging.error("Sauvola input image is None."); return None
    if window_size % 2 == 0: window_size += 1
    pad = window_size // 2
    if gray_image.dtype != np.uint8:
        if np.max(gray_image) <= 1.0: gray_image = (gray_image * 255).astype(np.uint8)
        else: gray_image = gray_image.astype(np.uint8)
    padded_image = cv2.copyMakeBorder(gray_image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    local_mean = cv2.boxFilter(padded_image, cv2.CV_64F, (window_size, window_size))[pad:-pad, pad:-pad]
    padded_image_sq = padded_image.astype(np.float64)**2
    local_mean_sq = cv2.boxFilter(padded_image_sq, cv2.CV_64F, (window_size, window_size))[pad:-pad, pad:-pad]
    local_std_dev = np.sqrt(np.maximum(0, local_mean_sq - local_mean**2))
    threshold_sauvola = local_mean * (1 + k * ((local_std_dev / R) - 1))
    binarized_image_text_white = np.zeros_like(gray_image, dtype=np.uint8)
    binarized_image_text_white[gray_image <= threshold_sauvola] = 255
    return binarized_image_text_white

# --- Text Block Detection ---
def detect_text_blocks(binarized_image_text_white):
    """Detects text blocks using morphological operations."""
    if binarized_image_text_white is None: logging.error("Block detection input image is None."); return []
    img_h, img_w = binarized_image_text_white.shape[:2]
    strategy_profiles = [
        {"name": "Default", "dilate_w_factor": 70, "dilate_h_factor": 300, "dilate_min_w": 15, "dilate_min_h": 3, "dilate_iter": 1, "close_w_factor": 50, "close_h_factor": 80, "close_min_w": 15, "close_min_h": 10, "close_iter": 1, "min_w": 20, "min_h": 10, "min_area": 200, "density_thresh": 0.01, "logo_filter_max_density": 0.15},
        {"name": "AggressiveHorizontal", "dilate_w_factor": 40, "dilate_h_factor": 400, "dilate_min_w": 25, "dilate_min_h": 2, "dilate_iter": 1, "close_w_factor": 30, "close_h_factor": 60, "close_min_w": 30, "close_min_h": 15, "close_iter": 2, "min_w": 15, "min_h": 8, "min_area": 150, "density_thresh": 0.008, "logo_filter_max_density": 0.10},
        {"name": "SmallerComponents", "dilate_w_factor": 100, "dilate_h_factor": 500, "dilate_min_w": 10, "dilate_min_h": 1, "dilate_iter": 1, "close_w_factor": 80, "close_h_factor": 100, "close_min_w": 10, "close_min_h": 5, "close_iter": 1, "min_w": 10, "min_h": 5, "min_area": 100, "density_thresh": 0.015, "logo_filter_max_density": 0.20},
        {"name": "StrongerBlockClosing", "dilate_w_factor": 90, "dilate_h_factor": 300, "dilate_min_w": 10, "dilate_min_h": 3, "dilate_iter": 1, "close_w_factor": 40, "close_h_factor": 30, "close_min_w": 20, "close_min_h": 20, "close_iter": 2, "min_w": 20, "min_h": 10, "min_area": 200, "density_thresh": 0.01, "logo_filter_max_density": 0.15},
    ]
    final_text_blocks_rois = []
    for idx, profile in enumerate(strategy_profiles):
        dilate_w = max(profile["dilate_min_w"], img_w // profile["dilate_w_factor"]); dilate_w += (1 - dilate_w % 2)
        dilate_h = max(profile["dilate_min_h"], img_h // profile["dilate_h_factor"]); dilate_h += (1 - dilate_h % 2)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_w, dilate_h))
        dilated_image = cv2.dilate(binarized_image_text_white, dilate_kernel, iterations=profile["dilate_iter"])
        close_w = max(profile["close_min_w"], img_w // profile["close_w_factor"]); close_w += (1 - close_w % 2)
        close_h = max(profile["close_min_h"], img_h // profile["close_h_factor"]); close_h += (1 - close_h % 2)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
        current_morph_closed = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, close_kernel, iterations=profile["close_iter"])
        contours, _ = cv2.findContours(current_morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_text_blocks_rois = []
        min_w_block_abs, min_h_block_abs, min_area_contour_abs = profile["min_w"], profile["min_h"], profile["min_area"]
        density_thresh, max_area_contour, logo_filter_max_density = profile["density_thresh"], img_w * img_h * 0.95, profile["logo_filter_max_density"]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour);
            if h == 0: continue
            aspect_ratio, contour_area = w / float(h), cv2.contourArea(contour)
            is_potential_header_or_logo_geom = (y < img_h * 0.15 and h < img_h * 0.10 and w > img_w * 0.15 and aspect_ratio > 2.0 and contour_area < (img_w * img_h * 0.10))
            if is_potential_header_or_logo_geom:
                roi_logo_candidate_bin = binarized_image_text_white[y:y+h, x:x+w]
                if roi_logo_candidate_bin.size > 0:
                    density_logo_candidate = np.sum(roi_logo_candidate_bin > 0) / roi_logo_candidate_bin.size
                    if density_logo_candidate < logo_filter_max_density: continue
            if (contour_area > min_area_contour_abs and contour_area < max_area_contour and w > min_w_block_abs and h > min_h_block_abs and 0.05 < aspect_ratio < 75):
                roi_bin = binarized_image_text_white[y:y+h, x:x+w]
                if roi_bin.size > 0:
                    density = np.sum(roi_bin > 0) / roi_bin.size
                    if density > density_thresh: current_text_blocks_rois.append((x, y, w, h))
        if current_text_blocks_rois:
            final_text_blocks_rois = current_text_blocks_rois; break
    if not final_text_blocks_rois: logging.warning("No text blocks found after trying all strategies.")
    return final_text_blocks_rois

# --- Line Segmentation within Blocks ---
def segment_lines_in_block(block_roi_binarized_text_white, block_coords):
    """Segments text lines within a binarized block ROI."""
    x_block, y_block, w_block, h_block = block_coords
    if block_roi_binarized_text_white is None or block_roi_binarized_text_white.size == 0: return []
    img_h_block, img_w_block = block_roi_binarized_text_white.shape[:2]
    if img_h_block < 5 or img_w_block < 10:
        if np.sum(block_roi_binarized_text_white > 0) / block_roi_binarized_text_white.size > 0.01: return [(x_block, y_block, w_block, h_block)]
        return []
    horizontal_projection = np.sum(block_roi_binarized_text_white, axis=1) / 255.0
    min_pixels_for_line_calc = max(1, img_w_block * 0.005)
    meaningful_projections = horizontal_projection[horizontal_projection > min_pixels_for_line_calc]
    if len(meaningful_projections) < 3:
        if np.sum(block_roi_binarized_text_white > 0) / block_roi_binarized_text_white.size > 0.01: return [(x_block, y_block, w_block, h_block)]
        return []
    line_threshold = np.mean(meaningful_projections) * 0.25
    line_threshold = max(line_threshold, 1.0)
    in_line, line_start_y_local_coord = False, 0
    line_rois_local_y_coords = []
    min_line_h_pixel = max(3, img_h_block // 50)
    for i, projection_val in enumerate(horizontal_projection):
        if not in_line and projection_val > line_threshold: in_line, line_start_y_local_coord = True, i
        elif in_line and (projection_val <= line_threshold or i == len(horizontal_projection) - 1):
            in_line, line_end_y_local_coord = False, i
            if i == len(horizontal_projection) - 1 and projection_val > line_threshold: line_end_y_local_coord = i + 1
            current_line_h_local = line_end_y_local_coord - line_start_y_local_coord
            if current_line_h_local >= min_line_h_pixel: line_rois_local_y_coords.append((line_start_y_local_coord, line_end_y_local_coord))
    line_rois_global = []
    for y_start_local, y_end_local in line_rois_local_y_coords:
        current_line_h_local = y_end_local - y_start_local
        if current_line_h_local <= 0: continue
        line_strip_bwhite = block_roi_binarized_text_white[y_start_local:y_end_local, :]
        min_text_pixels_in_line_strip = max(1, current_line_h_local * img_w_block * 0.002)
        if line_strip_bwhite.size == 0 or np.sum(line_strip_bwhite > 0) < min_text_pixels_in_line_strip: continue
        vertical_projection_check = np.sum(line_strip_bwhite, axis=0) / 255.0
        min_col_height_for_text = max(1, 0.03 * current_line_h_local)
        if not np.any(vertical_projection_check > min_col_height_for_text): continue
        ocr_line_x_local, ocr_line_w_local = 0, img_w_block
        gx, gy, gh = x_block + ocr_line_x_local, y_block + y_start_local, current_line_h_local
        actual_content_ys = np.where(np.sum(line_strip_bwhite, axis=1) > 0)[0]
        if len(actual_content_ys) > 0:
            tight_y_start_local, tight_y_end_local = actual_content_ys[0], actual_content_ys[-1] + 1
            if (tight_y_end_local - tight_y_start_local) >= min_line_h_pixel // 2:
                gy = y_block + y_start_local + tight_y_start_local
                gh = tight_y_end_local - tight_y_start_local
        line_rois_global.append((gx, gy, ocr_line_w_local, gh))
    return line_rois_global

# --- OCR Integration (with lang parameter) ---
def ocr_on_rois(
    original_gray_image_source, rois, lang="eng", psm_mode="7",
    use_custom_binarization_for_ocr=False, sauvola_full_page_text_white=None,
    apply_opening_on_binary=False, opening_kernel_width=3, enable_upscaling=True,
    denoise_gray_roi=True, apply_clahe_gray_roi=True, sharpen_gray_roi=True, target_dpi=300,
):
    """Performs OCR on ROIs with language parameter."""
    all_text = []
    config_params = [f"--oem 3", f"--psm {psm_mode}", f"--dpi {target_dpi}", r"-c preserve_interword_spaces=1"]
    custom_config = " ".join(config_params)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    for i, (x, y, w, h) in enumerate(rois):
        if w <= 3 or h <= 3: continue
        y_end, x_end = min(y + h, original_gray_image_source.shape[0]), min(x + w, original_gray_image_source.shape[1])
        abs_y_start, abs_x_start = max(0, y), max(0, x)
        if abs_y_start >= y_end or abs_x_start >= x_end: continue
        gray_roi = original_gray_image_source[abs_y_start:y_end, abs_x_start:x_end]
        if gray_roi.size == 0: continue
        ocr_ready_roi, current_h, current_w = None, gray_roi.shape[0], gray_roi.shape[1]
        if use_custom_binarization_for_ocr:
            binary_roi_for_ocr = None
            if sauvola_full_page_text_white is not None:
                s_y_end, s_x_end = min(y+h, sauvola_full_page_text_white.shape[0]), min(x+w, sauvola_full_page_text_white.shape[1])
                if abs_y_start < s_y_end and abs_x_start < s_x_end:
                    bin_roi_text_white = sauvola_full_page_text_white[abs_y_start:s_y_end, abs_x_start:s_x_end]
                    if bin_roi_text_white.size > 0: binary_roi_for_ocr = cv2.bitwise_not(bin_roi_text_white)
            if binary_roi_for_ocr is None: _, binary_roi_for_ocr = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.OTSU)
            if binary_roi_for_ocr is not None:
                current_h, current_w = binary_roi_for_ocr.shape[:2]
                if apply_opening_on_binary and opening_kernel_width > 0 and current_w > opening_kernel_width:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_width, 1))
                    binary_roi_for_ocr = cv2.morphologyEx(binary_roi_for_ocr, cv2.MORPH_OPEN, kernel)
                if enable_upscaling and current_h < TARGET_TEXT_HEIGHT_FOR_OCR and current_h > 0:
                    binary_roi_for_ocr = upscale_roi(binary_roi_for_ocr, TARGET_TEXT_HEIGHT_FOR_OCR, current_h)
                ocr_ready_roi = binary_roi_for_ocr
            else: ocr_ready_roi = gray_roi
        else:
            processed_gray_roi = gray_roi.copy()
            if denoise_gray_roi: processed_gray_roi = cv2.fastNlMeansDenoising(processed_gray_roi, None, h=10, templateWindowSize=7, searchWindowSize=21)
            if apply_clahe_gray_roi: processed_gray_roi = clahe.apply(processed_gray_roi)
            if sharpen_gray_roi: processed_gray_roi = sharpen_image(processed_gray_roi)
            current_h_gray, _ = processed_gray_roi.shape[:2]
            if enable_upscaling and current_h_gray < TARGET_TEXT_HEIGHT_FOR_OCR and current_h_gray > 0:
                processed_gray_roi = upscale_roi(processed_gray_roi, TARGET_TEXT_HEIGHT_FOR_OCR, current_h_gray)
            ocr_ready_roi = processed_gray_roi
        if ocr_ready_roi is not None:
            padding_color = 255 if len(ocr_ready_roi.shape) < 3 or ocr_ready_roi.shape[2] == 1 else (255, 255, 255)
            ocr_ready_roi_padded = cv2.copyMakeBorder(ocr_ready_roi, ROI_PADDING_OCR, ROI_PADDING_OCR, ROI_PADDING_OCR, ROI_PADDING_OCR, cv2.BORDER_CONSTANT, value=padding_color)
            try:
                text = pytesseract.image_to_string(ocr_ready_roi_padded, lang=lang, config=custom_config)
                extracted_text = text.strip()
                if extracted_text: all_text.append(extracted_text)
            except pytesseract.TesseractNotFoundError: logging.error("Tesseract not found."); return "[Tesseract Not Found Error]"
            except Exception as e:
                logging.error(f"OCR Error on ROI {i} (lang={lang}, x{x} y{y} w{w} h{h}): {e}", exc_info=True)
                logging.error(f"  ROI sent to Tesseract had shape: {ocr_ready_roi_padded.shape if ocr_ready_roi_padded is not None else 'None'}")
                all_text.append(f"[OCR Error ROI {i}]")
    return "\n".join(filter(None, all_text))

# --- Basic Post-OCR Text Processing ---
def basic_post_process_text(raw_text):
    """Cleans up OCR text output."""
    if not raw_text or not raw_text.strip():
        return ""
    text = raw_text
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    text = "\n".join(stripped_lines)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = text.strip()
    return text

# --- Core Image Data Processing Function ---
def process_image_data(image_data, lang="eng", source_name="image"):
    """Processes image data (NumPy array) to extract text."""
    logging.info(f"Starting text extraction from {source_name} (Lang: {lang})")
    try:
        if image_data is None: return None, f"Error: Input image data is None for {source_name}."
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        elif len(image_data.shape) == 2:
            gray = image_data
        else:
             return None, f"Error: Unsupported image format (shape: {image_data.shape}) for {source_name}."

        logging.info(f"Image data loaded ({gray.shape[1]}x{gray.shape[0]}). Applying preprocessing...")

        blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sauvola_text_white = sauvola_binarization(blurred_gray, window_size=21, k=0.2)
        if sauvola_text_white is None:
            logging.warning(f"{source_name}: Sauvola failed. Falling back to Otsu.")
            _, otsu_binary_inv = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            sauvola_text_white = cv2.bitwise_not(otsu_binary_inv)
            if sauvola_text_white is None: return None, f"Error: Binarization failed (Sauvola and Otsu) for {source_name}."

        processed_gray = gray.copy()
        processed_gray = cv2.fastNlMeansDenoising(processed_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
        processed_gray = clahe.apply(processed_gray)
        processed_gray = sharpen_image(processed_gray)

        padding_color = 255
        processed_gray_padded = cv2.copyMakeBorder(processed_gray, ROI_PADDING_OCR, ROI_PADDING_OCR, ROI_PADDING_OCR, ROI_PADDING_OCR, cv2.BORDER_CONSTANT, value=padding_color)

        config_params = [f"--oem 3", f"--psm 6", f"--dpi 300", r"-c preserve_interword_spaces=1"]
        custom_config = " ".join(config_params)
        
        try:
            text = pytesseract.image_to_string(processed_gray_padded, lang=lang, config=custom_config)
            processed_text = basic_post_process_text(text)

            if not processed_text.strip():
                logging.warning(f"{source_name}: OCR process completed, but no text was extracted or survived basic cleanup.")
                return "", None

            logging.info(f"Text extraction from {source_name} (Lang: {lang}) and simplified cleanup successful.")
            return processed_text, None

        except pytesseract.TesseractNotFoundError:
            err_msg = "Tesseract Error: Executable not found. Ensure Tesseract is installed and in your system's PATH."
            logging.error(err_msg)
            return None, err_msg
        except pytesseract.TesseractError as e:
            err_msg = f"Tesseract Processing Error for {source_name}: {e}. Ensure language data ('{lang}.traineddata') is installed correctly."
            logging.error(err_msg)
            return None, err_msg

    except Exception as e:
        err_msg = f"An unexpected error occurred during processing {source_name}: {e}"
        logging.error(err_msg, exc_info=True)
        return None, err_msg

# --- Main Function for Image Files ---
def process_image_extract_text(image_path, lang="eng"):
    """Loads an image file and extracts text."""
    logging.info(f"Loading image file: {image_path}")
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            return None, f"Error: Could not load image file '{Path(image_path).name}'."
        return process_image_data(original_image, lang=lang, source_name=f"image '{Path(image_path).name}'")
    except Exception as e:
        err_msg = f"An unexpected error occurred loading or starting processing for image {image_path}: {e}"
        logging.error(err_msg, exc_info=True)
        return None, err_msg

# --- PDF Handling Functions ---
def render_pdf_page_to_image_data(pdf_doc, page_index):
    """Renders a single PDF page to an OpenCV (NumPy BGR) image."""
    if not fitz: return None, "PyMuPDF (fitz) is not installed."
    if not pdf_doc or page_index < 0 or page_index >= pdf_doc.page_count:
        return None, "Invalid PDF document or page index."
    try:
        page = pdf_doc.load_page(page_index)
        pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
        if pix.alpha:
            image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 4)
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        else:
             image_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
             rgb_image = image_np
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return bgr_image, None
    except Exception as e:
        err_msg = f"Error rendering PDF page {page_index + 1}: {e}"
        logging.error(err_msg, exc_info=True)
        return None, err_msg

def process_pdf_page_extract_text(pdf_doc, page_index, lang="eng"):
    """Renders a single PDF page and extracts text."""
    logging.info(f"Processing PDF page {page_index + 1} (Lang: {lang})")
    page_image_data, error = render_pdf_page_to_image_data(pdf_doc, page_index)
    if error:
        return None, error
    if page_image_data is None:
         return None, f"Failed to render PDF page {page_index + 1} to image data."
    return process_image_data(page_image_data, lang=lang, source_name=f"PDF page {page_index + 1}")

def process_entire_pdf_extract_text(pdf_path, lang="eng", progress_callback=None):
    """Processes all pages of a PDF file and extracts text."""
    if not fitz: return None, "PyMuPDF (fitz) is not installed. Cannot process PDF."
    logging.info(f"Starting full PDF processing for: {pdf_path} (Lang: {lang})")
    all_pages_text = []
    pdf_doc = None
    try:
        pdf_doc = fitz.open(pdf_path)
        num_pages = pdf_doc.page_count
        if num_pages == 0: return "", None
        for i in range(num_pages):
            page_num_human = i + 1
            if progress_callback:
                progress_callback(f"Processing PDF Page {page_num_human}/{num_pages}...")
            page_text, error = process_pdf_page_extract_text(pdf_doc, i, lang=lang)
            if error:
                 logging.error(f"Error processing page {page_num_human} in {pdf_path}: {error}")
                 all_pages_text.append(f"\n\n--- Error on Page {page_num_human}: {error} ---\n\n")
            elif page_text:
                 separator = f"\n\n--- Page {page_num_human} ---\n\n" if i > 0 else f"--- Page {page_num_human} ---\n\n"
                 all_pages_text.append(separator + page_text)
            else:
                 separator = f"\n\n--- Page {page_num_human} (No text found) ---\n\n" if i > 0 else f"--- Page {page_num_human} (No text found) ---\n\n"
                 all_pages_text.append(separator)
        full_text = "".join(all_pages_text).strip()
        logging.info(f"Finished full PDF processing for: {pdf_path}")
        if progress_callback: progress_callback(f"Finished processing {num_pages} PDF pages.")
        return full_text, None
    except fitz.fitz.FileNotFoundError:
         err_msg = f"PDF Error: File not found at '{pdf_path}'."
         logging.error(err_msg)
         if progress_callback: progress_callback(f"Error: PDF not found.")
         return None, err_msg
    except fitz.fitz.FileDataError:
         err_msg = f"PDF Error: Cannot open or read '{Path(pdf_path).name}'. File may be corrupted or password-protected."
         logging.error(err_msg)
         if progress_callback: progress_callback(f"Error: Could not read PDF.")
         return None, err_msg
    except Exception as e:
        err_msg = f"An unexpected error occurred during full PDF processing for {pdf_path}: {e}"
        logging.error(err_msg, exc_info=True)
        if progress_callback: progress_callback(f"Error: Unexpected processing error.")
        return None, err_msg
    finally:
        if pdf_doc:
            try:
                pdf_doc.close()
                logging.info(f"Closed PDF document: {pdf_path}")
            except Exception as e_close:
                logging.warning(f"Error closing PDF document {pdf_path}: {e_close}")

# --- Example Usage ---
if __name__ == "__main__":
    print("Testing source.py functions (including PDF, Language Param, Simplified Post-Processing)...")
    Path("inputs").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    test_image_path = None
    possible_image_names = ["test_image.png", "input_file_0.png", "input_file_1.png", "image_e2b132.jpg"]
    for name in possible_image_names:
        if Path("inputs", name).exists(): test_image_path = str(Path("inputs", name)); break
        elif Path(name).exists(): test_image_path = name; break

    if test_image_path:
        print(f"\n--- Testing Image Extraction (Lang: eng) ---")
        processed_text, error = process_image_extract_text(test_image_path, lang="eng")
        if error: print(f"Error during image extraction: {error}")
        elif processed_text is not None:
            print("Processed Image Text:\n---------------------------------\n" + (processed_text if processed_text else "[No Text Found]") + "\n---------------------------------")
            out_file = Path("outputs") / f"test_output_{Path(test_image_path).stem}_img.txt"
            try:
                with open(out_file, "w", encoding="utf-8") as f: f.write(processed_text if processed_text else "[No Text Found]")
                print(f"Saved processed image text to {out_file}")
            except Exception as e: print(f"Error saving image output file: {e}")
        else: print("Image text extraction returned None without an error message.")
    else: print(f"\nNo test image found (looked for {', '.join(possible_image_names)}).")

    if fitz:
        test_pdf_path = None
        possible_pdf_names = ["test_document.pdf", "input_doc.pdf"]
        for name in possible_pdf_names:
            if Path("inputs", name).exists(): test_pdf_path = str(Path("inputs", name)); break
            elif Path(name).exists(): test_pdf_path = name; break

        if test_pdf_path:
            print(f"\n--- Testing Full PDF Extraction (Lang: eng) ---")
            def cli_progress(msg): print(f"  Progress: {msg}")
            processed_text, error = process_entire_pdf_extract_text(test_pdf_path, lang="eng", progress_callback=cli_progress)
            if error: print(f"Error during PDF extraction: {error}")
            elif processed_text is not None:
                print("Processed PDF Text (All Pages):\n---------------------------------\n" + (processed_text if processed_text else "[No Text Found]") + "\n---------------------------------")
                out_file = Path("outputs") / f"test_output_{Path(test_pdf_path).stem}_pdf_full.txt"
                try:
                    with open(out_file, "w", encoding="utf-8") as f: f.write(processed_text if processed_text else "[No Text Found]")
                    print(f"Saved processed PDF text to {out_file}")
                except Exception as e: print(f"Error saving PDF output file: {e}")
            else: print("PDF text extraction returned None without an error message.")
        else: print(f"\nNo test PDF found (looked for {', '.join(possible_pdf_names)}).")
    else:
        print("\nSkipping PDF tests because PyMuPDF (fitz) is not installed.")
