import sys
import os
import json
import pytesseract
from pathlib import Path
import getpass
import numpy
import cv2

from PyQt6.QtWidgets import (
       QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
       QPushButton, QTextEdit, QLabel, QFileDialog, QFrame, QComboBox,
       QMessageBox, QSizePolicy, QSpacerItem, QToolBar, QMenu,
       QListWidget, QListWidgetItem, QToolButton, QWidgetAction,
       QFontDialog,
       QGraphicsDropShadowEffect
   )
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QObject, QSize, QStandardPaths, QUrl, QMimeData,
    QTimer, QPoint,
    QPropertyAnimation, QEasingCurve
)
from PyQt6.QtGui import (
    QFont, QPalette, QColor, QPixmap, QIcon,
    QTextEdit, QClipboard, QCursor, QDragEnterEvent,
    QDragLeaveEvent, QDropEvent, QImage
)

try:
    import fitz
except ImportError:
    QMessageBox.critical(None, "Dependency Missing", "PyMuPDF (fitz) is required for PDF processing but not installed.\nPlease run: pip install PyMuPDF")
    fitz = None

try:
    from source.source import (
        process_image_extract_text,
        process_pdf_page_extract_text,
        process_entire_pdf_extract_text,
        render_pdf_page_to_image_data
    )
except ImportError as e:
    print(f"Error importing functions from 'source.source': {e}")
    QMessageBox.critical(None, "Import Error", f"Could not import processing functions from source.source:\n{e}\n\nPlease ensure source/source.py exists and is valid.")
    sys.exit(1)
except Exception as e_gen:
    print(f"An unexpected error occurred during source import: {e_gen}")
    QMessageBox.critical(None, "Import Error", f"An unexpected error occurred importing source:\n{e_gen}")
    sys.exit(1)


class Worker(QObject):
    finished = pyqtSignal(object, object)
    progress = pyqtSignal(str)
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function = function; self.args = args; self.kwargs = kwargs
        if 'progress_callback' in self.function.__code__.co_varnames:
            self.kwargs['progress_callback'] = self.progress.emit
        self._is_running = False
        self._thread = None

    def run(self):
        try:
            self._is_running = True
            self.progress.emit("Processing started...")
            result, error = self.function(*self.args, **self.kwargs)
            if not self.kwargs.get('progress_callback'):
                 self.progress.emit("Processing finished.")
            self.finished.emit(result, error)
        except Exception as e:
            self.progress.emit(f"Processing failed: {e}")
            self.finished.emit(None, f"Worker thread error: {e}")
        finally:
            self._is_running = False

    def set_thread(self, thread):
        self._thread = thread

    def cleanup(self):
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
        self._thread = None

# (HoverScaleToolButton class)
class HoverScaleToolButton(QToolButton):
    def __init__(self, icon_path, base_size=QSize(22, 22), scale_factor=1.15, parent=None):
        super().__init__(parent)
        self.icon_path = icon_path
        self.base_size = base_size
        self.scaled_size = QSize(int(base_size.width() * scale_factor), int(base_size.height() * scale_factor))
        self.current_icon_obj = QIcon(self.icon_path)
        self.setIcon(self.current_icon_obj)
        self.setIconSize(self.base_size)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.setText("")
    def set_current_icon(self, new_icon_path):
        self.icon_path = new_icon_path
        self.current_icon_obj = QIcon(self.icon_path)
        self.setIcon(self.current_icon_obj)
        # Keep base size reference, but update display size based on state
        is_scaled = self.iconSize() == self.scaled_size
        self.setIconSize(self.scaled_size if is_scaled else self.base_size)
        self.update() # Force redraw
    def enterEvent(self, event): self.setIconSize(self.scaled_size); super().enterEvent(event)
    def leaveEvent(self, event): self.setIconSize(self.base_size); super().leaveEvent(event)

# (ImageDropLabel class)
class ImageDropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True); self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("ImageDropLabel"); self.normal_style = ""; self.drag_over_style = ""
    def setStyles(self, normal_style, drag_over_style):
        self.normal_style = normal_style; self.drag_over_style = drag_over_style
        self.setStyleSheet(self.normal_style)
    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls() and len(mime_data.urls()) == 1:
            url = mime_data.urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile().lower()
                # Accept images OR PDFs (if fitz is available)
                accepted_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
                if fitz: # Only accept PDF if library is loaded
                    accepted_extensions.append('.pdf')
                if any(file_path.endswith(ext) for ext in accepted_extensions):
                    event.acceptProposedAction(); self.setStyleSheet(self.drag_over_style); return
        event.ignore()
    def dragLeaveEvent(self, event: QDragLeaveEvent):
        self.setStyleSheet(self.normal_style); super().dragLeaveEvent(event)
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet(self.normal_style)
        # Pass the event up to the main window to handle the file path
        if hasattr(self.window(), 'handle_dropped_file'):
            self.window().handle_dropped_file(event.mimeData())
        else: event.ignore() # Don't accept if main window can't handle it


class MainWindow(QMainWindow):
    # (THEMES dictionary)
    THEMES = {
    "Claude Dark": {
        "primary": "#D8B589",  
        "secondary": "#E6CFAF",
        "background1": "#2A2622",
        "background2": "#35302B",
        "background3": "#423A33",
        "text1": "#EDE4D9",   
        "text2": "#C6B7A9",
        "text3": "#9E9083",
        "placeholder_text": "#756A60",
        "border": "#5F554C",
        "shadow": "#181411",  
        "accent_bg": "#D8B589",
        "accent_fg": "#2C2824",
        "button_default_bg": "#423A33",
        "button_hover": "#524A42",
        "button_pressed": "#625A51",
        "list_item_hover": "#4A423B",
        "history_disabled_text": "#9E9083",
        "drop_zone_bg": "#3A322D",
        "drop_zone_border_drag": "#D8B589",
        "splitter_handle": "#35302B"
    },
    "Claude Light": { 
        "primary": "#B08D65", 
        "secondary": "#C8A888",
        "background1": "#FEFBF3", 
        "background2": "#FAF3E9",
        "background3": "#F3EAE0",
        "text1": "#4A3A2B",    
        "text2": "#705E4A",
        "text3": "#8A7A6A",
        "placeholder_text": "#A89A8A",
        "border": "#DACEBE",
        "shadow": "#E8E0D4",
        "accent_bg": "#B08D65",
        "accent_fg": "#FFFFFF",
        "button_default_bg": "#F3EAE0",
        "button_hover": "#EADFCE",
        "button_pressed": "#DBCFB8",
        "list_item_hover": "#F0E8DC",
        "history_disabled_text": "#8A7A6A",
        "drop_zone_bg": "#FAF3E9",
        "drop_zone_border_drag": "#B08D65",
        "splitter_handle": "#FAF3E9"
    },
    "Gemini Dark": {
        "primary": "#8AB4F8",
        "secondary": "#AECBFA",
        "background1": "#1E1E1E",
        "background2": "#282828",
        "background3": "#333333",
        "text1": "#E8EAED",
        "text2": "#BDC1C6",
        "text3": "#9AA0A6",
        "placeholder_text": "#7F848A",
        "border": "#5A90C8",
        "shadow": "#111111",
        "accent_bg": "#8AB4F8",
        "accent_fg": "#1F1F1F",
        "button_default_bg": "#333333",
        "button_hover": "#3C4043",
        "button_pressed": "#4A4A4A",
        "list_item_hover": "#303134",
        "history_disabled_text": "#9AA0A6",
        "drop_zone_bg": "#2D2D2D",
        "drop_zone_border_drag": "#8AB4F8",
        "splitter_handle": "#282828"
    },
    "Gemini Light": {
        "primary": "#1A73E8",
        "secondary": "#4285F4",
        "background1": "#FFFFFF",
        "background2": "#F8F9FA",
        "background3": "#F1F3F4",
        "text1": "#202124",
        "text2": "#5F6368",
        "text3": "#80868B",
        "placeholder_text": "#A0A0A0",
        "border": "#B3D8FF",
        "shadow": "#E0E0E0",
        "accent_bg": "#1A73E8",
        "accent_fg": "#FFFFFF",
        "button_default_bg": "#F1F3F4",
        "button_hover": "#E8F0FE",
        "button_pressed": "#D2E3FC",
        "list_item_hover": "#E8EAED",
        "history_disabled_text": "#80868B",
        "drop_zone_bg": "#F1F3F4",
        "drop_zone_border_drag": "#1A73E8",
        "splitter_handle": "#F8F9FA"
    },
    "Obsidian Dark": { 
        "primary": "#D1D5DB",  
        "secondary": "#9CA3AF",
        "background1": "#111827", 
        "background2": "#1F2937", 
        "background3": "#374151",
        "text1": "#F3F4F6",    
        "text2": "#D1D5DB",   
        "text3": "#9CA3AF",   
        "placeholder_text": "#6B7280",
        "border": "#4B5563",   
        "shadow": "#0B0F1A",    
        "accent_bg": "#9CA3AF", 
        "accent_fg": "#111827",
        "button_default_bg": "#374151",
        "button_hover": "#4B5563",
        "button_pressed": "#5E6A7E",
        "list_item_hover": "#2c3647",
        "history_disabled_text": "#9CA3AF",
        "drop_zone_bg": "#1F2937",
        "drop_zone_border_drag": "#D1D5DB",
        "splitter_handle": "#1F2937"
    },
    "Porcelain Light": {
        "primary": "#6B7280", 
        "secondary": "#9CA3AF", 
        "background1": "#FFFFFF", 
        "background2": "#F9FAFB", 
        "background3": "#F3F4F6", 
        "text1": "#111827",    
        "text2": "#374151",   
        "text3": "#6B7280",    
        "placeholder_text": "#A1A1AA", 
        "border": "#E5E7EB",    
        "shadow": "#F0F0F0",    
        "accent_bg": "#6B7280",
        "accent_fg": "#FFFFFF",
        "button_default_bg": "#F3F4F6",
        "button_hover": "#E5E7EB",
        "button_pressed": "#D1D5DB",
        "list_item_hover": "#F0F2F5",
        "history_disabled_text": "#6B7280",
        "drop_zone_bg": "#F9FAFB",
        "drop_zone_border_drag": "#6B7280",
        "splitter_handle": "#F9FAFB"
    },

    # --- Completely New Themes ---
    "Sakura Dream": { # Light, pinkish, spring-inspired
        "primary": "#FFB7C5", # Light Pink (Sakura petal)
        "secondary": "#FFC0CB", # Pink
        "background1": "#FFF8FA", # Very light, almost white with a hint of pink
        "background2": "#FCEFF2", # Light pinkish off-white
        "background3": "#F8E7EB", # Slightly more saturated pinkish off-white
        "text1": "#6D4B52",    # Dark rosy brown
        "text2": "#8B6C73",    # Medium rosy brown
        "text3": "#A98F97",    # Light rosy brown
        "placeholder_text": "#CBB9BE",
        "border": "#F2DCE2",
        "shadow": "#F0E0E5",
        "accent_bg": "#FFB7C5",
        "accent_fg": "#603E45", # Darker for contrast on light pink
        "button_default_bg": "#F8E7EB",
        "button_hover": "#F0DDE1",
        "button_pressed": "#E8D3D7",
        "list_item_hover": "#F5E0E6",
        "history_disabled_text": "#A98F97",
        "drop_zone_bg": "#FCEFF2",
        "drop_zone_border_drag": "#FFB7C5",
        "splitter_handle": "#FCEFF2"
    },
    "Nordic Noir": { # Cool, desaturated blues and grays, minimalist
        "primary": "#52799E", # Desaturated Blue (like a foggy fjord)
        "secondary": "#6497B1", # Lighter desaturated blue
        "background1": "#1A202C", # Very Dark Slate Blue (almost black)
        "background2": "#202836", # Dark Slate Blue
        "background3": "#2A3444", # Medium-Dark Slate Blue
        "text1": "#E2E8F0",    # Light Bluish Gray (almost white)
        "text2": "#CBD5E0",    # Bluish Gray
        "text3": "#A0AEC0",    # Medium Bluish Gray
        "placeholder_text": "#718096",
        "border": "#4A5568",
        "shadow": "#0D1016",    # Very dark, near black
        "accent_bg": "#52799E",
        "accent_fg": "#EDF2F7", # Very light for contrast
        "button_default_bg": "#2A3444",
        "button_hover": "#354152",
        "button_pressed": "#404C60",
        "list_item_hover": "#2D3748",
        "history_disabled_text": "#A0AEC0",
        "drop_zone_bg": "#202836",
        "drop_zone_border_drag": "#52799E",
        "splitter_handle": "#202836"
    },
    "Oceanic Calm": {
        "primary": "#3DB2C7",  # Clear Teal Blue
        "secondary": "#66CDDD", # Medium Turquoise
        "background1": "#F0F9FA", # Very light cyan, almost white
        "background2": "#E0F2F7", # Light Cyan
        "background3": "#CFEEF7", # Pale Cyan
        "text1": "#104C5F",    # Dark Slate Cyan
        "text2": "#2E6D84",    # Medium Slate Cyan
        "text3": "#578EA1",    # Soft Slate Cyan
        "placeholder_text": "#90B0BB",
        "border": "#ADD8E6",    # Light Blue
        "shadow": "#D0E0E8",
        "accent_bg": "#3DB2C7",
        "accent_fg": "#FFFFFF",
        "button_default_bg": "#CFEEF7",
        "button_hover": "#BDE6F0",
        "button_pressed": "#A8DDE9",
        "list_item_hover": "#D8EFF5",
        "history_disabled_text": "#578EA1",
        "drop_zone_bg": "#E0F2F7",
        "drop_zone_border_drag": "#3DB2C7",
        "splitter_handle": "#E0F2F7"
    },
    "Desert Mirage Dark": {
        "primary": "#E07A5F",  # Burnt Sienna / Terracotta
        "secondary": "#F28C68", # Lighter Terracotta / Coral
        "background1": "#2A211B", # Deepest warm brown (like dark earth)
        "background2": "#3D312A", # Dark warm brown
        "background3": "#4F4239", # Medium-dark warm brown
        "text1": "#FCEBDC",    # Pale sandy color
        "text2": "#E8D5C1",    # Light sand
        "text3": "#C8B8A6",    # Sand
        "placeholder_text": "#A08D7C",
        "border": "#6B5B4F",
        "shadow": "#17120F",    # Very dark brown, almost black
        "accent_bg": "#E07A5F",
        "accent_fg": "#2A211B", # Dark for contrast on accent
        "button_default_bg": "#4F4239",
        "button_hover": "#605249",
        "button_pressed": "#716259",
        "list_item_hover": "#453830",
        "history_disabled_text": "#C8B8A6",
        "drop_zone_bg": "#3D312A",
        "drop_zone_border_drag": "#E07A5F",
        "splitter_handle": "#3D312A"
    },
    "Evergreen Light": {
        "primary": "#38A169",  # Rich Medium Green (Chakra UI green.500)
        "secondary": "#48BB78", # Brighter Green (Chakra UI green.400)
        "background1": "#F9FFF6", # Very light, slightly green-tinted white
        "background2": "#F0FDF4", # (Chakra UI green.50)
        "background3": "#E6F5E9", # Slightly more saturated light green
        "text1": "#1A472A",    # Darkest Forest Green
        "text2": "#2F6C40",    # Forest Green
        "text3": "#4A8A5B",    # Medium Forest Green
        "placeholder_text": "#84A98C",
        "border": "#BEE3C8",    # Pale Green
        "shadow": "#E0EFE5",
        "accent_bg": "#38A169",
        "accent_fg": "#FFFFFF",
        "button_default_bg": "#E6F5E9",
        "button_hover": "#D9F0DE",
        "button_pressed": "#C9EAD6",
        "list_item_hover": "#EDF8EE",
        "history_disabled_text": "#4A8A5B",
        "drop_zone_bg": "#F0FDF4",
        "drop_zone_border_drag": "#38A169",
        "splitter_handle": "#F0FDF4"
    },
    "Cyberpunk Neon": {
        "primary": "#F000B8",  # Electric Pink
        "secondary": "#00F0FF", # Electric Cyan
        "background1": "#0D0221", # Deep Indigo/Purple base
        "background2": "#12082E", # Darker Purple/Blue
        "background3": "#1A0E3A", # Dark Purple
        "text1": "#E0D8FF",    # Light Lavender/Off-white
        "text2": "#B8A6FF",    # Lavender
        "text3": "#9070FF",    # Medium Purple
        "placeholder_text": "#6040C0",
        "border": "#402080",   # Dark violet border
        "shadow": "#05010F",    # Almost black with purple tint
        "accent_bg": "#F000B8",
        "accent_fg": "#0D0221", # Base bg for contrast
        "button_default_bg": "#1A0E3A", # Button bg based on dark purple
        "button_hover": "#2C184E", # Hover - slightly lighter purple
        "button_pressed": "#3E2262", # Pressed - even lighter purple
        "list_item_hover": "#221041",
        "history_disabled_text": "#9070FF",
        "drop_zone_bg": "#12082E",
        "drop_zone_border_drag": "#00F0FF", # Cyan for drop zone drag
        "splitter_handle": "#12082E"
    },
     "Royal Velvet": { # Luxurious deep purples and gold accents
        "primary": "#6A0DAD",  # Deep Purple (Purple Heart)
        "secondary": "#FFD700", # Gold
        "background1": "#241B2F", # Very Dark Desaturated Purple
        "background2": "#31253D", # Dark Desaturated Purple
        "background3": "#3E2E4A", # Medium Dark Desaturated Purple
        "text1": "#EAE6F0",    # Off-white with slight lavender tint
        "text2": "#C8C0D3",    # Light Lavender Grey
        "text3": "#A39AAF",    # Muted Lavender
        "placeholder_text": "#7D738A",
        "border": "#503F5F",   # Dark Purple Border
        "shadow": "#100A1A",    # Nearly black purple tint
        "accent_bg": "#FFD700", # Gold accent
        "accent_fg": "#4B0082", # Indigo / Darkest Purple for text on gold
        "button_default_bg": "#3E2E4A",
        "button_hover": "#4F3E5F",
        "button_pressed": "#604E74",
        "list_item_hover": "#3A2C47",
        "history_disabled_text": "#A39AAF",
        "drop_zone_bg": "#31253D",
        "drop_zone_border_drag": "#FFD700",
        "splitter_handle": "#31253D"
    },
}
    DEFAULT_THEME = "Claude Dark"
    HISTORY_FILE = Path("history/history.json")
    USER_SETTINGS_FILE = Path("history/UserSettings.json")
    MAX_HISTORY_ITEMS = 20
    ICON_BASE_SIZE = QSize(20,20)
    ICON_SCALE_FACTOR = 1.15
    PDF_BTN_BASE_SIZE = QSize(18, 18) # Smaller icons for PDF controls
    PDF_BTN_SCALE_FACTOR = 1.15

    # --- Icons ---
    ICON_HISTORY_ON = "icons/toggle-right.svg"
    ICON_HISTORY_OFF = "icons/toggle-left.svg"
    ICON_CLEAR_HISTORY = "icons/trash-2.svg"
    ICON_FOLDER_OPEN = "icons/folder.svg"
    ICON_REFRESH_CW = "icons/refresh-cw.svg"
    ICON_COPY = "icons/copy.svg"
    ICON_SAVE = "icons/save.svg"
    ICON_LAYOUT = "icons/layout.svg"
    ICON_TEXT_COLOR = "icons/text-color.svg"
    ICON_FONT = "icons/font.svg"
    ICON_IMAGE_PLACEHOLDER = "icons/image-placeholder.svg"
    # Icons for PDF Controls (placed in image area)
    ICON_PREV_PAGE = "icons/arrow-left-circle.svg"
    ICON_NEXT_PAGE = "icons/arrow-right-circle.svg"
    ICON_PROCESS_ALL = "icons/layers.svg"
    # --- ---

    PREDEFINED_TEXT_COLORS = {
        "Auto": None, "Black": "#000000", "White": "#FFFFFF", "Light Gray": "#D3D3D3",
        "Dark Gray": "#A9A9A9", "Soft Blue": "#AECBFA", "Mint Green": "#98FB98",
    }

    def __init__(self):
        super().__init__()
        self.font_family = "Inter, Segoe UI, Arial, sans-serif"
        self.history_enabled = True
        self._load_user_settings()
        self.colors = self.THEMES[self.current_theme_name]
        self.setWindowTitle("OCR Extraction Suite")
        self.setGeometry(100, 100, 1300, 800)
        self.setMinimumSize(1000, 650)
        self.setAcceptDrops(True)
        self.main_content_widget = None

        # File state
        self.current_file_path = None
        self.current_file_type = None  # File type: 'image', 'pdf', or None
        self.extracted_text = ""
        self.current_pixmap = None

        # PDF state
        self.current_pdf_doc = None
        self.current_pdf_page_num = -1
        self.total_pdf_pages = 0

        # Worker thread state
        self.worker_thread = None
        self.worker = None

        # UI Elements
        self.history_list = []
        self.theme_menu = QMenu(self)
        self.text_color_menu = QMenu(self)

        # PDF Control Buttons (inside image area)
        self.pdf_prev_button = None
        self.pdf_next_button = None
        self.pdf_process_all_button = None
        self.pdf_controls_widget = None

        self._ensure_history_dir()
        self._init_ui_structure()

        # Shadow effects for main panels
        try:
            shadow_blur_radius = 24
            shadow_offset = 4
            shadow_opacity = 60
            self.image_area_shadow = QGraphicsDropShadowEffect(self)
            self.image_area_shadow.setBlurRadius(shadow_blur_radius)
            self.image_area_shadow.setOffset(0, shadow_offset)
            self.image_area_widget.setGraphicsEffect(self.image_area_shadow)
            self.text_panel_shadow = QGraphicsDropShadowEffect(self)
            self.text_panel_shadow.setBlurRadius(shadow_blur_radius)
            self.text_panel_shadow.setOffset(0, shadow_offset)
            self.text_panel_frame.setGraphicsEffect(self.text_panel_shadow)
            self.history_area_shadow = QGraphicsDropShadowEffect(self)
            self.history_area_shadow.setBlurRadius(shadow_blur_radius)
            self.history_area_shadow.setOffset(0, shadow_offset)
            self.history_area_widget.setGraphicsEffect(self.history_area_shadow)
        except Exception as e:
            print(f"Shadow effect initialization error: {e}")

        self.apply_theme(self.current_theme_name)
        self._load_history()
        self._update_pdf_buttons_state() 

        # --- SplashOverlay setup ---
        try:
            user_name = getpass.getuser().capitalize()
        except Exception:
            user_name = "User"
        self.splash_overlay = SplashOverlay(self, logo_path="icons/logo.svg", welcome_text=f"Welcome, {user_name}!")
        self.splash_overlay.setGeometry(self.rect())
        self.splash_overlay.raise_()

    def _init_ui_structure(self):
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        top_h_layout = QHBoxLayout(central_widget)
        top_h_layout.setContentsMargins(12,12,12,12); top_h_layout.setSpacing(0)

        self._create_toolbar() # Toolbar creation moved before main content layout
        self.main_content_widget = QWidget()
        main_content_layout = QHBoxLayout(self.main_content_widget)
        main_content_layout.setContentsMargins(0,0,0,0); main_content_layout.setSpacing(0)
        top_h_layout.addWidget(self.main_content_widget) # Main content takes remaining space

        # --- Left Panel Setup ---
        left_processing_panel = QFrame(); left_processing_panel.setObjectName("LeftProcessingPanel")
        left_v_layout = QVBoxLayout(left_processing_panel)
        left_v_layout.setContentsMargins(0,0,0,0); left_v_layout.setSpacing(12)

        # Header
        header_frame = QFrame(); header_frame.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(20, 15, 20, 15); header_layout.setSpacing(10)
        self.main_title_label = QLabel("TextLens"); self.main_title_label.setObjectName("MainTitleLabel")
        self.file_info_label = QLabel("| No file selected"); self.file_info_label.setObjectName("FileInfoLabel")
        header_layout.addWidget(self.main_title_label); header_layout.addWidget(self.file_info_label); header_layout.addStretch(1)
        left_v_layout.addWidget(header_frame)

        # Vertical Splitter for Image/History
        self.image_text_splitter = QSplitter(Qt.Orientation.Vertical); self.image_text_splitter.setObjectName("ImageTextSplitter"); self.image_text_splitter.setHandleWidth(8)

        # Image Area Widget (Frame for border and shadow)
        self.image_area_widget = QFrame(); self.image_area_widget.setObjectName("ImageAreaWidget")
        image_area_outer_layout = QVBoxLayout(self.image_area_widget); 
        image_area_outer_layout.setContentsMargins(18,18,18,10); 
        image_area_outer_layout.setSpacing(8) 

        # Image Drop Label
        self.image_label = ImageDropLabel(self)
        self.image_label.setText(f"<img src='{self.ICON_IMAGE_PLACEHOLDER}' width='48' height='48'><br><br>Drag & Drop Image or PDF Here<br>or use Open button on toolbar")
        self.image_label.setWordWrap(True); self.image_label.setMinimumSize(250, 150); # Min height reduced slightly
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        image_area_outer_layout.addWidget(self.image_label, 1) 

        # --- PDF Controls Area (inside Image Area) ---
        self.pdf_controls_widget = QWidget(self.image_area_widget) # Container for controls
        self.pdf_controls_widget.setObjectName("PdfControlsWidget")
        pdf_controls_layout = QHBoxLayout(self.pdf_controls_widget)
        pdf_controls_layout.setContentsMargins(0, 5, 0, 5) # Padding around controls
        pdf_controls_layout.setSpacing(15)

        self.pdf_prev_button = QPushButton("Prev", self.pdf_controls_widget)
        self.pdf_prev_button.setObjectName("PdfPrevButton")
        self.pdf_prev_button.clicked.connect(self.previous_page)

        self.pdf_process_all_button = QPushButton("All", self.pdf_controls_widget)
        self.pdf_process_all_button.setObjectName("PdfAllButton")
        self.pdf_process_all_button.clicked.connect(self.process_all_pdf_pages)

        self.pdf_next_button = QPushButton("Next", self.pdf_controls_widget)
        self.pdf_next_button.setObjectName("PdfNextButton")
        self.pdf_next_button.clicked.connect(self.next_page)

        pdf_controls_layout.addStretch(1) # Push Prev button left
        pdf_controls_layout.addWidget(self.pdf_prev_button)
        pdf_controls_layout.addStretch(1) # Space between Prev and Process All
        pdf_controls_layout.addWidget(self.pdf_process_all_button)
        pdf_controls_layout.addStretch(1) # Space between Process All and Next
        pdf_controls_layout.addWidget(self.pdf_next_button)
        pdf_controls_layout.addStretch(1) # Push Next button right

        self.pdf_controls_widget.setVisible(False) 
        image_area_outer_layout.addWidget(self.pdf_controls_widget, 0) 
        # --- End PDF Controls Area ---

        self.image_text_splitter.addWidget(self.image_area_widget)

        # History Area (structure unchanged)
        self.history_area_widget = QFrame(); self.history_area_widget.setObjectName("HistoryAreaWidget")
        history_area_layout = QVBoxLayout(self.history_area_widget);
        history_area_layout.setContentsMargins(18,10,18,18); history_area_layout.setSpacing(10) # Match padding style
        history_panel_title_frame = QFrame(); history_panel_title_frame.setObjectName("PanelTitleFrame")
        history_panel_header_layout = QHBoxLayout(history_panel_title_frame)
        history_panel_header_layout.setContentsMargins(0,0,0,0); history_panel_header_layout.setSpacing(10)
        history_panel_title = QLabel("History"); history_panel_title.setObjectName("PanelTitleLabel")
        history_panel_header_layout.addWidget(history_panel_title); history_panel_header_layout.addStretch()
        history_area_layout.addWidget(history_panel_title_frame)
        self.history_list_widget = QListWidget(); self.history_list_widget.setObjectName("HistoryList")
        self.history_list_widget.itemDoubleClicked.connect(self._load_from_history_item)
        history_area_layout.addWidget(self.history_list_widget)
        self.image_text_splitter.addWidget(self.history_area_widget)

        self.image_text_splitter.setSizes([400, 250]) # Initial split ratio
        left_v_layout.addWidget(self.image_text_splitter) # Add splitter to left layout

        # Text Panel 
        self.text_panel_frame = QFrame(); self.text_panel_frame.setObjectName("TextPanelFrame")
        text_panel_layout = QVBoxLayout(self.text_panel_frame)
        text_panel_layout.setContentsMargins(18,10,18,18); text_panel_layout.setSpacing(10) # Match padding style
        text_panel_title_frame = QFrame(); text_panel_title_frame.setObjectName("PanelTitleFrame")
        text_panel_header_layout = QHBoxLayout(text_panel_title_frame)
        text_panel_header_layout.setContentsMargins(0,0,0,0); text_panel_header_layout.setSpacing(10)
        text_panel_title = QLabel("Extracted Text"); text_panel_title.setObjectName("PanelTitleLabel")
        text_panel_header_layout.addWidget(text_panel_title); text_panel_header_layout.addStretch()
        text_panel_layout.addWidget(text_panel_title_frame)
        self.text_edit = QTextEdit(); self.text_edit.setObjectName("TextEdit"); self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("Extracted text will appear here..."); self.text_edit.textChanged.connect(self.update_counts)
        text_panel_layout.addWidget(self.text_edit)

        # Top Level Splitter 
        self.top_level_splitter = QSplitter(Qt.Orientation.Horizontal); self.top_level_splitter.setObjectName("TopLevelSplitter"); self.top_level_splitter.setHandleWidth(8)
        self.top_level_splitter.addWidget(left_processing_panel); self.top_level_splitter.addWidget(self.text_panel_frame)
        self.top_level_splitter.setSizes([550, 450]); 
        self.top_level_splitter.setStretchFactor(0, 3); self.top_level_splitter.setStretchFactor(1, 2) 
        main_content_layout.addWidget(self.top_level_splitter)

        # Status Bar 
        self.statusBar().setObjectName("StatusBar")
        self.status_label = QLabel("Ready. Select or drop an image or PDF file.")
        self.count_label = QLabel("Chars: 0 | Words: 0"); self.count_label.setObjectName("InfoLabel")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.count_label)

        self.main_content_widget.setVisible(False)

    # (show_and_fade_in_content, _force_toolbar_icons_update, _ensure_history_dir)
    def show_and_fade_in_content(self):
        if self.main_content_widget:
            self.main_content_widget.setWindowOpacity(0.0)
            self.main_content_widget.setVisible(True)
            self.content_fade_in = QPropertyAnimation(self.main_content_widget, b"windowOpacity")
            self.content_fade_in.setDuration(600)
            self.content_fade_in.setStartValue(0.0)
            self.content_fade_in.setEndValue(1.0)
            self.content_fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.content_fade_in.finished.connect(self._force_toolbar_icons_update)
            self.content_fade_in.start()

    def _force_toolbar_icons_update(self):
        # print("Forcing toolbar icon update") 
        if hasattr(self, 'toolbar'):
            for action in self.toolbar.actions():
                widget = self.toolbar.widgetForAction(action)
                if isinstance(widget, HoverScaleToolButton):
                    # print(f"  Updating icon for: {widget.toolTip()}") 
                    widget.set_current_icon(widget.icon_path) # Refresh icon using its method
                elif widget:
                     # print(f"  Updating generic widget") 
                     widget.update()
            self.toolbar.update() # Update the toolbar itself


    def _ensure_history_dir(self):
        self.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # (_load_user_settings, _save_user_settings)
    def _load_user_settings(self):
        self.current_theme_name = self.DEFAULT_THEME
        self.user_text_color_name = "Default"
        self.user_font = QFont(self.font_family.split(',')[0].strip(), 10)
        if self.USER_SETTINGS_FILE.exists():
            try:
                with open(self.USER_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.current_theme_name = settings.get("last_theme", self.DEFAULT_THEME)
                    self.user_text_color_name = settings.get("text_color_name", "Default")
                    font_family = settings.get("font_family", self.font_family.split(',')[0].strip())
                    font_size = settings.get("font_size", 10)
                    font_bold = settings.get("font_bold", False)
                    font_italic = settings.get("font_italic", False)
                    self.user_font = QFont(font_family, font_size)
                    self.user_font.setBold(font_bold)
                    self.user_font.setItalic(font_italic)
                    self.history_enabled = settings.get("history_enabled", True)
            except (json.JSONDecodeError, IOError) as e: print(f"Error loading user settings: {e}. Using defaults.")
        if self.current_theme_name not in self.THEMES: self.current_theme_name = self.DEFAULT_THEME
        if hasattr(self, 'toggle_history_action'): self.update_history_toggle_appearance()

    def _save_user_settings(self):
        settings = {
            "last_theme": self.current_theme_name,
            "text_color_name": self.user_text_color_name,
            "font_family": self.user_font.family(),
            "font_size": self.user_font.pointSize(),
            "font_bold": self.user_font.bold(),
            "font_italic": self.user_font.italic(),
            "history_enabled": self.history_enabled
        }
        try:
            with open(self.USER_SETTINGS_FILE, 'w', encoding='utf-8') as f: json.dump(settings, f, indent=2)
        except IOError as e: print(f"Error saving user settings: {e}")

    # (_create_toolbar_action helper)
    def _create_toolbar_action(self, icon_path, text_for_action, tooltip, shortcut=None):
        action = QAction(text_for_action, self)
        if shortcut: action.setShortcut(shortcut)
        action.setText("") 
        button = HoverScaleToolButton(icon_path, base_size=self.ICON_BASE_SIZE, scale_factor=self.ICON_SCALE_FACTOR, parent=self.toolbar)
        button.setDefaultAction(action) 
        button.setToolTip(tooltip)      
        button.setMinimumWidth(self.ICON_BASE_SIZE.width() + 12) 
        button.setMaximumWidth(self.ICON_BASE_SIZE.width() + 12)

        widget_action = QWidgetAction(self)
        widget_action.setDefaultWidget(button)
        return action, widget_action 


    def _create_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar"); self.toolbar.setObjectName("MainToolBar")
        self.toolbar.setMovable(False); self.toolbar.setFloatable(False); self.toolbar.setOrientation(Qt.Orientation.Vertical)
        self.toolbar.setIconSize(self.ICON_BASE_SIZE) # Base size for toolbar icons
        self.toolbar.setFixedWidth(self.ICON_BASE_SIZE.width() + 35)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.toolbar)

        # --- Standard Actions ---
        self.open_action, open_widget_action = self._create_toolbar_action(self.ICON_FOLDER_OPEN, "Open", "Open Image/PDF (Ctrl+O)", "Ctrl+O")
        self.open_action.triggered.connect(self.open_file_dialog)
        self.toolbar.addAction(open_widget_action)

        self.reset_action, reset_widget_action = self._create_toolbar_action(self.ICON_REFRESH_CW, "Reset", "Reset UI (Ctrl+R)", "Ctrl+R")
        self.reset_action.triggered.connect(self.reset_ui)
        self.toolbar.addAction(reset_widget_action)

        self.toolbar.addSeparator()

        self.copy_action, self.copy_widget_action = self._create_toolbar_action(self.ICON_COPY, "Copy", "Copy Text (Ctrl+C)", "Ctrl+C")
        self.copy_action.triggered.connect(self.copy_text); self.copy_action.setEnabled(False)
        self.toolbar.addAction(self.copy_widget_action)

        self.save_action, self.save_widget_action = self._create_toolbar_action(self.ICON_SAVE, "Save", "Save Text (Ctrl+S)", "Ctrl+S")
        self.save_action.triggered.connect(self.save_text); self.save_action.setEnabled(False)
        self.toolbar.addAction(self.save_widget_action)

        self.toolbar.addSeparator()


        # --- History Actions ---
        self.toggle_history_button = HoverScaleToolButton("", base_size=self.ICON_BASE_SIZE, scale_factor=self.ICON_SCALE_FACTOR, parent=self.toolbar)
        self.toggle_history_action = QAction("", self) 
        self.toggle_history_button.setDefaultAction(self.toggle_history_action)
        self.toggle_history_action.triggered.connect(self.toggle_history_enabled)
        toggle_history_widget_action = QWidgetAction(self); toggle_history_widget_action.setDefaultWidget(self.toggle_history_button)
        self.toolbar.addAction(toggle_history_widget_action)

        self.clear_history_action, self.clear_history_widget_action = self._create_toolbar_action(self.ICON_CLEAR_HISTORY, "Clear History", "Clear History")
        self.clear_history_action.triggered.connect(self._clear_history)
        self.toolbar.addAction(self.clear_history_widget_action)

        self.toolbar.addSeparator()

        # --- Appearance Actions ---
        self.text_color_button_tb = HoverScaleToolButton(self.ICON_TEXT_COLOR, base_size=self.ICON_BASE_SIZE, scale_factor=self.ICON_SCALE_FACTOR, parent=self.toolbar)
        self.text_color_action = QAction("", self)
        self.text_color_button_tb.setDefaultAction(self.text_color_action)
        self.text_color_button_tb.setToolTip("Select Text Color")
        self.text_color_action.triggered.connect(self.show_text_color_menu)
        text_color_widget_action = QWidgetAction(self); text_color_widget_action.setDefaultWidget(self.text_color_button_tb)
        self.toolbar.addAction(text_color_widget_action)

        self.font_button_tb = HoverScaleToolButton(self.ICON_FONT, base_size=self.ICON_BASE_SIZE, scale_factor=self.ICON_SCALE_FACTOR, parent=self.toolbar)
        self.font_action = QAction("", self)
        self.font_button_tb.setDefaultAction(self.font_action)
        self.font_button_tb.setToolTip("Select Font")
        self.font_action.triggered.connect(self.select_font)
        font_widget_action = QWidgetAction(self); font_widget_action.setDefaultWidget(self.font_button_tb)
        self.toolbar.addAction(font_widget_action)

        # --- Spacer ---
        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.toolbar.addWidget(spacer)

        # --- Theme Action (at the bottom) ---
        self.theme_button_tb = HoverScaleToolButton(self.ICON_LAYOUT, base_size=self.ICON_BASE_SIZE, scale_factor=self.ICON_SCALE_FACTOR, parent=self.toolbar)
        self.theme_action = QAction("", self)
        self.theme_button_tb.setDefaultAction(self.theme_action)
        self.theme_button_tb.setToolTip("Change Theme")
        self.theme_action.triggered.connect(self.show_theme_menu)
        theme_widget_action = QWidgetAction(self); theme_widget_action.setDefaultWidget(self.theme_button_tb)
        self.toolbar.addAction(theme_widget_action) 

        QTimer.singleShot(0, self.update_history_toggle_appearance)


    def apply_theme(self, theme_name):
        if theme_name not in self.THEMES: theme_name = self.DEFAULT_THEME
        self.current_theme_name = theme_name
        self.colors = self.THEMES[self.current_theme_name]
        c = self.colors; base_font = QFont(self.font_family.split(',')[0].strip(), 10); QApplication.setFont(base_font)

        shadow_color = QColor(c.get('shadow', '#000000'))
        shadow_color.setAlpha(60) 
        if hasattr(self, 'image_area_shadow'): self.image_area_shadow.setColor(shadow_color)
        if hasattr(self, 'text_panel_shadow'): self.text_panel_shadow.setColor(shadow_color)
        if hasattr(self, 'history_area_shadow'): self.history_area_shadow.setColor(shadow_color)

        self.apply_text_edit_style()
        normal_drop_style = f"""QLabel#ImageDropLabel {{ color: {c['placeholder_text']}; background-color: {c['drop_zone_bg']}; border: 1px solid {c['border']}; border-radius: 8px; font-size: 11pt; padding: 20px; }}"""
        drag_over_drop_style = f"""QLabel#ImageDropLabel {{ color: {c['accent_fg']}; background-color: {c['accent_bg']}; border: 2px solid {c['primary']}; border-radius: 8px; font-size: 11pt; padding: 20px; }}"""
        if hasattr(self, 'image_label'): self.image_label.setStyles(normal_drop_style, drag_over_drop_style)

        if hasattr(self, 'image_label') and not self.current_pixmap:
             self.image_label.setText(f"<img src='{self.ICON_IMAGE_PLACEHOLDER}' width='48' height='48'><br><br>Drag & Drop Image or PDF Here<br>or use Open button on toolbar")

        splitter_handle_bg = c.get('splitter_handle', c['background2'])

        stylesheet = f"""
            QMainWindow {{ background-color: {c['background1']}; }}
            QWidget {{ color: {c['text1']}; font-family: '{self.font_family}'; }}
            QFrame#HeaderFrame {{ background-color: {c['background1']}; border-bottom: 1px solid {c['border']}; padding: 2px 0; }}
            QLabel#MainTitleLabel {{ font-size: 13pt; font-weight: 500; color: {c['text1']}; padding: 8px 0px 8px 0px; background: transparent;}}
            QLabel#FileInfoLabel {{ font-size: 9pt; color: {c['text3']}; padding: 8px 0 8px 5px; background: transparent;}}
            QLabel {{ color: {c['text2']}; }}
            QFrame#LeftProcessingPanel {{ background-color: transparent; border: none; }}
            QFrame#PanelTitleFrame {{ background-color: transparent; border-bottom: 1px solid {c['border']}; }}
            QLabel#PanelTitleLabel {{ font-size: 10pt; font-weight: 500; color: {c['text1']}; padding: 10px 12px; background: transparent; }}
            QFrame#HistoryAreaWidget, QFrame#TextPanelFrame {{ /* Target the frames for styling */
                background-color: {c['background2']}; border: 1.5px solid {c['border']}; border-radius: 12px;
                 padding: 18px 15px; /* Outer frame padding */
            }}
             /* Specific frame styling, ensuring padding set on inner layout or here */
            QFrame#HistoryAreaWidget {{ padding: 0; /* Use layout margins */ }}
            QFrame#HistoryAreaWidget > QVBoxLayout {{ /* Target direct child layout */
                 contentsMargins: 18px 10px 18px 18px;
             }}
             QFrame#TextPanelFrame {{ padding: 0; }}
             QFrame#TextPanelFrame > QVBoxLayout {{
                 contentsMargins: 18px 10px 18px 18px;
             }}
             QFrame#ImageAreaWidget {{ /* Style for image area container */
                background-color: {c['background1']}; border: 1.5px solid {c['border']}; border-radius: 12px;
                 padding: 0; /* Padding handled by its layout */
             }}
            QFrame#ImageAreaWidget > QVBoxLayout {{ /* Target direct child layout */
                 contentsMargins: 18px 18px 10px 18px; /* Adjust for controls */
            }}
            QWidget#PdfControlsWidget {{ background-color: transparent; border: none; }}
            /* Style the PDF Control Buttons specifically */
            QPushButton#PdfPrevButton, QPushButton#PdfNextButton, QPushButton#PdfAllButton {{
                 background-color: {c.get('button_default_bg', 'transparent')};
                 border: none;
                 border-radius: 4px;
                 padding: 4px 8px;
                 margin: 0px;
                 color: {c['text1']};
            }}
            QPushButton#PdfPrevButton:hover, QPushButton#PdfNextButton:hover, QPushButton#PdfAllButton:hover {{ 
                background-color: {c['button_hover']}; 
            }}
            QPushButton#PdfPrevButton:pressed, QPushButton#PdfNextButton:pressed, QPushButton#PdfAllButton:pressed {{ 
                background-color: {c['button_pressed']}; 
            }}
            QPushButton#PdfPrevButton:disabled, QPushButton#PdfNextButton:disabled, QPushButton#PdfAllButton:disabled {{ 
                background-color: transparent; 
                opacity: 0.4; 
            }}

            QListWidget#HistoryList {{ background-color: transparent; border: none; padding: 0px; }}
            QListWidget#HistoryList::item {{ padding: 7px 10px; color: {c['text2']}; border-radius: 4px; border: none; margin-bottom: 3px; font-size: 9pt; }}
            QListWidget#HistoryList::item:disabled {{ color: {c.get('history_disabled_text', c['text3'])}; background-color: transparent; }}
            QListWidget#HistoryList::item:hover {{ background-color: {c.get('list_item_hover', c['button_hover'])}; color: {c['text1']}; }}
            QListWidget#HistoryList::item:selected {{ background-color: {c['primary']}; color: {c['accent_fg']}; }}
            QToolBar#MainToolBar {{ background-color: {c['background2']}; border-right: 1px solid {c['border']}; spacing: 0px; padding: 6px 0px; width: {self.ICON_BASE_SIZE.width() + 24}px; }} /* Width set in code */
            QToolBar#MainToolBar HoverScaleToolButton {{ /* Target only toolbar buttons */
                background-color: {c.get('button_default_bg', 'transparent')}; border: none; border-radius: 6px; padding: 6px; margin: 3px 4px;
                min-width: {self.ICON_BASE_SIZE.width() + 12}px; max-width: {self.ICON_BASE_SIZE.width() + 12}px;
                min-height: {self.ICON_BASE_SIZE.height() + 12}px; max-height: {self.ICON_BASE_SIZE.height() + 12}px;
             }}
            QToolBar#MainToolBar HoverScaleToolButton:hover {{ background-color: {c['button_hover']}; }}
            QToolBar#MainToolBar HoverScaleToolButton:pressed {{ background-color: {c['button_pressed']}; }}
            QToolBar#MainToolBar HoverScaleToolButton:disabled {{ background-color: transparent; opacity: 0.5; }}
            QToolBar::separator {{ background-color: {c['border']}; width: 50%; height: 1px; margin: 7px 25%; }}
            /* QLabel#ImageDropLabel Style applied by setStyles */
            QTextEdit#TextEdit {{
                background-color: transparent; border: none; border-radius: 8px; padding: 0px; /* No padding, frame has it */
                selection-background-color: {c['secondary']}; selection-color: {c['accent_fg']};
            }}
            QTextEdit#TextEdit[placeholderText] {{ color: {c['placeholder_text']}; }}
            QLabel#InfoLabel {{ color: {c['text3']}; font-size: 8pt; padding: 0px; }}
            QStatusBar {{ background-color: {c['background1']}; border-top: 1px solid {c['border']}; padding: 3px 0; }}
            QStatusBar QLabel {{ color: {c['text2']}; font-size: 9pt; padding: 0 10px; background: transparent;}}
            QSplitter::handle {{ background-color: {splitter_handle_bg}; }}
            QSplitter::handle:horizontal {{ width: 8px; margin: 0px 0px; border-left: 1px solid {c['border']}; border-right: 1px solid {c['border']}; }}
            QSplitter::handle:vertical {{ height: 8px; margin: 0px 0px; border-top: 1px solid {c['border']}; border-bottom: 1px solid {c['border']}; }}
            QSplitter::handle:hover {{ background-color: {c['primary']}; }}
            QToolTip {{ background-color: {c['background3']}; color: {c['text1']}; border: 1px solid {c['border']}; padding: 5px; border-radius: 3px; opacity: 230; }}
            QMenu {{ background-color: {c['background2']}; border: 1px solid {c['border']}; padding: 5px; color: {c['text1']}; }}
            QMenu::item {{ padding: 6px 22px 6px 22px; background-color: transparent; border-radius: 4px; }}
            QMenu::item:selected {{ background-color: {c['primary']}; color: {c['accent_fg']}; }}
            QMenu::item:checked {{ font-weight: 500; background-color: {c.get('button_default_bg', c['background3'])}; }}
            QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 0px; }}
            /* Scrollbar Styling */
            QScrollBar:vertical {{
                background-color: {c['background2']};
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {c['border']};
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {c['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
                background: none;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
            QScrollBar:horizontal {{
                background-color: {c['background2']};
                height: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {c['border']};
                min-width: 30px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background-color: {c['primary']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
                background: none;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: none;
            }}
            QSplitter::handle {{
                background: transparent;
                background-color: transparent;
                border: none;
                margin: 0;
            }}
            QSplitter::handle:horizontal {{
                min-width: 6px;
                max-width: 8px;
                background: transparent;
                background-color: transparent;
                border: none;
                margin: 0;
            }}
            QSplitter::handle:vertical {{
                min-height: 6px;
                max-height: 8px;
                background: transparent;
                background-color: transparent;
                border: none;
                margin: 0;
            }}
            QSplitter::handle:hover {{
                background: transparent;
                background-color: transparent;
                border: none;
            }}
         """
        self.setStyleSheet(stylesheet)
        self.statusBar().setStyleSheet(f"background-color: {c['background1']}; border-top: 1px solid {c['border']};")

        self.update_history_toggle_appearance()
        self._populate_history_widget_items() 
        self._force_toolbar_icons_update() 
        self.apply_text_edit_style() 


    # (apply_text_edit_style, show_text_color_menu, handle_text_color_selection, select_font, show_theme_menu, handle_theme_selection)
    def apply_text_edit_style(self):
        if hasattr(self, 'text_edit'):
            self.text_edit.setFont(self.user_font)
            if self.user_text_color_name == "Default":
                text_color_hex = self.colors.get('text1', '#E0E0E0')
            else:
                text_color_hex = self.PREDEFINED_TEXT_COLORS.get(self.user_text_color_name, self.colors.get('text1'))

            self.text_edit.setStyleSheet(f"""
                QTextEdit#TextEdit {{
                    color: {text_color_hex};
                    background-color: transparent;
                    border: none;
                    padding: 0px; /* padding is handled by TextPanelFrame */
                    selection-background-color: {self.colors['secondary']};
                    selection-color: {self.colors['accent_fg']};
                }}
                QTextEdit#TextEdit[placeholderText] {{
                    color: {self.colors['placeholder_text']};
                }}
            """)

    def show_text_color_menu(self):
        """Shows the text color selection menu."""
        self.text_color_menu.clear()
        
        for color_name in self.PREDEFINED_TEXT_COLORS:
            action = QAction(color_name, self)
            action.setCheckable(True)
            action.setChecked(color_name == self.user_text_color_name)
            # Fix: Use a lambda that captures the color_name correctly
            action.triggered.connect(lambda checked, name=color_name: self.handle_text_color_selection(name))
            self.text_color_menu.addAction(action)
        
        # Show menu at button position
        button = self.toolbar.widgetForAction(self.text_color_action)
        if button:
            menu_pos = button.mapToGlobal(button.rect().bottomLeft())
            self.text_color_menu.exec(menu_pos)
        else:
            # Fallback to cursor position if button not found
            self.text_color_menu.exec(QCursor.pos())

    def handle_text_color_selection(self, color_name):
        """Handles text color selection from menu."""
        print(f"Color selected: {color_name}")  # Debug print
        self.user_text_color_name = color_name
        self._save_user_settings()
        
        # Get the color value
        if color_name == "Default":
            text_color_hex = self.colors.get('text1', '#E0E0E0')
        else:
            text_color_hex = self.PREDEFINED_TEXT_COLORS.get(color_name, self.colors.get('text1'))
        
        # Apply the color directly to the text edit
        if hasattr(self, 'text_edit'):
            self.text_edit.setStyleSheet(f"""
                QTextEdit#TextEdit {{
                    color: {text_color_hex};
                    background-color: transparent;
                    border: none;
                    border-radius: 8px;
                    padding: 0px;
                }}
            """)
            print(f"Applied color: {text_color_hex}")  

    def select_font(self):
        original_app_palette = QApplication.palette()
        dialog_palette = QPalette()
        dialog_bg_color = QColor(self.colors.get('background2', "#2D2D2D"))
        dialog_text_color = QColor(self.colors.get('text1', "#FFFFFF"))
        dialog_palette.setColor(QPalette.ColorRole.Window, dialog_bg_color)
        dialog_palette.setColor(QPalette.ColorRole.WindowText, dialog_text_color)
        dialog_palette.setColor(QPalette.ColorRole.Base, QColor(self.colors.get('background1', "#232323")))
        dialog_palette.setColor(QPalette.ColorRole.Text, QColor(self.colors.get('text1', "#FFFFFF"))) 
        dialog_palette.setColor(QPalette.ColorRole.Button, QColor(self.colors.get('button_default_bg', "#333333")))
        dialog_palette.setColor(QPalette.ColorRole.ButtonText, dialog_text_color)
        dialog_palette.setColor(QPalette.ColorRole.Highlight, QColor(self.colors.get('primary', "#8AB4F8")))
        dialog_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(self.colors.get('accent_fg', "#000000")))
        if hasattr(QPalette.ColorRole, "PlaceholderText"):
             dialog_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(self.colors.get('placeholder_text', "#A0A0A0")))
        QApplication.setPalette(dialog_palette)

        font_dialog = QFontDialog(self)
        font_dialog.setCurrentFont(self.user_font)
        font_dialog.setWindowTitle("Select Font for Extracted Text")
        font_dialog.setStyleSheet(f"""
            QFontDialog {{ background-color: {self.colors.get('background2', '#2D2D2D')}; color: {self.colors.get('text1', '#FFFFFF')}; }}
            QLabel, QCheckBox, QGroupBox, QRadioButton {{ color: {self.colors.get('text1', '#FFFFFF')}; background-color: transparent; }}
            QLineEdit, QComboBox, QListView {{ background-color: {self.colors.get('background1', '#1E1E1E')}; color: {self.colors.get('text1', '#FFFFFF')}; border: 1px solid {self.colors.get('border', '#4A4A4A')}; selection-background-color: {self.colors.get('primary', '#8AB4F8')}; selection-color: {self.colors.get('accent_fg', '#000000')}; }}
            QListView::item {{ color: {self.colors.get('text1', '#FFFFFF')}; }}
            QListView::item:selected {{ background-color: {self.colors.get('primary', '#8AB4F8')}; color: {self.colors.get('accent_fg', '#000000')}; }}
            QComboBox QAbstractItemView {{ background-color: {self.colors.get('background1', '#1E1E1E')}; color: {self.colors.get('text1', '#FFFFFF')}; selection-background-color: {self.colors.get('primary', '#8AB4F8')}; selection-color: {self.colors.get('accent_fg', '#000000')}; }}
            QPushButton {{ background-color: {self.colors.get('button_default_bg', '#333333')}; color: {self.colors.get('text1', '#FFFFFF')}; border: 1px solid {self.colors.get('border', '#4A4A4A')}; padding: 5px 10px; border-radius: 4px; min-height: 20px; }}
            QPushButton:hover {{ background-color: {self.colors.get('button_hover', '#3C4043')}; }}
            QPushButton:pressed {{ background-color: {self.colors.get('button_pressed', '#484C4F')}; }}
            QPushButton:disabled {{ background-color: {self.colors.get('button_default_bg', '#404040')}; color: {self.colors.get('text3', '#808080')}; border-color: {self.colors.get('border', '#505050')}; }}
        """)

        font, ok = QFontDialog.getFont(self.user_font, self, "Select Font for Extracted Text")
        QApplication.setPalette(original_app_palette) # Restore original palette
        if ok:
            self.user_font = font
            self.apply_text_edit_style()
            self._save_user_settings()
            print(f"Font selected: {font.family()}, {font.pointSize()}")
        else:
            print("Font selection cancelled.")

    def show_theme_menu(self):
        self.theme_menu.clear()
        for theme_name in self.THEMES.keys():
            action = QAction(theme_name, self); action.setCheckable(True)
            action.setChecked(theme_name == self.current_theme_name)
            action.triggered.connect(self.handle_theme_selection)
            self.theme_menu.addAction(action)
        button_widget = self.theme_button_tb
        if button_widget:
            # Position menu to the right of the button, vertically centered
             button_pos = button_widget.mapToGlobal(button_widget.rect().bottomLeft())
             menu_y_offset = -(self.theme_menu.sizeHint().height() // 2) + (button_widget.height() // 2)
             self.theme_menu.exec(button_pos + QPoint(button_widget.width() + 10, menu_y_offset ))
        else: self.theme_menu.exec(QCursor.pos())

    def handle_theme_selection(self):
        action = self.sender()
        if action and isinstance(action, QAction):
            self.apply_theme(action.text())
            self._save_user_settings() # Save theme selection

    # (update_history_toggle_appearance, _load_history, _display_history_disabled_message, _save_history, _add_to_history, _populate_history_widget_items, _load_from_history_item, _clear_history, toggle_history_enabled)
    def update_history_toggle_appearance(self):
        if hasattr(self, 'toggle_history_button'):
            icon_path = self.ICON_HISTORY_ON if self.history_enabled else self.ICON_HISTORY_OFF
            self.toggle_history_button.set_current_icon(icon_path)
            self.toggle_history_button.setToolTip("History is ON. Click to Disable." if self.history_enabled else "History is OFF. Click to Enable.")
            if hasattr(self, 'clear_history_action') and hasattr(self, 'clear_history_widget_action'):
                self.clear_history_action.setEnabled(self.history_enabled)
                widget = self.toolbar.widgetForAction(self.clear_history_widget_action)
                if widget:
                    widget.setEnabled(self.history_enabled)
                    # Force icon refresh for clear history button
                    if isinstance(widget, HoverScaleToolButton):
                        widget.set_current_icon(self.ICON_CLEAR_HISTORY)


    def _load_history(self):
        self.history_list_widget.clear()
        if self.history_enabled:
            if self.HISTORY_FILE.exists():
                try:
                    with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f: self.history_list = json.load(f)
                except (json.JSONDecodeError, IOError): self.history_list = []
            else: self.history_list = []
            self._populate_history_widget_items()
        else: self._display_history_disabled_message()

    def _display_history_disabled_message(self):
        self.history_list_widget.clear()
        disabled_item = QListWidgetItem("History is currently disabled.")
        disabled_item.setFlags(disabled_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
        disabled_color = self.colors.get('history_disabled_text', self.colors.get('text3', '#9E9E9E'))
        disabled_item.setForeground(QColor(disabled_color))
        self.history_list_widget.addItem(disabled_item)

    def _save_history(self):
        if not self.history_enabled: return
        try:
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(self.history_list, f, indent=2)
        except IOError as e: print(f"Error saving history: {e}")

    def _add_to_history(self, file_path):
        if not self.history_enabled: return
        abs_path = str(Path(file_path).resolve())
        if abs_path in self.history_list: self.history_list.remove(abs_path)
        self.history_list.insert(0, abs_path)
        self.history_list = self.history_list[:self.MAX_HISTORY_ITEMS]
        self._save_history(); self._populate_history_widget_items()

    def _populate_history_widget_items(self):
        self.history_list_widget.clear()
        if not self.history_enabled:
            self._display_history_disabled_message()
            if hasattr(self, 'clear_history_action'): self.clear_history_action.setEnabled(False)
            widget = self.toolbar.widgetForAction(self.clear_history_widget_action) # Also update button visually
            if widget: widget.setEnabled(False)
            return

        # Enable/disable clear button based on list content and enabled state
        can_clear = self.history_enabled and bool(self.history_list)
        if hasattr(self, 'clear_history_action'): self.clear_history_action.setEnabled(can_clear)
        widget = self.toolbar.widgetForAction(self.clear_history_widget_action)
        if widget: widget.setEnabled(can_clear)


        if not self.history_list:
            no_history_item = QListWidgetItem("No history yet.")
            no_history_item.setFlags(no_history_item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
            disabled_color = self.colors.get('history_disabled_text', self.colors.get('text3', '#9E9E9E'))
            no_history_item.setForeground(QColor(disabled_color))
            self.history_list_widget.addItem(no_history_item)
            return

        for path_str in self.history_list:
            path_obj = Path(path_str)
            # Use specific icons for file types in history
            icon_name = "file-text.svg" # Default
            if path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
                icon_name = "image.svg"
            elif path_obj.suffix.lower() == '.pdf':
                icon_name = "file-text.svg" 
            item_icon = QIcon(f"icons/{icon_name}")

            item_text = f"{path_obj.name}  ({path_obj.parent.name})"
            item = QListWidgetItem(item_icon, item_text)
            item.setData(Qt.ItemDataRole.UserRole, path_str); item.setToolTip(path_str)
            self.history_list_widget.addItem(item)


    def _load_from_history_item(self, item):
        if not item or not (item.flags() & Qt.ItemFlag.ItemIsEnabled): return 
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and Path(file_path).exists():
            self.process_file(file_path) 
        elif file_path:
            self.show_error("History Error", f"File not found:\n{file_path}\nIt may have been moved or deleted.")
            if file_path in self.history_list: self.history_list.remove(file_path)
            self._save_history(); self._populate_history_widget_items()

    def _clear_history(self):
        if not self.history_enabled or not self.history_list: return 
        reply = QMessageBox.question(self, "Clear History", "Are you sure you want to clear all history items?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.history_list = []; self._save_history(); self._populate_history_widget_items()
            self.update_status("History cleared.")

    def toggle_history_enabled(self):
        self.history_enabled = not self.history_enabled
        self._save_user_settings()
        self.update_history_toggle_appearance() 
        self._load_history() 

    def update_status(self, message):
        self.status_label.setText(message); QApplication.processEvents()
    def show_error(self, title, message): QMessageBox.critical(self, title, message)
    def show_warning(self, title, message): QMessageBox.warning(self, title, message)
    def show_info(self, title, message): QMessageBox.information(self, title, message)

    # (File Handling: open_file_dialog, handle_dropped_file)
    def open_file_dialog(self):
        start_dir = str(Path.home())
        img_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)"
        pdf_filter = "PDF Files (*.pdf)" if fitz else ""
        all_filter = "All Files (*)"

        filters = [img_filter]
        if pdf_filter: filters.append(pdf_filter)
        filters.append(all_filter)
        file_filter_str = ";;".join(filters)

        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Image or PDF File", start_dir, file_filter_str)
        if file_path:
             self.process_file(file_path)

    def handle_dropped_file(self, mime_data: QMimeData):
        """Handles a file dropped onto the ImageDropLabel."""
        if mime_data.hasUrls():
            url = mime_data.urls()[0]
            if url.isLocalFile():
                file_path = url.toLocalFile()
                accepted_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
                if fitz: accepted_extensions.append('.pdf')
                if any(file_path.lower().endswith(ext) for ext in accepted_extensions):
                    self.process_file(file_path)
                else:
                    self.show_warning("Unsupported File", f"Unsupported file type dropped: {Path(file_path).suffix}")

    def process_file(self, file_path):
        """Determines file type and initiates processing."""
        if self.worker_thread and self.worker_thread.isRunning():
             self.show_warning("Busy", "Processing is already in progress. Please wait or reset."); return

        self._reset_pdf_state() 
        self.current_file_path = file_path
        file_ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).name

        self.text_edit.clear(); self.extracted_text = ""
        self.copy_action.setEnabled(False); self.save_action.setEnabled(False)
        self.update_counts()
        self.current_pixmap = None 

        if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            self.current_file_type = 'image'
            self.file_info_label.setText(f"| Image: {file_name}")
            self.update_status(f"Loading image: {file_name}...")
            self._update_pdf_buttons_state() 

            self.current_pixmap = QPixmap(file_path)
            if self.current_pixmap.isNull():
                self.show_error("Load Error", f"Could not load image file:\n{file_path}")
                self.image_label.setText("Failed to load image."); self.image_label.setStyleSheet(self.image_label.normal_style)
                self.update_status("Error loading image.")
                self.current_pixmap = None; self.file_info_label.setText(" | Error loading image")
                self.current_file_type = None
                self._update_pdf_buttons_state() 
                return
            else:
                self.display_image_preview() 

            if self.history_enabled: self._add_to_history(file_path)

            self.update_status(f"Processing image: {file_name}...")
            self.text_edit.setPlaceholderText(f"Extracting text from {file_name}...")
            self.run_task(process_image_extract_text, self.handle_ocr_result, self.current_file_path)

        elif file_ext == '.pdf' and fitz:
            self.current_file_type = 'pdf'
            self.update_status(f"Loading PDF: {file_name}...")
            try:
                self.current_pdf_doc = fitz.open(file_path)
                self.total_pdf_pages = self.current_pdf_doc.page_count
                self.current_pdf_page_num = 0 # Start at first page

                if self.total_pdf_pages == 0:
                    self.show_warning("Empty PDF", "The selected PDF file has no pages.")
                    self._reset_pdf_state()
                    self.reset_ui()
                    return

                self.file_info_label.setText(f"| PDF: {file_name} (Page {self.current_pdf_page_num + 1}/{self.total_pdf_pages})")
                self._update_pdf_buttons_state() 

                if self.history_enabled: self._add_to_history(file_path)

                self.display_and_process_pdf_page(self.current_pdf_page_num)

            except fitz.fitz.FileNotFoundError:
                 self.show_error("PDF Error", f"File not found:\n{file_path}")
                 self.reset_ui(); return
            except fitz.fitz.FileDataError:
                 self.show_error("PDF Error", f"Cannot open or read PDF:\n{file_name}\nFile may be corrupted or password-protected.")
                 self.reset_ui(); return
            except Exception as e:
                 self.show_error("PDF Load Error", f"An unexpected error occurred loading the PDF:\n{e}")
                 self.reset_ui(); return
        else:
            self.show_error("Unsupported File", f"Unsupported file type: {file_ext}" + ("\n(PDF processing requires PyMuPDF library)" if file_ext == '.pdf' else ""))
            self.reset_ui()


    def display_and_process_pdf_page(self, page_index):
        """Renders, displays, and initiates OCR for a specific PDF page."""
        if not self.current_pdf_doc or page_index < 0 or page_index >= self.total_pdf_pages:
            return

        if self.worker_thread and self.worker_thread.isRunning():
            self.update_status("Waiting for previous task to finish...")
            return

        self.update_status(f"Loading PDF Page {page_index + 1}/{self.total_pdf_pages}...")
        self.text_edit.clear(); self.extracted_text = ""
        self.copy_action.setEnabled(False); self.save_action.setEnabled(False)
        self.update_counts()

        # Render page to image data (using source function)
        page_image_data, error = render_pdf_page_to_image_data(self.current_pdf_doc, page_index)

        if error or page_image_data is None:
            self.show_error("PDF Render Error", f"Could not render page {page_index + 1}: {error}")
            self.image_label.setText(f"Failed to render page {page_index + 1}."); self.image_label.setStyleSheet(self.image_label.normal_style)
            self.current_pixmap = None
            self.update_status(f"Error rendering page {page_index + 1}.")
            return

        # Convert OpenCV BGR to QImage/QPixmap
        try:
            height, width, channel = page_image_data.shape
            bytes_per_line = 3 * width
            # Crucially, BGR -> RGB conversion needed for QImage
            q_image = QImage(cv2.cvtColor(page_image_data, cv2.COLOR_BGR2RGB).data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            # Alternative if source returns RGB: QImage.Format.Format_RGB888

            self.current_pixmap = QPixmap.fromImage(q_image)
            if self.current_pixmap.isNull():
                 raise ValueError("QPixmap conversion resulted in null pixmap")
            self.display_image_preview() 
        except Exception as e:
             self.show_error("Display Error", f"Could not display rendered PDF page {page_index + 1}: {e}")
             self.image_label.setText(f"Error displaying page {page_index + 1}."); self.image_label.setStyleSheet(self.image_label.normal_style)
             self.current_pixmap = None
             self.update_status(f"Error displaying page {page_index + 1}.")
             return

        # Start OCR for the rendered page
        self.file_info_label.setText(f"| PDF: {Path(self.current_file_path).name} (Page {page_index + 1}/{self.total_pdf_pages})")
        self.update_status(f"Processing PDF Page {page_index + 1}/{self.total_pdf_pages}...")
        self.text_edit.setPlaceholderText(f"Extracting text from page {page_index + 1}...")

        self.run_task(process_pdf_page_extract_text, self.handle_ocr_result, self.current_pdf_doc, page_index)

    def display_image_preview(self):
        # Show the current pixmap (image or PDF page) in the image label
        if self.current_pixmap and not self.current_pixmap.isNull():
            try:
                available_size = self.image_label.size()
                scaled_pixmap = self.current_pixmap.scaled(
                    available_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)

                self.image_label.setPixmap(scaled_pixmap)
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            except Exception as e:
                print(f"Error scaling/setting pixmap: {e}")
                self.image_label.setText("Error displaying preview.")
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.image_label.setStyleSheet(self.image_label.normal_style)
        elif hasattr(self, 'image_label'):
            # Show placeholder if no pixmap
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"<img src='{self.ICON_IMAGE_PLACEHOLDER}' width='48' height='48'><br><br>Drag & Drop Image or PDF Here<br>or use Open button on toolbar")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet(self.image_label.normal_style)


    # --- PDF Navigation ---
    def _update_pdf_buttons_state(self):
        # Update visibility and enabled state of PDF navigation buttons
        is_pdf = (self.current_file_type == 'pdf' and self.total_pdf_pages > 0)

        if hasattr(self, 'pdf_controls_widget'):
            self.pdf_controls_widget.setVisible(is_pdf)

        if is_pdf and self.pdf_prev_button and self.pdf_next_button and self.pdf_process_all_button:
            # Enable/disable based on current page number and worker state
            is_processing = self.worker_thread is not None and self.worker_thread.isRunning()

            can_go_prev = not is_processing and self.current_pdf_page_num > 0
            can_go_next = not is_processing and self.current_pdf_page_num < self.total_pdf_pages - 1
            can_process_all = not is_processing and self.total_pdf_pages > 0 # Allow processing even on 1 page PDFs

            self.pdf_prev_button.setEnabled(can_go_prev)
            self.pdf_next_button.setEnabled(can_go_next)
            self.pdf_process_all_button.setEnabled(can_process_all)

        # If not PDF, ensure buttons (though hidden) are marked disabled
        elif not is_pdf and self.pdf_prev_button and self.pdf_next_button and self.pdf_process_all_button:
            self.pdf_prev_button.setEnabled(False)
            self.pdf_next_button.setEnabled(False)
            self.pdf_process_all_button.setEnabled(False)

    # (_reset_pdf_state - Unchanged, already called _update_pdf_buttons_state)
    def _reset_pdf_state(self):
        # Close the current PDF document and reset PDF state
        if self.current_pdf_doc:
            try:
                self.current_pdf_doc.close()
                # print("Closed previous PDF document.")
            except Exception as e:
                print(f"Warning: Error closing PDF document: {e}")
        self.current_pdf_doc = None
        self.current_pdf_page_num = -1
        self.total_pdf_pages = 0
        if self.current_file_type == 'pdf': # Only reset if it was a pdf
            self.current_file_type = None
        self._update_pdf_buttons_state() # Hide/disable buttons


    def previous_page(self):
        print(f"[DEBUG] previous_page called. current_file_type={self.current_file_type}, current_pdf_page_num={self.current_pdf_page_num}")
        if self.current_file_type == 'pdf' and self.current_pdf_page_num > 0:
            self.current_pdf_page_num -= 1
            self._update_pdf_buttons_state() # Disable buttons during load/process
            self.display_and_process_pdf_page(self.current_pdf_page_num)
        else:
            self.update_status("Already at the first page or not a PDF.")

    def next_page(self):
        print(f"[DEBUG] next_page called. current_file_type={self.current_file_type}, current_pdf_page_num={self.current_pdf_page_num}, total_pdf_pages={self.total_pdf_pages}")
        if self.current_file_type == 'pdf' and self.current_pdf_page_num < self.total_pdf_pages - 1:
            self.current_pdf_page_num += 1
            self._update_pdf_buttons_state() # Disable buttons during load/process
            self.display_and_process_pdf_page(self.current_pdf_page_num)
        else:
            self.update_status("Already at the last page or not a PDF.")

    def process_all_pdf_pages(self):
        print(f"[DEBUG] process_all_pdf_pages called. current_file_type={self.current_file_type}, total_pdf_pages={self.total_pdf_pages}")
        if self.current_file_type != 'pdf' or not self.current_file_path:
            self.show_warning("No PDF Loaded", "Please open a PDF file first.")
            return
        if not self.current_pdf_doc or self.total_pdf_pages <= 0:
             self.show_warning("PDF Error", "No valid PDF document or pages loaded.")
             return

        self.update_status(f"Starting full PDF processing ({self.total_pdf_pages} pages)...")
        self.text_edit.clear()
        self.text_edit.setPlaceholderText(f"Extracting text from all {self.total_pdf_pages} pages...")
        self.extracted_text = ""
        self.copy_action.setEnabled(False)
        self.save_action.setEnabled(False)
        self.update_counts()
        self._update_pdf_buttons_state()
        self.run_task(process_entire_pdf_extract_text, self.handle_ocr_result, self.current_file_path)

    # (resizeEvent - Unchanged)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Trigger redraw/rescale slightly delayed
        QTimer.singleShot(50, self.display_image_preview)
        # Ensure splash overlay always covers the window (with margin)
        if hasattr(self, 'splash_overlay') and self.splash_overlay.isVisible():
            margin = min(80, self.width() // 4, self.height() // 4)
            self.splash_overlay.setGeometry(margin, margin, max(1, self.width()-2*margin), max(1, self.height()-2*margin))

    # --- Worker Thread Management ---
    def run_task(self, function, callback_slot, *args, **kwargs):
        if self.worker_thread and self.worker_thread.isRunning():
            self.show_warning("Busy", "Processing is already in progress."); return

        self.toolbar.setEnabled(False)
        self._update_pdf_buttons_state()

        self.worker = Worker(function, *args, **kwargs)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker.set_thread(self.worker_thread)

        # Connect signals
        self.worker.finished.connect(callback_slot)
        self.worker.finished.connect(self.on_task_finished) # Runs AFTER callback_slot
        self.worker.progress.connect(self.update_status)
        self.worker_thread.started.connect(self.worker.run)

        # Cleanup connections
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.cleanup)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: setattr(self, 'worker_thread', None))
        self.worker_thread.finished.connect(lambda: setattr(self, 'worker', None))
        # Add explicit button state update after worker cleanup
        self.worker_thread.finished.connect(self._update_pdf_buttons_state)

        self.worker_thread.start()

    def on_task_finished(self):
        # This runs after the callback_slot (handle_ocr_result)
        self.toolbar.setEnabled(True) # Re-enable the toolbar

        # Re-enable specific actions based on state
        self.copy_action.setEnabled(bool(self.extracted_text))
        self.save_action.setEnabled(bool(self.extracted_text))
        
        # Force update button states
        QTimer.singleShot(0, self._update_pdf_buttons_state)

        # Update status if not already set to a final message by the progress signal
        current_status = self.status_label.text()
        if not any(end in current_status.lower() for end in ["complete.", "finished.", "processed.", "failed.", "error", "cancelled"]):
             if self.current_file_type == 'pdf':
                 self.update_status(f"Ready. PDF Page {self.current_pdf_page_num + 1}/{self.total_pdf_pages} processed.")
             elif self.current_file_type == 'image':
                 self.update_status("Ready. Image processed.")
             else:
                 self.update_status("Ready.")

        self._force_toolbar_icons_update()

    def handle_ocr_result(self, result, error):
        # This slot receives the (text, error) tuple from the worker
        if error:
            self.show_error("Processing Error", f"Failed to extract text: {error}")
            self.text_edit.setPlaceholderText(f"Error extracting text: {error}\nPlease check the file or try again.")
            self.extracted_text = ""
            if self.current_file_path:
                 self.file_info_label.setText(f"| {Path(self.current_file_path).name} - Error")
            else:
                 self.file_info_label.setText(" | Error processing")
            self.update_status(f"Error: {error}")
        elif result is not None:
            self.extracted_text = result
            # Display result or placeholder if empty
            display_text = self.extracted_text if self.extracted_text else "[No text detected in the image/page]"
            self.text_edit.setText(display_text)
            # Determine appropriate status message
            status_msg = "Text extraction complete."
            # Check status for context (single page vs all pages)
            current_status_lower = self.status_label.text().lower()
            if self.current_file_type == 'pdf':
                if "processing pdf page" in current_status_lower: # Single page context
                     status_msg = f"PDF Page {self.current_pdf_page_num + 1}/{self.total_pdf_pages} processed."
                elif "full pdf processing" in current_status_lower: # All pages context
                     status_msg = f"Full PDF processing complete ({self.total_pdf_pages} pages)."
                # Else: keep generic "complete" message if context unclear
            elif self.current_file_type == 'image':
                status_msg = "Image processing complete."

            self.update_status(status_msg) # Use the more specific message
        else:
             # This case might indicate an unexpected issue in the worker
             self.show_error("Processing Error", "An unknown error occurred during processing (no result or error returned).")
             self.text_edit.setPlaceholderText("Unknown error during text extraction.")
             self.extracted_text = ""
             if self.current_file_path:
                 self.file_info_label.setText(f"| {Path(self.current_file_path).name} - Unknown Error")
             else:
                 self.file_info_label.setText(" | Unknown processing error")
             self.update_status("Error: Unknown processing failure.")

        self.update_counts()
        # Enable copy/save based on whether text was actually extracted
        self.copy_action.setEnabled(bool(self.extracted_text))
        self.save_action.setEnabled(bool(self.extracted_text))
        # Note: on_task_finished will re-enable toolbar and call _update_pdf_buttons_state

    # (reset_ui - logic unchanged, relies on _reset_pdf_state)
    def reset_ui(self):
        if self.worker_thread and self.worker_thread.isRunning():
             self.show_warning("Busy", "Cannot reset while processing is in progress."); return

        self._reset_pdf_state() # Close PDF doc, reset vars, hide PDF controls

        self.current_file_path = None
        # self.current_file_type handled by _reset_pdf_state
        self.extracted_text = ""
        self.current_pixmap = None

        self.image_label.clear()
        self.display_image_preview() # This will show the placeholder text

        self.text_edit.clear()
        self.text_edit.setPlaceholderText("Extracted text will appear here...")
        self.file_info_label.setText(" | No file selected")
        self.update_counts()
        self.copy_action.setEnabled(False)
        self.save_action.setEnabled(False)
        self.toolbar.setEnabled(True) # Ensure toolbar is enabled
        # self._update_pdf_buttons_state() is called by _reset_pdf_state
        self.update_status("Ready. Select or drop an image or PDF file.")

        # Force icon refresh after reset
        QTimer.singleShot(0, self._force_toolbar_icons_update)
        # Also ensure history toggle icon is correct
        QTimer.singleShot(0, self.update_history_toggle_appearance)

    # (update_counts, copy_text, save_text - unchanged)
    def update_counts(self):
        text = self.text_edit.toPlainText(); char_count = len(text)
        word_count = len(text.split()) if text else 0
        self.count_label.setText(f"Chars: {char_count} | Words: {word_count}")

    def copy_text(self):
        current_text = self.text_edit.toPlainText()
        if not current_text or current_text == "[No text detected in the image/page]": # Don't copy placeholder
             self.update_status("No text to copy.")
             return
        clipboard = QApplication.clipboard(); clipboard.setText(current_text)
        self.update_status("Text copied to clipboard.")

    def save_text(self):
        current_text = self.text_edit.toPlainText()
        if not current_text or current_text == "[No text detected in the image/page]":
            self.update_status("No text to save.")
            return

        # Suggest filename based on original file
        suggested_name = "extracted_text.txt" # Default
        if self.current_file_path:
            base_name = Path(self.current_file_path).stem
            if self.current_file_type == 'pdf':
                 # Check if 'full pdf processing complete' status msg was shown recently
                 status_lower = self.status_label.text().lower()
                 if "full pdf processing complete" in status_lower or self.total_pdf_pages == 1 :
                      suggested_name = f"{base_name}_ocr.txt"
                 elif self.current_pdf_page_num >= 0: # If a single page was processed (and not page 0 of many)
                      suggested_name = f"{base_name}_page{self.current_pdf_page_num + 1}_ocr.txt"
                 else: # Fallback if page number is invalid somehow
                      suggested_name = f"{base_name}_ocr.txt"
            else: # Image
                 suggested_name = f"{base_name}_ocr.txt"


        try:
            downloads_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        except:
            downloads_dir = str(Path.home())

        save_path, _ = QFileDialog.getSaveFileName(self,"Save Text File", os.path.join(downloads_dir, suggested_name),"Text Files (*.txt);;All Files (*)")
        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f: f.write(current_text)
                self.update_status(f"Text saved to: {Path(save_path).name}")
            except Exception as e:
                self.show_error("Save Error", f"Could not save file:\n{e}")
                self.update_status("Error saving file.")

    # (closeEvent - unchanged, _reset_pdf_state is called)
    def closeEvent(self, event):
        # Ensure any running worker is properly cleaned up (should be handled by signals)
        # if self.worker:
        #     self.worker.cleanup()
        if self.worker_thread and self.worker_thread.isRunning():
             print("Warning: Closing while worker thread is still running.")
             # Consider adding a prompt or preventing close here if needed

        self._reset_pdf_state() # Ensure PDF doc is closed before saving settings/exiting
        self._save_user_settings()
        super().closeEvent(event)

    # (dragEnterEvent, dropEvent - unchanged)
    def dragEnterEvent(self, event: QDragEnterEvent):
        if hasattr(self, 'image_label'):
            # Map global position to the label's coordinates
            label_pos = self.image_label.mapFromGlobal(event.position().toPoint())
            # Check if the mapped position is within the label's rectangle
            if self.image_label.rect().contains(label_pos):
                self.image_label.dragEnterEvent(event) # Pass event to label
                return # Event handled by label

        # Ignore event if not over the label
        event.ignore()


    def dropEvent(self, event: QDropEvent):
         # Only let the label handle drops that occur within its bounds
        if hasattr(self, 'image_label'):
            label_pos = self.image_label.mapFromGlobal(event.position().toPoint())
            if self.image_label.rect().contains(label_pos):
                 # Note: DropEvent itself doesn't usually change visual style back, dragLeave does.
                 # We let ImageDropLabel's dropEvent call the main window's handler.
                 # Need to make sure the main window's handler ignores if not appropriate.
                 pass # Allow label's default dropEvent to occur which should call our handler via window()

        # Ignore the drop event in the main window otherwise
        event.ignore()

    def show_with_splash(self, user_name="User"):
        self.show()  # Ensure the main window is shown
        # Create splash overlay if not already present
        if not hasattr(self, 'splash_overlay') or self.splash_overlay is None:
            self.splash_overlay = SplashOverlay(self, welcome_text=f"Welcome, {user_name}!")
        self.splash_overlay.setGeometry(self.rect())
        self.splash_overlay.raise_()
        self.splash_overlay.setVisible(True)
        self.splash_overlay.opacity_effect.setOpacity(0.0)

        # Fade in
        self._splash_fade_in = QPropertyAnimation(self.splash_overlay.opacity_effect, b"opacity")
        self._splash_fade_in.setDuration(900)
        self._splash_fade_in.setStartValue(0.0)
        self._splash_fade_in.setEndValue(1.0)
        self._splash_fade_in.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Pause
        self._splash_pause = QTimer(self)
        self._splash_pause.setSingleShot(True)

        # Fade out
        self._splash_fade_out = QPropertyAnimation(self.splash_overlay.opacity_effect, b"opacity")
        self._splash_fade_out.setDuration(900)
        self._splash_fade_out.setStartValue(1.0)
        self._splash_fade_out.setEndValue(0.0)
        self._splash_fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def start_fade_out():
            self._splash_fade_out.start()

        def show_main_content():
            self.splash_overlay.setVisible(False)
            self.show_and_fade_in_content()

        self._splash_fade_in.finished.connect(lambda: self._splash_pause.start(1200))
        self._splash_pause.timeout.connect(start_fade_out)
        self._splash_fade_out.finished.connect(show_main_content)

        self._splash_fade_in.start()


# --- SplashOverlay Widget ---
class SplashOverlay(QWidget):
    def __init__(self, parent, welcome_text="Welcome!"):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            background: rgba(20, 20, 20, 180);
        """)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAutoFillBackground(True)
        self.setVisible(False)

        # Opacity effect for fade in/out
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        # Store label as instance variable for dynamic font adjustment
        self.label = QLabel(welcome_text)
        self.label.setStyleSheet("""
            color: #fff;
            font-weight: 600;
            font-family: 'Segoe UI', 'Arial', sans-serif;
            background: transparent;
        """)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        self._update_label_font()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_label_font()

    def _update_label_font(self):
        # Dynamically adjust font size based on widget height (with min/max)
        h = max(1, self.height())
        w = max(1, self.width())
        # Use a percentage of the smaller dimension for font size
        base = min(w, h)
        font_size = max(18, min(72, int(base * 0.09)))  # 9% of min(w, h), clamp 18-72pt
        font = self.label.font()
        font.setPointSize(font_size)
        self.label.setFont(font)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setOrganizationName("MyCompany")
    app.setApplicationName("OCRExtractionSuite")
    tesseract_ok = False
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Found Tesseract version: {version}")
        tesseract_ok = True
    except pytesseract.TesseractNotFoundError:
        QMessageBox.critical(None, "Tesseract Not Found", "Tesseract OCR executable was not found. Please install Tesseract and ensure it's in your system's PATH. The application cannot run without it.")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not verify Tesseract version ({e}).")
        QMessageBox.warning(None, "Tesseract Warning", f"Could not verify Tesseract version: {e}. OCR functionality may be affected.")
        tesseract_ok = True
    if not fitz:
        print("PyMuPDF (fitz) not found. PDF processing will be disabled.")
    if tesseract_ok:
        try:
            user_name = getpass.getuser().capitalize()
        except Exception:
            user_name = "User"
        main_win = MainWindow()
        main_win.show_with_splash(user_name)
        sys.exit(app.exec())
    else:
        sys.exit(1)
