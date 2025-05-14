# TextLens: Image & PDF Text Extractor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/UI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Tesseract OCR](https://img.shields.io/badge/OCR-Tesseract-orange.svg)](https://github.com/tesseract-ocr/tesseract)

A powerful, user-friendly desktop application for extracting text from images and PDF documents using Tesseract OCR.

![TextLens Screenshot](https://raw.githubusercontent.com/Mxneeb/TextLens-IMG-PDF-Text-Extractor/main/screenshots/main.png)

## ✨ Features

- **Multi-Format Support**: Process PNG, JPG/JPEG, BMP, TIFF images and PDF documents
- **PDF Navigation**: Preview, navigate through pages, and selectively OCR pages
- **Enhanced OCR**: Advanced image preprocessing for improved text recognition accuracy
- **Modern UI**: Intuitive interface with multiple themes (light/dark) and customizable text display
- **Workflow Tools**: File history, drag & drop support, copy/save functions
- **Responsive Design**: Asynchronous processing with worker threads prevents UI freezing

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract) (with appropriate language data files)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mxneeb/TextLens-IMG-PDF-Text-Extractor.git
   cd TextLens-IMG-PDF-Text-Extractor
   ```

2. **Set up virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python gui.py
   ```

## 📦 Dependencies

- **PyQt6**: Modern GUI framework
- **Pytesseract**: Python wrapper for Tesseract OCR
- **OpenCV-Python**: Image processing library
- **PyMuPDF**: PDF handling
- **NumPy**: Numerical operations

## 🖼️ Screenshots

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/Mxneeb/TextLens-IMG-PDF-Text-Extractor/main/screenshots/light_theme.png" alt="Light Theme" width="400"/></td>
    <td><img src="https://raw.githubusercontent.com/Mxneeb/TextLens-IMG-PDF-Text-Extractor/main/screenshots/dark_theme.png" alt="Dark Theme" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>Light Theme</em></td>
    <td align="center"><em>Dark Theme</em></td>
  </tr>
</table>

## 🔧 Advanced Configuration

### Tesseract Installation

#### Windows
1. Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. During installation, make sure to:
   - Add Tesseract to PATH
   - Install language data files (at minimum, English)

#### macOS
```bash
brew install tesseract
brew install tesseract-lang  # for additional language support
```

#### Linux
```bash
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-eng  # for English language data
```

## 📂 Project Structure

```
TextLens-IMG-PDF-Text-Extractor/
├── gui.py                 # Main application GUI
├── source/
│   └── source.py          # OCR and image processing logic
├── icons/                 # UI icons
├── history/               # Created at runtime (user settings & history)
├── README.md
└── requirements.txt
```

## 🔮 Future Enhancements

- [ ] Language selection in GUI
- [ ] Advanced image preprocessing options
- [ ] Export to searchable PDF
- [ ] Batch processing
- [ ] Cross-platform binary releases

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
