#!/bin/bash
echo "Installing Tesseract OCR..."
apt-get update && apt-get install -y tesseract-ocr poppler-utils
echo "Tesseract installation complete!"
which tesseract
tesseract --version
