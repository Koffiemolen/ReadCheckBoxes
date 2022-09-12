# ReadCheckBoxes

Python script for processing scanned files in pdf format and detecting checkboxes with openCV

Needed modueles:
pip install fitz
pip install PyMuPDF
pip install opencv-python numpy Pillow

Script order:
- First extract png from pdf files, if a pdf file contains multiple pages each page gets exported to seperate file
- Iterate over each png file to find checkboxes

# Todo:
- Find only the 3 checkboxes
- Reduce false positives
- Determine if a checkbox is checked
- Determine which checkbox is checked
