# 📄 Image to Text Extraction Using OCR with Python

## 📌 Project Overview

This project demonstrates how to extract text and structured information from an image using **Optical Character Recognition (OCR)**.

The application uses **Tesseract OCR** through the Python `pytesseract` library to recognize text from an input image. 
After extracting the complete text, **Regular Expressions (Regex)** are used to identify and retrieve specific fields from the OCR output.

The project provides a simple example of converting unstructured image-based information into structured data that can later be stored, analyzed or integrated into other applications.

---

## 🎯 Project Objective

The main objectives of this project are to:

* Load an image containing textual information.
* Display and inspect the input image.
* Convert image content into machine-readable text.
* Use **Tesseract OCR** for text recognition.
* Extract specific information using **Regular Expressions**.
* Demonstrate how OCR can automate manual data-entry processes.

---

## 🖼️ Sample Input

The sample image used in this project contains information such as:

```text
Name: Sample
Unique Policy Number: 12345
Amount: 100000
Start Date: 1/10/2019
End Date: 1/11/2019
Geo-Coordinates: 13.89,83.49
```

This represents a simple document containing policy-related information.

---

## 🔄 Project Workflow

The overall workflow of the project is:

```text
Input Image
     ↓
Load Image using PIL
     ↓
Display Image using Matplotlib
     ↓
Convert Image to RGB
     ↓
Tesseract OCR
     ↓
Extract Raw Text
     ↓
Regular Expression Processing
     ↓
Extract Required Fields
     ↓
Structured Information
```

---

# 🛠️ Technologies Used

| Technology                 | Purpose                              |
| -------------------------- | ------------------------------------ |
| Python                     | Core programming language            |
| Jupyter Notebook           | Development environment              |
| Tesseract OCR              | Optical Character Recognition engine |
| pytesseract                | Python interface for Tesseract       |
| Pillow (PIL)               | Image loading and processing         |
| Matplotlib                 | Image visualization                  |
| Regular Expressions (`re`) | Extracting specific fields           |

---

# 📂 Project Files

```text
Image-to-Text-OCR/
│
├── Image to text (OCR).ipynb
├── test.JPG
└── README.md
```

### `Image to text (OCR).ipynb`

Contains the complete OCR implementation including:

* Importing required libraries
* Loading the image
* Displaying the image
* Configuring Tesseract
* Extracting text
* Extracting individual fields using Regex

### `test.JPG`

Sample image used as input for OCR processing.

---

# 🚀 Project Implementation

## Step 1: Import Required Libraries

The project begins by importing the necessary Python libraries.

```python
import matplotlib.pyplot as plt
import PIL
import pytesseract
import re

%matplotlib inline
```

### Libraries Explained

**Matplotlib**

Used to display the input image directly inside the Jupyter Notebook.

**PIL / Pillow**

Used to open and manipulate the image before sending it to the OCR engine.

**pytesseract**

Provides a Python interface to the Tesseract OCR engine.

**re**

Python's Regular Expression module is used to search the OCR-generated text and extract specific information.

---

# 🖼️ Step 2: Load the Image

The image is loaded using Pillow.

```python
img = PIL.Image.open('test.JPG')
plt.imshow(img)
```

This allows us to verify the image visually before performing OCR.

---

# ⚙️ Step 3: Configure Tesseract OCR

The notebook configures the locally installed Tesseract OCR executable.

```python
pytesseract.pytesseract.tesseract_cmd = \
    'C:/Program Files/Tesseract-OCR/tesseract'

TESSDATA_PREFIX = 'C:/Program Files/Tesseract-OCR'
```

> The Tesseract installation path may be different depending on the operating system and installation directory.

---

# 🔍 Step 4: Convert Image to Text

The image is converted into RGB format and passed to Tesseract.

```python
text_data = pytesseract.image_to_string(
    img.convert('RGB'),
    lang='eng'
)
```

The OCR engine analyzes the image and converts the visible characters into machine-readable text.

---

# 📝 Step 5: Display Extracted Text

The OCR result is displayed using:

```python
print(text_data)
```

For the sample image, the extracted information represents fields such as:

```text
Name
Unique Policy Number
Amount
Start Date
End Date
Geo-Coordinates
```

Once the text has been extracted, it can be processed programmatically.

---

# 🧩 Step 6: Extract Specific Fields Using Regex

One of the important parts of this project is converting the OCR-generated text into more useful structured information.

Instead of manually searching through the extracted text, **Regular Expressions** are used.

---

## 👤 Extract Name

```python
m = re.search("Name: (\\w+)", text_data)
name = m[1]
name
```

The Regex searches for the text following:

```text
Name:
```

For the sample document:

```text
Sample
```

is extracted.

---

## 📅 Extract Start Date

```python
m = re.search("Start Date: (\\S+)", text_data)
start_date = m[1]
start_date
```

This searches for the value appearing immediately after:

```text
Start Date:
```

Example result:

```text
1/10/2019
```

---

## 📍 Extract Geo-Coordinates

```python
m = re.search("Geo-Coordinates: (\\S+)", text_data)
coordinates = m[1]
coordinates
```

This extracts the geographical coordinates from the OCR-generated text.

Example:

```text
13.89,83.49
```

---

# 📊 Example Structured Output

After OCR and Regex processing, the extracted information can conceptually be represented as:

| Field           | Extracted Value |
| --------------- | --------------- |
| Name            | Sample          |
| Start Date      | 1/10/2019       |
| Geo-Coordinates | 13.89,83.49     |

The notebook specifically demonstrates Regex extraction for these three fields.

The sample image also contains additional information such as **Unique Policy Number, Amount, and End Date**, which could be extracted in the same way by extending the Regex logic.

---

# 💡 Why Use OCR?

Many organizations still receive information through:

* Scanned documents
* Images
* Forms
* Receipts
* Invoices
* Insurance documents
* Identification documents
* PDF scans

These documents contain valuable information but cannot always be queried directly like database records.

OCR helps convert this information into machine-readable text.

---

# 🌎 Real-World Applications

This type of OCR workflow can be extended to applications such as:

### 🏦 Banking

Extract information from:

* Loan applications
* Checks
* Account documents
* Financial statements

### 🏥 Healthcare

Extract information from:

* Medical forms
* Patient documents
* Lab reports
* Scanned records

### 🛡️ Insurance

Extract fields such as:

* Policy number
* Customer name
* Coverage amount
* Start date
* End date

### 🧾 Invoice Processing

Extract:

* Invoice number
* Vendor
* Amount
* Date
* Tax information

### 📑 Document Digitization

Convert scanned paper documents into searchable digital information.

---

# 🧠 Key Concepts Demonstrated

Through this project, I worked with:

* Optical Character Recognition
* Image processing
* Text extraction
* Python image libraries
* Tesseract OCR
* Regex pattern matching
* Semi-structured data extraction
* Image-to-text conversion
* Basic document automation

---

# 📈 Potential Improvements

The current project provides a simple OCR implementation. It can be enhanced further by adding:

* Image preprocessing before OCR
* Grayscale conversion
* Noise reduction
* Thresholding
* Image resizing
* Automatic document detection
* Multiple-image processing
* PDF-to-text extraction
* Additional Regex patterns
* Data validation
* Exception handling
* Exporting extracted data to CSV/Excel
* Saving results into a database

For example, the remaining fields from the sample image could also be extracted:

```text
Unique Policy Number
Amount
End Date
```

This would allow the entire document to be converted into structured information.

---

# 🔮 Future Scope

A production-level version of this project could follow a workflow such as:

```text
Document/Image Upload
        ↓
Image Preprocessing
        ↓
OCR Engine
        ↓
Text Extraction
        ↓
Field Identification
        ↓
Data Validation
        ↓
Structured Dataset
        ↓
Database / API / Analytics
```

This could form the foundation for an automated **Document Intelligence System**.

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Image-to-Text-OCR
```

## 2. Install Python Dependencies

```bash
pip install matplotlib pillow pytesseract
```

## 3. Install Tesseract OCR

Tesseract OCR must also be installed on the local machine.

After installation, update the path in the notebook if necessary.

Example for Windows:

```python
pytesseract.pytesseract.tesseract_cmd = \
    'C:/Program Files/Tesseract-OCR/tesseract'
```

## 4. Add the Input Image

Place:

```text
test.JPG
```

in the same project directory as the notebook.

## 5. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Image to text (OCR).ipynb
```

and execute the cells sequentially.

---

# 🎓 What I Learned

This project helped me understand how OCR technology can bridge the gap between **image-based documents and structured digital data**.

I gained practical experience with:

* Reading images using Python
* Integrating Tesseract with Python
* Performing OCR
* Extracting raw textual information
* Searching OCR output using Regex
* Converting document information into structured fields
* Understanding how OCR can support document automation workflows

---

# 🏁 Conclusion

This project demonstrates a straightforward **Image-to-Text OCR pipeline using Python and Tesseract**.

The workflow converts an image into machine-readable text and then uses **Regular Expressions** to retrieve selected information such as the **Name, Start Date, and Geo-Coordinates**.

Although the implementation is intentionally simple, the same concept can be expanded into larger document-processing solutions for insurance, banking, healthcare, finance and other industries where information needs to be extracted from scanned documents.

---
