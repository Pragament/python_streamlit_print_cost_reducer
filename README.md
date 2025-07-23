

-----


````markdown
# 🧾 Advanced PDF Printing Cost Estimator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

An interactive web application built with Streamlit to analyze PDF files, estimate printing costs, and provide tools to reduce costs by selectively removing colors.

This tool goes beyond simple page counting by analyzing the pixel data of each page to differentiate between black & white, low-color, and high-color pages, providing a more accurate cost estimate.

---

## ✨ Features

- **Detailed Cost Breakdown**: Get estimates based on print type (color/B&W), paper size (A4/A3), duplex printing, binding, and number of copies.
- **Intelligent Color Analysis**: Automatically detects the amount of color on each page to provide a tiered cost structure.
- **Advanced Color Removal**:
  - Use an interactive **color picker** to visually select specific colors to remove.
  - Manually input a list of hex codes for precise color removal.
  - Adjust the **sensitivity threshold** to control how closely a pixel must match to be removed.
- **Cost & Ink Comparison**: Instantly see a side-by-side comparison of the original vs. modified PDF, including total cost savings and percentage reduction in "ink".
- **PDF Regeneration**: Download a new, optimized PDF with the selected colors removed.
- **Actionable Suggestions**: Receive tips on how to further reduce costs.

---

## 📸 Screenshots

Here is a look at the application's interface and features.

<p align="center">
  <em>Main interface showing printing options and color removal tools.</em><br>
  <img src="./assets/1.png" width="700" alt="Application Screenshot 1">
</p>
<p align="center">
  <em>Cost and ink comparison between the original and modified PDF.</em><br>
  <img src="./assets/2.png" width="700" alt="Application Screenshot 2">
</p>

---

## 🚀 How to Run Locally

### 1. Prerequisites

- **Python**: Version 3.8 or higher.
- **Poppler**: `pdf2image` requires the Poppler utility.

  - **On macOS (using Homebrew):**
    ```bash
    brew install poppler
    ```
  - **On Ubuntu/Debian:**
    ```bash
    sudo apt-get update && sudo apt-get install -y poppler-utils
    ```
  - **On Windows:**
    Download and install Poppler, then add its `bin` folder to your system's PATH.

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-directory>
````

### 3\. Install Dependencies

It's recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required Python packages
pip install -r requirements.txt
```

You will need a `requirements.txt` file with the following content:

```txt
streamlit
pdf2image
Pillow
numpy
```

### 4\. Run the Streamlit App

```bash
streamlit run app.py
```

-----

## 🛠️ Technology Stack

  - **Framework**: [Streamlit](https://streamlit.io/)
  - **PDF Processing**: [pdf2image](https://github.com/Belval/pdf2image)
  - **Image Manipulation**: [Pillow (PIL)](https://www.google.com/search?q=https://python-pillow.org/)
  - **Numerical Operations**: [NumPy](https://numpy.org/)

-----

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

```
```