Here's your updated and **nicely formatted** `README.md` file for the **Advanced PDF Printing Cost Estimator**, with proper Markdown structure and working image visibility using relative paths (if you're using GitHub or a local Markdown renderer):

---

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

### 🔧 Main Interface
<p align="center">
  <img src="assets/1.png" width="700" alt="Application Screenshot 1">
  <br><em>Main interface showing printing options and color removal tools.</em>
</p>

### 📊 Cost & Ink Comparison
<p align="center">
  <img src="assets/2.png" width="700" alt="Application Screenshot 2">
  <br><em>Cost and ink comparison between the original and modified PDF.</em>
</p>

---

## 🚀 How to Run Locally

### 1. Prerequisites

- **Python**: Version 3.8 or higher.
- **Poppler**: Required by `pdf2image`.

  - **macOS (Homebrew)**:
    ```bash
    brew install poppler
    ```

  - **Ubuntu/Debian**:
    ```bash
    sudo apt-get update && sudo apt-get install -y poppler-utils
    ```

  - **Windows**:
    - Download Poppler from: [http://blog.alivate.com.au/poppler-windows/](http://blog.alivate.com.au/poppler-windows/)
    - Extract and add the `bin` folder to your system PATH.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/python_streamlit_print_cost_reducer.git
cd python_streamlit_print_cost_reducer
````

### 3. Install Dependencies

Use a virtual environment (recommended):

```bash
# Create and activate a virtual environment
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Sample `requirements.txt`

```
streamlit
pdf2image
Pillow
numpy
```

### 4. Run the App

```bash
streamlit run app.py
```

---

## 🛠️ Technology Stack

* **Framework**: [Streamlit](https://streamlit.io/)
* **PDF to Image Conversion**: [pdf2image](https://github.com/Belval/pdf2image)
* **Image Processing**: [Pillow](https://python-pillow.org/)
* **Numerical Computation**: [NumPy](https://numpy.org/)

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

```

---

### ✅ Notes:
- Ensure `assets/1.png` and `assets/2.png` exist in the correct path.
- Update the repository URL in the clone step.
- This version is styled for GitHub README or any Markdown viewer.

Let me know if you want this exported as a `.md` file!
```
