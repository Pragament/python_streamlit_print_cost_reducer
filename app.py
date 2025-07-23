import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import io

# --- Helper function to convert hex to RGB ---
def hex_to_rgb(hex_code):
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_code = hex_code.lstrip('#')
    try:
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return None

# --- Function to remove specified colors from an image ---
def remove_colors_from_image(image, colors_to_remove, threshold):
    """
    Removes specified colors from a PIL image, replacing them with white.
    """
    if not colors_to_remove:
        return image

    arr = np.array(image.convert("RGB"), dtype=int)
    final_mask = np.zeros(arr.shape[:2], dtype=bool)

    for color in colors_to_remove:
        distances = np.linalg.norm(arr - color, axis=2)
        current_mask = distances < threshold
        final_mask |= current_mask

    arr[final_mask] = [255, 255, 255]
    
    return Image.fromarray(arr.astype('uint8'), 'RGB')

# --- Function to generate a new PDF from modified images ---
def generate_pdf_from_images(images):
    """Saves a list of PIL Images to a PDF in memory."""
    if not images:
        return None
    
    pdf_buffer = io.BytesIO()
    images[0].save(
        pdf_buffer,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=images[1:]
    )
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# --- Functions for analysis ---
def count_color_pixels(image):
    """Counts pixels that are not grayscale."""
    arr = np.array(image.convert("RGB"))
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray_mask = (np.abs(r.astype(int) - g.astype(int)) < 15) & \
                (np.abs(r.astype(int) - b.astype(int)) < 15) & \
                (np.abs(g.astype(int) - b.astype(int)) < 15)
    non_gray_pixels = np.size(gray_mask) - np.count_nonzero(gray_mask)
    return non_gray_pixels

# --- NEW: Function to count non-white pixels as a proxy for ink usage ---
def count_non_white_pixels(image):
    """Counts every pixel that is not pure white (255, 255, 255)."""
    arr = np.array(image.convert("RGB"))
    # A mask where True means the pixel is white
    white_mask = np.all(arr == [255, 255, 255], axis=2)
    # Total pixels minus the count of white pixels
    return arr.shape[0] * arr.shape[1] - np.count_nonzero(white_mask)

def analyze_pdf_ink_usage(images):
    """Analyzes images for color pixels to estimate cost."""
    page_data = []
    for i, img in enumerate(images):
        color_pixels = count_color_pixels(img)
        page_data.append({"page": i + 1, "color_pixels": color_pixels})
    return page_data

# --- Functions for cost calculation and suggestions ---
def estimate_page_cost(color_pixels, is_color, paper_multiplier):
    if not is_color:
        return 1 * paper_multiplier
    if color_pixels < 1000:
        return 1 * paper_multiplier
    elif color_pixels < 5000:
        return 2 * paper_multiplier
    else:
        return 5 * paper_multiplier

def calculate_total_cost(page_data, copies, duplex, binding_type, is_color, paper_size):
    binding_costs = {"None": 0, "Spiral": 20, "Thermal": 40}
    paper_multiplier = {"A4": 1.0, "A3": 1.5}[paper_size]
    paper_count = len(page_data)
    
    if duplex:
        paper_used = (paper_count + 1) // 2 * copies
    else:
        paper_used = paper_count * copies

    page_costs = [estimate_page_cost(p["color_pixels"], is_color, paper_multiplier) for p in page_data]
    total_page_cost = sum(page_costs) * copies
    total_binding_cost = binding_costs[binding_type] * copies
    
    return {
        "Total Pages": paper_count,
        "Total Page Cost": round(total_page_cost, 2),
        "Paper Used": paper_used,
        "Binding Cost": total_binding_cost,
        "Total Cost": round(total_page_cost + total_binding_cost, 2),
    }

def suggest_savings(page_data, is_color, duplex, paper_size, copies):
    suggestions = []
    total_pages = len(page_data)
    color_page_count = sum(1 for p in page_data if p["color_pixels"] > 1000)
    
    full_color_cost = sum(estimate_page_cost(p["color_pixels"], True, 1.0) for p in page_data) * copies
    bw_cost = sum(estimate_page_cost(p["color_pixels"], False, 1.0) for p in page_data) * copies
    
    if is_color and color_page_count > 0:
        suggestions.append(f"Printing in B/W could save approx. ₹{full_color_cost - bw_cost:.2f}.")
    if not duplex:
        paper_saved = total_pages * copies - ((total_pages + 1) // 2) * copies
        suggestions.append(f"Use duplex printing to save {paper_saved} sheets of paper.")
    if paper_size == "A3":
        suggestions.append("Switch to A4 paper to reduce paper cost by approx. 33% per page.")
    return suggestions

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("🧾 Advanced PDF Printing Cost Estimator")

# Use session state to store data across reruns
if 'original_images' not in st.session_state:
    st.session_state.original_images = None
if 'hex_codes_input' not in st.session_state:
    st.session_state.hex_codes_input = ""

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    if st.session_state.original_images is None:
        with st.spinner("Analyzing PDF... This may take a moment."):
            pdf_bytes = uploaded_file.read()
            st.session_state.original_images = convert_from_bytes(pdf_bytes, dpi=72)
    
    images = st.session_state.original_images
    original_page_data = analyze_pdf_ink_usage(images)
    
    st.success(f"Analyzed {len(original_page_data)} pages. You can now set printing options and calculate the cost.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚙️ Printing Options")
        color_option = st.radio("Print Type", ["Color", "Black & White"])
        is_color = color_option == "Color"
        copies = st.number_input("Number of Copies", min_value=1, value=1)
        duplex = st.checkbox("Double-sided Printing (Duplex)", value=True)
        paper_size = st.selectbox("Paper Size", ["A4", "A3"])
        binding = st.selectbox("Binding Type", ["None", "Spiral", "Thermal"])

    with col2:
        st.subheader("🎨 Advanced Color Removal")
        st.info("Use the picker to add colors, or type/paste hex codes directly into the text box below.")
        
        picker_col, button_col = st.columns([1, 3])
        
        with picker_col:
            selected_color = st.color_picker("Pick a color")
        
        with button_col:
            st.write("")
            if st.button("➕ Add Color to List"):
                st.session_state.hex_codes_input += f"{selected_color} "

        st.session_state.hex_codes_input = st.text_area(
            "Colors to Remove (one hex code per space)",
            value=st.session_state.hex_codes_input,
            height=100
        )
        
        threshold = st.slider("Color Matching Sensitivity (Threshold)", 0, 100, 30)

    st.markdown("---")

    if st.button("Calculate Cost & Generate New PDF", type="primary"):
        hex_codes_input = st.session_state.hex_codes_input
        original_costs = calculate_total_cost(original_page_data, copies, duplex, binding, is_color, paper_size)
        
        target_rgbs = [hex_to_rgb(code) for code in hex_codes_input.split()]
        target_rgbs = [rgb for rgb in target_rgbs if rgb is not None]
        
        ink_reduction_percent = 0
        modified_images = images # Default to original images

        if target_rgbs:
            with st.spinner("Removing colors and recalculating..."):
                modified_images = [remove_colors_from_image(img, target_rgbs, threshold) for img in images]
                modified_page_data = analyze_pdf_ink_usage(modified_images)
                new_costs = calculate_total_cost(modified_page_data, copies, duplex, binding, is_color, paper_size)
                st.session_state.new_pdf_bytes = generate_pdf_from_images(modified_images)

                # --- NEW: Calculate percentage change in non-white pixels ---
                original_ink = sum(count_non_white_pixels(img) for img in images)
                modified_ink = sum(count_non_white_pixels(img) for img in modified_images)
                
                if original_ink > 0:
                    ink_reduction_percent = ((modified_ink - original_ink) / original_ink) * 100
                # --- END OF NEW CALCULATION ---
        else:
            new_costs = original_costs
            st.session_state.new_pdf_bytes = None

        st.subheader("💰 Cost & Ink Comparison")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("#### Original PDF")
            st.metric("Total Cost", f"₹{original_costs['Total Cost']:.2f}")
            with st.expander("Original Cost Breakdown"):
                st.write(f"Total Pages: {original_costs['Total Pages']}")
                st.write(f"Paper Used: {original_costs['Paper Used']} sheets")
                st.write(f"Binding Cost: ₹{original_costs['Binding Cost']:.2f}")
                st.write(f"Total Page Cost: ₹{original_costs['Total Page Cost']:.2f}")

        with res_col2:
            st.markdown("#### Modified PDF")
            st.metric("New Total Cost", f"₹{new_costs['Total Cost']:.2f}", delta=f"₹{new_costs['Total Cost'] - original_costs['Total Cost']:.2f}")
            # --- NEW: Display the ink reduction percentage ---
            st.metric("Ink Reduction (Non-White Pixels)", f"{ink_reduction_percent:.2f}%")
            # --- END OF NEW DISPLAY ---
            
            if st.session_state.get('new_pdf_bytes'):
                st.download_button(
                    # Download Modified PDF
                    label="📥 Download Modified PDF",
                    data=st.session_state.new_pdf_bytes,
                    file_name="modified_document.pdf",
                    mime="application/pdf"
                )

        st.markdown("---")
        st.subheader("💡 General Savings Suggestions")
        for s in suggest_savings(original_page_data, is_color, duplex, paper_size, copies):
            st.info(s)

if st.sidebar.button("Reset and Start Over"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()