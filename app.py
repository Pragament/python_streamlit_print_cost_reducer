import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import io 
from pdf2image import convert_from_bytes

# images = convert_from_bytes(pdf_bytes, dpi=72)


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
def generate_pdf_from_images(images, num_images_per_page=1, orientation="Portrait", gap=10, margins=None, apply_margins_to_odd_pages=False):
    """Saves a list of PIL Images to a PDF in memory, arranging images per page and orientation.
    Supports custom margins for odd pages if specified."""
    if not images:
        return None

    # Standard page sizes in pixels (at 100 DPI)
    page_sizes = {
        "Portrait": {"A4": (595, 842), "A3": (842, 1191)},
        "Landscape": {"A4": (842, 595), "A3": (1191, 842)}
    }
    
    # Use A4 size for now (can be made configurable)
    page_size = page_sizes[orientation]["A4"]
    
    pages = []
    for i in range(0, len(images), num_images_per_page):
        imgs = images[i:i+num_images_per_page]
        page_index = i // num_images_per_page
        is_odd_page = (page_index % 2 == 0)  # 0-indexed, so page 1 is index 0
        
        # Create a blank page
        page = Image.new("RGB", page_size, (255, 255, 255))
        
        if num_images_per_page == 1:
            # Single image per page - center it
            img = imgs[0]
            
            # Default margins
            margin_top = 50
            margin_bottom = 50
            margin_left = 50
            margin_right = 50
            
            # Apply custom margins for odd pages if specified
            if margins and apply_margins_to_odd_pages and is_odd_page:
                margin_top = int(margins.get("top", 50))
                margin_bottom = int(margins.get("bottom", 50))
                margin_left = int(margins.get("left", 50))
                margin_right = int(margins.get("right", 50))
            
            # Calculate scaling to fit page with margins
            max_width = page_size[0] - margin_left - margin_right
            max_height = page_size[1] - margin_top - margin_bottom
            
            # Calculate scale to fit image within page bounds
            scale_w = max_width / img.size[0]
            scale_h = max_height / img.size[1]
            scale = min(scale_w, scale_h)
            
            new_width = int(img.size[0] * scale)
            new_height = int(img.size[1] * scale)
            img_resized = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Position the image according to margins
            x = margin_left + (max_width - new_width) // 2
            y = margin_top + (max_height - new_height) // 2
            page.paste(img_resized, (x, y))
            
        else:
            # Multiple images per page - arrange based on orientation
            if orientation == "Landscape":
                # Landscape: arrange images side by side (horizontal)
                grid_rows, grid_cols = 1, num_images_per_page
            else:
                # Portrait: arrange images in a grid (2 columns for better layout)
                grid_cols = 2
                grid_rows = int(np.ceil(num_images_per_page / grid_cols))
            
            # Default margins
            margin_top = 30
            margin_bottom = 30
            margin_left = 30
            margin_right = 30
            
            # Apply custom margins for odd pages if specified
            if margins and apply_margins_to_odd_pages and is_odd_page:
                margin_top = int(margins.get("top", 30))
                margin_bottom = int(margins.get("bottom", 30))
                margin_left = int(margins.get("left", 30))
                margin_right = int(margins.get("right", 30))
            
            # Add margins around the entire grid
            available_width = page_size[0] - margin_left - margin_right
            available_height = page_size[1] - margin_top - margin_bottom
            
            # Calculate cell size accounting for gaps in both directions
            total_gaps_w = (grid_cols - 1) * gap
            total_gaps_h = (grid_rows - 1) * gap
            cell_w = (available_width - total_gaps_w) // grid_cols
            cell_h = (available_height - total_gaps_h) // grid_rows
            
            for idx, img in enumerate(imgs):
                row = idx // grid_cols
                col = idx % grid_cols
                
                # Calculate position with gaps and margins
                x = margin_left + col * (cell_w + gap)
                y = margin_top + row * (cell_h + gap)
                
                # Resize image to fit cell while maintaining aspect ratio
                img_aspect = img.size[0] / img.size[1]
                cell_aspect = cell_w / cell_h
                
                if img_aspect > cell_aspect:
                    # Image is wider than cell - fit to width
                    new_width = cell_w
                    new_height = int(cell_w / img_aspect)
                    # Center vertically in cell
                    y_offset = (cell_h - new_height) // 2
                    y += y_offset
                else:
                    # Image is taller than cell - fit to height
                    new_height = cell_h
                    new_width = int(cell_h * img_aspect)
                    # Center horizontally in cell
                    x_offset = (cell_w - new_width) // 2
                    x += x_offset
                
                img_resized = img.resize((new_width, new_height), Image.LANCZOS)
                page.paste(img_resized, (x, y))
        
        pages.append(page)

    pdf_buffer = io.BytesIO()
    pages[0].save(
        pdf_buffer,
        "PDF",
        resolution=100.0,
        save_all=True,
        append_images=pages[1:]
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
            st.session_state.original_images = convert_from_bytes(
                pdf_bytes, 
                dpi=72
            )
    
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
        # --- NEW FIELDS ---
        num_images_per_page = st.number_input("Number of Images per page", min_value=1, value=1)
        page_orientation = st.selectbox("Page Orientation", ["Portrait", "Landscape"])
        gap_between_images = st.number_input("Gap between images (pixels)", min_value=0, value=10)
        # --- END NEW FIELDS ---

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
        
        # Margin settings for odd pages
        st.subheader("📏 Margin Settings for Odd Pages")
        st.info("Set custom margins for odd-numbered pages (1, 3, 5...) to improve image clarity. Use positive, zero, or negative values in pixels.")
        
        margin_col1, margin_col2 = st.columns(2)
        
        with margin_col1:
            margin_top = st.number_input("Top Margin (pixels)", value=50, step=1)
            margin_bottom = st.number_input("Bottom Margin (pixels)", value=50, step=1)
        
        with margin_col2:
            margin_left = st.number_input("Left Margin (pixels)", value=50, step=1)
            margin_right = st.number_input("Right Margin (pixels)", value=50, step=1)
            
        apply_margins = st.checkbox("Apply custom margins to odd pages", value=False)

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
                # Create margins dictionary
                margins = {
                    "top": margin_top,
                    "bottom": margin_bottom,
                    "left": margin_left,
                    "right": margin_right
                }
                
                st.session_state.new_pdf_bytes = generate_pdf_from_images(
                    modified_images, num_images_per_page, page_orientation, gap_between_images,
                    margins=margins, apply_margins_to_odd_pages=apply_margins
                )
                # --- NEW: Calculate percentage change in non-white pixels ---
                original_ink = sum(count_non_white_pixels(img) for img in images)
                modified_ink = sum(count_non_white_pixels(img) for img in modified_images)
                if original_ink > 0:
                    ink_reduction_percent = ((modified_ink - original_ink) / original_ink) * 100
                # --- END OF NEW CALCULATION ---
        else:
            new_costs = original_costs
            # Create margins dictionary
            margins = {
                "top": margin_top,
                "bottom": margin_bottom,
                "left": margin_left,
                "right": margin_right
            }
            
            st.session_state.new_pdf_bytes = generate_pdf_from_images(
                images, num_images_per_page, page_orientation, gap_between_images,
                margins=margins, apply_margins_to_odd_pages=apply_margins
            )
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
            st.metric("Ink Reduction (Non-White Pixels)", f"{ink_reduction_percent:.2f}%")
            if st.session_state.get('new_pdf_bytes'):
                # --- Enhanced PDF Preview ---
                st.subheader("📄 PDF Preview")
                
                try:
                    from pdf2image import convert_from_bytes
                    # Convert only first few pages for faster preview (max 3 pages)
                    max_preview_pages = min(3, len(images) // num_images_per_page + 1)
                    preview_images = convert_from_bytes(
                        st.session_state.new_pdf_bytes, 
                        dpi=150,
                        first_page=1,
                        last_page=max_preview_pages
                    )
                    
                    # Show page navigation
                    total_pages = len(images) // num_images_per_page + (1 if len(images) % num_images_per_page > 0 else 0)
                    preview_pages = len(preview_images)
                    
                    if total_pages > 1:
                        page_num = st.selectbox(
                            f"Select page to preview (Showing first {preview_pages} of {total_pages} pages)",
                            range(1, preview_pages + 1),
                            format_func=lambda x: f"Page {x}"
                        ) - 1
                    else:
                        page_num = 0
                    
                    # Display selected page
                    st.image(
                        preview_images[page_num], 
                        caption=f"Page {page_num + 1} of {total_pages} (Preview)",
                        use_column_width=True
                    )
                    
                    # Show page info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Pages", total_pages)
                    with col2:
                        st.metric("Images per Page", num_images_per_page)
                    with col3:
                        st.metric("Orientation", page_orientation)
                        
                    if total_pages > preview_pages:
                        st.info(f"📄 Showing preview of first {preview_pages} pages. Download the PDF to see all {total_pages} pages.")
                        
                except Exception as e:
                    st.error(f"Preview not available: {e}")
                    st.info("You can still download the PDF below.")
                
                # Download button
                st.download_button(
                    label="📥 Download Generated PDF",
                    data=st.session_state.new_pdf_bytes,
                    file_name=f"generated_document_{page_orientation.lower()}_{num_images_per_page}images.pdf",
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
