import os
import sys
import streamlit as st
import time
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path

# Add the parent directory to the path to import core module
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
from core.car_search_core import (
    initialize_databases,
    search_using_text_with_fts,
    search_cars_by_image,
    is_valid_image_path,
    load_image_from_url_or_path
)

# Set page configuration
st.set_page_config(
    page_title="🚗 Car AI Search Engine",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI (shortened for brevity)
st.markdown("""
<style>
    /* Main Styles */
    body {
        background-color: #1a1a1a;
        color: #ffffff;
    }

    /* Main Header Styles */
    .main-header {
        background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
        border: 1px solid #4d4d4d;
        animation: fadeIn 0.8s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0.7; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        margin-bottom: 1.2rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        text-align: center;
        font-family: 'Arial', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: 1px;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #e0e0e0;
        margin-bottom: 1rem;
        line-height: 1.5;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .tagline {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 8px 16px;
        border-radius: 8px;
        display: inline-block;
        margin-top: 10px;
        font-weight: 500;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Card Styles */
    .card {
        background: linear-gradient(135deg, #2d2d2d 0%, #333333 100%);
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #4d4d4d;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .card:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 15px 15px 0 0;
    }
    
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border-color: #5d5d5d;
    }

    /* Search Box Styles */
    .search-box {
        background: linear-gradient(135deg, #2d2d2d 0%, #333333 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #4d4d4d;
        margin: 30px 0;
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .search-box:hover {
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        border-color: #5d5d5d;
    }
    
    /* Input field styling */
    [data-testid="stTextInput"] input {
        background-color: #3d3d3d !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid #4d4d4d !important;
        padding: 12px 15px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stTextInput"] input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    [data-testid="stTextInput"] input::placeholder {
        color: #aaaaaa !important;
    }

    /* Image Container */
    .image-container {
        background: #2d2d2d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #3d3d3d;
    }
    
    /* Fix for View Full Details text color */
    [data-testid="stExpander"] {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    [data-testid="stExpander"] div p {
        color: white !important;
    }
    
    /* Fix for car details background */
    [data-testid="stExpander"] div div[style*="background"] {
        background-color: #3d3d3d !important;
        color: white !important;
    }
    
    /* Fix for all text inside View Full Details */
    [data-testid="stExpanderContent"] * {
        color: white !important;
    }
    
    /* Fix for car info section */
    div[style*="background: #f8f9fa"] {
        background-color: #3d3d3d !important;
        color: white !important;
    }
    
    div[style*="background: #f8f9fa"] * {
        color: white !important;
    }
    
    /* Fix View Full Details button */
    .streamlit-expanderHeader {
        background-color: #3d3d3d !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    /* Fix View Full Details arrow icon */
    .streamlit-expanderHeader svg {
        fill: white !important;
    }
    
    /* Fix for the price information shown in View Full Details */
    [data-testid="stExpander"] h2, 
    [data-testid="stExpander"] h3, 
    [data-testid="stExpander"] h4 {
        color: white !important;
        background-color: #1E3A8A !important;
        padding: 10px !important;
        border-radius: 8px !important;
        margin-top: 10px !important;
    }
    
    /* Improved Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #2d2d2d;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #3d3d3d !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Improved Button Styling */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Primary Button (Search) */
    .stButton button[data-testid="baseButton-primary"] {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%) !important;
    }
    
    /* Example Buttons */
    .stButton button[kind="secondary"] {
        background-color: #3d3d3d !important;
        color: white !important;
    }
    
    .stButton button[kind="secondary"]:hover {
        background-color: #4d4d4d !important;
    }
</style>
""", unsafe_allow_html=True)

# Create database connections and models with caching to avoid rerunning on each interaction
@st.cache_resource(show_spinner=False)
def get_databases():
    """Initialize database connections with a progress indicator"""
    with st.spinner("🚀 Initializing AI Search Engine..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        databases = initialize_databases()
        st.success("✨ AI Search Engine Ready!")
        time.sleep(1)
        st.empty()
        return databases

def display_car_results(results, is_image_search=False):
    """
    Display car search results in a grid layout with enhanced UI.
    
    Args:
        results: Search results (list of dicts or LanceDB objects)
        is_image_search: Whether the results are from image search
    """
    if not results:
        st.warning("😔 No results found. Try adjusting your search criteria!")
        return

    # Display result count with animation
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h2 style="color: #1E3A8A;">🎉 Found {} Matching Cars</h2>
        </div>
    """.format(len(results)), unsafe_allow_html=True)

    # Create a progress bar for loading results
    progress_bar = st.progress(0)
    
    cols = st.columns(3)
    col_index = 0

    for idx, result in enumerate(results):
        # Update progress bar
        progress = int((idx + 1) / len(results) * 100)
        progress_bar.progress(progress)

        with cols[col_index]:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Extract result data based on search type
                if is_image_search:
                    label = result.label
                    car_info = result.car_info
                    image_uri = result.image_uri
                    image_urls = [image_uri] if image_uri else []
                else:
                    label = result["label"]
                    car_info = result["car_info"]
                    image_urls = result.get("image_urls", [])
                    if isinstance(image_urls, (np.ndarray, pd.Series)):
                        image_urls = image_urls.tolist()
                    elif not isinstance(image_urls, list):
                        image_urls = [image_urls] if image_urls else []

                # Enhanced car information display
                st.markdown(f"""
                    <div style="background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); 
                         color: white; padding: 12px 15px; border-radius: 10px; margin-bottom: 15px;
                         box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
                        <h3 style="margin: 0; font-size: 22px; font-weight: 600;">{label}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display car info with better formatting
                if len(str(car_info)) > 200:
                    short_info = str(car_info)[:200] + "..."
                    with st.expander("📋 View Full Details"):
                        st.markdown(f"""
                            <div style="background-color: #3d3d3d; color: white; padding: 15px; 
                                border-radius: 10px; line-height: 1.6; font-size: 16px;
                                box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1);">
                                {car_info}
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="color: #e0e0e0; margin-bottom: 15px; line-height: 1.5; 
                            padding: 5px; font-size: 15px;">
                            {short_info}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="background-color: #3d3d3d; color: white; padding: 15px; 
                            border-radius: 10px; line-height: 1.6; font-size: 16px;
                            box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.1);">
                            {car_info}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced image display
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                
                valid_image_urls = [url for url in image_urls if is_valid_image_path(url)]
                
                if valid_image_urls:
                    first_image = load_image_from_url_or_path(valid_image_urls[0])
                    if first_image:
                        st.image(first_image)
                    else:
                        st.info("🖼️ Image could not be loaded")
                    
                    if len(valid_image_urls) > 1:
                        with st.expander("🖼️ View More Images"):
                            for url in valid_image_urls[1:]:
                                additional_image = load_image_from_url_or_path(url)
                                if additional_image:
                                    st.image(additional_image)
                else:
                    st.info("📷 No images available for this car")
                    
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        col_index = (col_index + 1) % 3

    # Remove progress bar after loading
    progress_bar.empty()

def validate_image(image_file):
    """
    Validate the uploaded image file.
    
    Args:
        image_file: File object from file uploader
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to open the image with PIL to verify it's a valid image
        image = Image.open(image_file)
        image.verify()  # Verify it's actually an image
        image_file.seek(0)  # Reset file pointer
        
        # Check if format is supported
        if image.format.lower() not in ['jpeg', 'jpg', 'png']:
            return False, f"Unsupported image format: {image.format}. Please upload JPEG or PNG images only."
        
        # Check file size (limit to 5MB)
        if image_file.size > 5 * 1024 * 1024:  # 5MB in bytes
            return False, "Image file is too large. Please upload images smaller than 5MB."
            
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def main():
    """Main application function"""
    # App header with modern design
    st.markdown("""
        <div class="main-header">
            <div class="main-title">
                <span style="font-size: 42px; margin-right: 8px;">🚗</span>
                <span>Car AI Search Engine</span>
            </div>
            <div class="main-subtitle">
                Discover your perfect car using advanced AI technology - 
                Search by text or find similar cars by image
            </div>
            <div class="tagline">
                Built using LanceDB multimodal vectordb 🚀
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'temp_image_path' not in st.session_state:
        st.session_state.temp_image_path = None
    
    # Initialize databases
    text_table, image_table, CarInfo, Images = get_databases()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        # Use a more professional car-related image
        st.image("https://img.freepik.com/free-vector/modern-blue-urban-adventure-suv-vehicle-illustration_1344-205.jpg", 
                caption="AI-Powered Car Search")
        st.markdown("### About")
        st.info("""
        🎯 **Key Features:**
        * Advanced AI-powered search
        * Image similarity matching
        * Detailed car specifications
        * Real-time results
        
        Start your journey to finding the perfect car! 
        Developed by @Akash Desai using LanceDB multimodal vectordb🚀
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # Create tabs for different search methods
    tab1, tab2 = st.tabs(["🔤 Text Search", "📷 Image Search"])
    
    # Text Search Tab
    with tab1:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        
        # Search input with examples
        query = st.text_input("🔍 Search for a car:", 
                            placeholder="E.g., 'luxury 7 seater car' or 'tata motors car'")
        
        # Example chips for quick searches
        examples_col1, examples_col2, examples_col3, examples_col4 = st.columns(4)
        with examples_col1:
            if st.button("7 Seater car"):
                query = "7 Seater car"
        with examples_col2:
            if st.button("Tata Motors car"):
                query = "Tata Motors car"
        with examples_col3:
            if st.button("5 lakh budget car"):
                query = "5 lakh budget car"
        with examples_col4:
            if st.button("25.0 kmpl mileage car"):
                query = "25.0 kmpl mileage car"
                
        # Search button and reset button
        search_col, reset_col = st.columns([3, 1])
        with search_col:
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        with reset_col:
            if st.button("🔄 Reset", use_container_width=True):
                query = ""
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Perform search when button clicked or query entered
        if (search_clicked or query) and query.strip():
            with st.spinner("Searching for cars..."):
                results = search_using_text_with_fts(text_table, query)
            
            if results:
                st.markdown(f'<p class="result-count">🎉 Found {len(results)} cars matching "{query}"</p>', unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                display_car_results(results)
            else:
                st.warning("😔 No cars found matching your search. Try different keywords!")
    
    # Image Search Tab
    with tab2:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        st.markdown("""
            ### Find similar cars by uploading an image
            
            **Supported formats:** JPEG, PNG
            **Maximum size:** 5MB
        """)
        
        # Image upload section with error handling
        uploaded_file = st.file_uploader("📁 Upload a car image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Validate the uploaded image
            is_valid, error_message = validate_image(uploaded_file)
            
            if not is_valid:
                st.error(f"⚠️ {error_message}")
                uploaded_file = None
            else:
                try:
                    # Get the base directory for temp files
                    base_dir = Path(__file__).parent.parent.parent
                    temp_dir = base_dir / "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Save the uploaded file to a temporary location
                    image_path = temp_dir / "temp_image.jpg"
                    
                    # Convert image to JPEG format for consistency
                    img = Image.open(uploaded_file)
                    if img.mode in ('RGBA', 'LA'):
                        # Remove alpha channel if present
                        bg = Image.new('RGB', img.size, (255, 255, 255))
                        bg.paste(img, mask=img.split()[-1])
                        img = bg
                    
                    # Save as JPEG with error handling
                    img.convert('RGB').save(image_path, 'JPEG', quality=85)
                    
                    st.session_state.temp_image_path = str(image_path)
                    
                    # Show the uploaded image
                    st.markdown("### Your Uploaded Image")
                    st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
                    st.image(img, caption="Uploaded Car Image")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"⚠️ Error processing image: {str(e)}")
                    st.session_state.temp_image_path = None
                    uploaded_file = None
        
        # Create columns for search and reset buttons
        img_search_col, img_reset_col = st.columns([3, 1])
        
        with img_search_col:
            img_search_clicked = st.button(
                "🔍 Find Similar Cars",
                type="primary",
                use_container_width=True,
                key="img_search",
                disabled=not uploaded_file  # Disable if no valid image
            )
        with img_reset_col:
            if st.button("🔄 Reset", use_container_width=True, key="img_reset"):
                st.session_state.temp_image_path = None
                uploaded_file = None
                # Clean up temporary file if it exists
                temp_image_path = Path(__file__).parent.parent.parent / "temp" / "temp_image.jpg"
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
                st.rerun()
        
        # Perform image search
        if img_search_clicked and st.session_state.temp_image_path:
            with st.spinner("Searching for similar cars..."):
                # Add a small delay to make the spinner visible
                time.sleep(0.5)
                results = search_cars_by_image(image_table, Images, st.session_state.temp_image_path)
            
            if results:
                st.markdown(f'<p class="result-count">🎉 Found {len(results)} similar cars</p>', unsafe_allow_html=True)
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                display_car_results(results, is_image_search=True)
            else:
                st.warning("😔 No similar cars found. Try uploading a different image!")

# Run the application
if __name__ == "__main__":
    main() 