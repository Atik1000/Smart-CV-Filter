"""
Smart CV Filter - Streamlit Application
Main UI for CV analysis against Job Descriptions using RAG + LLM
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extract_text import extract_text
from utils.analyze_cv import CVAnalyzer

# Import LLM provider info
try:
    from utils.llm_provider import LLMProvider
    LLM_PROVIDER_AVAILABLE = True
except ImportError:
    LLM_PROVIDER_AVAILABLE = False

# Optional RAG support
try:
    from utils.embedding_db import CVEmbeddingDB
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    print(f"[INFO] ChromaDB not available - RAG features disabled (Reason: {type(e).__name__})")


# Page configuration
st.set_page_config(
    page_title="Smart CV Filter - AI-Powered CV Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# Custom CSS - Enhanced UI
st.markdown("""
    <style>
    /* Icon styling */
    .fa-icon {
        margin-right: 0.5rem;
        color: inherit;
    }
    
    .header-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
        vertical-align: middle;
    }
    
    /* Professional metric card styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Enhanced skill badges */
    .skill-badge {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        margin: 0.4rem;
        display: inline-block;
        font-weight: 600;
        color: #2e7d32;
        border: 2px solid #66bb6a;
        box-shadow: 0 2px 6px rgba(76, 175, 80, 0.2);
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    
    .skill-badge:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
    }
    
    .skill-badge i {
        margin-right: 0.4rem;
        color: #4caf50;
    }
    
    .missing-badge {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        margin: 0.4rem;
        display: inline-block;
        font-weight: 600;
        color: #c62828;
        border: 2px solid #ef5350;
        box-shadow: 0 2px 6px rgba(244, 67, 54, 0.2);
        font-size: 0.9rem;
    }
    
    .missing-badge i {
        margin-right: 0.4rem;
        color: #f44336;
    }
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #1f77b4 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .info-box h3 {
        margin: 0;
        color: white;
    }
    
    /* Score Box */
    .score-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .score-box:hover {
        transform: translateY(-5px);
    }
    
    .score-high {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 3px solid #28a745;
    }
    
    .score-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 3px solid #ffc107;
    }
    
    .score-low {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 3px solid #dc3545;
    }
    
    /* Badges */
    .skill-badge {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        display: inline-block;
        font-weight: 600;
        color: #1565c0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .skill-badge:hover {
        transform: scale(1.05);
    }
    
    .missing-badge {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        display: inline-block;
        font-weight: 600;
        color: #c62828;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Upload Section */
    .upload-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Section Headers */
    .section-header {
        color: #1f77b4;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None


def render_score_box(score: float):
    """Render score with color coding."""
    if score >= 70:
        css_class = "score-high"
        icon = '<i class="fas fa-trophy" style="color: #28a745;"></i>'
        label = "Excellent Match"
    elif score >= 50:
        css_class = "score-medium"
        icon = '<i class="fas fa-thumbs-up" style="color: #ffc107;"></i>'
        label = "Good Match"
    else:
        css_class = "score-low"
        icon = '<i class="fas fa-exclamation-triangle" style="color: #dc3545;"></i>'
        label = "Needs Improvement"
    
    st.markdown(f"""
        <div class="score-box {css_class}">
            <h2>{icon} {score}%</h2>
            <p style="margin: 0; font-size: 1.2rem;">{label}</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application logic."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header"><i class="fas fa-crosshairs header-icon"></i>Smart CV Filter</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent CV Analysis - Match CVs with Job Descriptions Instantly!</p>', 
                unsafe_allow_html=True)
    
    # Welcome Message
    st.markdown("""
        <div class="info-box">
            <h3><i class="fas fa-hand-wave fa-icon"></i>Welcome! How to Use:</h3>
            <p style="margin: 0.5rem 0;">
                <i class="fas fa-check-circle fa-icon" style="color: #28a745;"></i><strong>Step 1:</strong> Upload your CV (PDF, DOCX, or TXT)<br>
                <i class="fas fa-check-circle fa-icon" style="color: #28a745;"></i><strong>Step 2:</strong> Paste the Job Description<br>
                <i class="fas fa-check-circle fa-icon" style="color: #28a745;"></i><strong>Step 3:</strong> Click "Analyze CV" and get instant results!<br>
                <i class="fas fa-lightbulb fa-icon" style="color: #ffd700;"></i><strong>Pro Tip:</strong> OpenAI API key is optional - the app works great without it!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown('<h2 style="margin: 0;"><i class="fas fa-cog fa-icon"></i>Settings</h2>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Info about no API key needed
        st.markdown("""
            <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
                <p style="margin: 0; color: #0c5460;">
                    <i class="fas fa-info-circle"></i> <strong>No API Key? No Problem!</strong>
                </p>
                <p style="margin: 0.5rem 0 0 0; color: #0c5460; font-size: 0.9rem;">
                    The app works perfectly without OpenAI. You'll get:<br>
                    • Match scores<br>
                    • Skill analysis<br>
                    • Keyword matching<br>
                    • Detailed recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Advanced Settings
        with st.expander("⚡ Advanced Settings (Optional)", expanded=False):
            # LLM Provider Selection
            st.markdown("**🤖 AI Provider Selection**")
            
            llm_providers = {
                "OpenAI GPT": "openai",
                "Anthropic Claude (Free Tier)": "anthropic",
                "Google Gemini (Free)": "google",
                "Groq (Fast & Free)": "groq",
                "Ollama (Local & Free)": "ollama"
            }
            
            selected_provider_name = st.selectbox(
                "Choose AI Provider",
                options=list(llm_providers.keys()),
                index=0,
                help="Select which AI provider to use for analysis"
            )
            
            llm_provider = llm_providers[selected_provider_name]
            
            # Show provider info
            if llm_provider == "ollama":
                st.info("💻 **Ollama**: Runs locally on your machine. Free & private. Install from ollama.ai")
                api_key_needed = False
                openai_key = None
            elif llm_provider in ["google", "groq", "anthropic"]:
                st.info(f"✨ **{selected_provider_name}**: Has a generous free tier! Great alternative to OpenAI.")
                api_key_needed = True
            else:
                api_key_needed = True
            
            # API Key Input
            if api_key_needed:
                openai_key = st.text_input(
                    f"{selected_provider_name} API Key",
                    type="password",
                    help=f"Enter your {selected_provider_name} API key for AI-powered insights"
                )
            else:
                openai_key = None
            
            # Show success message if API key provided or Ollama selected
            if openai_key or llm_provider == "ollama":
                st.markdown("""
                    <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem; border-radius: 5px;">
                        <p style="margin: 0; color: #155724;">
                            <i class="fas fa-check-circle"></i> <strong>AI features enabled!</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Use LLM toggle
            use_llm = st.checkbox(
                "Enable AI-Powered Analysis",
                value=bool(openai_key or llm_provider == "ollama"),
                help="Get advanced AI insights",
                disabled=not (openai_key or llm_provider == "ollama")
            )
            
            st.markdown("---")
            
            # RAG settings
            st.markdown('<h3><i class="fas fa-database fa-icon"></i>Database Features</h3>', unsafe_allow_html=True)
            if not RAG_AVAILABLE:
                st.markdown("""
                    <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 0.75rem; border-radius: 5px;">
                        <p style="margin: 0; color: #0c5460; font-size: 0.9rem;">
                            <i class="fas fa-info-circle"></i> Vector database features unavailable (ChromaDB not installed). Core analysis works perfectly!
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                use_rag = False
                save_to_db = False
            else:
                use_rag = st.checkbox(
                    "Enable Vector Database",
                    value=False,
                    help="Store and compare multiple CVs"
                )
                
                save_to_db = st.checkbox(
                    "Save to Database",
                    value=False,
                    help="Store current CV & JD for future comparisons",
                    disabled=not use_rag
                )
        
        # Initialize analyzer with selected LLM provider
        if openai_key or llm_provider == "ollama":
            st.session_state.analyzer = CVAnalyzer(llm_provider=llm_provider, api_key=openai_key)
        elif st.session_state.analyzer is None:
            st.session_state.analyzer = CVAnalyzer(llm_provider="openai", api_key=None)
        
        # Initialize database
        if use_rag and RAG_AVAILABLE and st.session_state.db is None:
            st.session_state.db = CVEmbeddingDB()
            
        # Database stats
        if use_rag and RAG_AVAILABLE and st.session_state.db:
            st.markdown("---")
            st.markdown('<h3><i class="fas fa-chart-bar fa-icon"></i>Database Stats</h3>', unsafe_allow_html=True)
            stats = st.session_state.db.get_collection_stats()
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Docs", stats['total_documents'])
            with col_stat2:
                st.metric("CVs Stored", stats['cvs'])
            
            if st.button("🗑️ Clear Database", use_container_width=True):
                st.session_state.db.clear_collection()
                st.markdown("""
                    <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem; border-radius: 5px; margin: 0.5rem 0;">
                        <p style="margin: 0; color: #155724;">
                            <i class="fas fa-check-circle"></i> <strong>Database cleared!</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.rerun()
        
        st.markdown("---")
        
        # Help Section
        with st.expander("❓ Need Help?", expanded=False):
            st.markdown("**Supported Formats:**")
            st.write("• PDF files")
            st.write("• DOCX files")
            st.write("• TXT files")
            
            st.markdown("")
            st.markdown("**Tips for Best Results:**")
            st.write("• Use well-formatted CVs")
            st.write("• Paste complete job descriptions")
            st.write("• Include technical skills in both")
            
            st.markdown("")
            st.markdown("**Scoring:**")
            col_h1, col_h2 = st.columns([1, 4])
            with col_h1:
                st.markdown("🏆")
            with col_h2:
                st.write("70-100%: Excellent match")
            
            col_h3, col_h4 = st.columns([1, 4])
            with col_h3:
                st.markdown("👍")
            with col_h4:
                st.write("50-69%: Good match")
            
            col_h5, col_h6 = st.columns([1, 4])
            with col_h5:
                st.markdown("⚠️")
            with col_h6:
                st.write("0-49%: Needs improvement")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<p class="section-header"><i class="fas fa-file-upload fa-icon"></i>Upload Your CV</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your CV file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your CV in PDF, DOCX, or TXT format (Max 10MB)",
            label_visibility="collapsed"
        )
        
        cv_text = ""
        if uploaded_file:
            try:
                with st.spinner("<i class='fas fa-spinner fa-spin'></i> Extracting text from your CV..."):
                    cv_text = extract_text(uploaded_file)
                
                # Success message with stats
                col_success1, col_success2 = st.columns([3, 1])
                with col_success1:
                    st.markdown("""
                        <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem; border-radius: 5px;">
                            <p style="margin: 0; color: #155724;">
                                <i class="fas fa-check-circle"></i> <strong>Successfully extracted text!</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                with col_success2:
                    st.markdown(f"""
                        <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 0.75rem; border-radius: 5px;">
                            <p style="margin: 0; color: #0c5460; text-align: center;">
                                <i class="fas fa-chart-line"></i> <strong>{len(cv_text)}</strong> chars
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("👁️ Preview CV Text (First 500 characters)"):
                    st.text_area(
                        "CV Content Preview",
                        cv_text[:500] + "..." if len(cv_text) > 500 else cv_text,
                        height=150,
                        label_visibility="collapsed"
                    )
            except Exception as e:
                st.markdown(f"""
                    <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 0.75rem; border-radius: 5px;">
                        <p style="margin: 0; color: #721c24;">
                            <i class="fas fa-times-circle"></i> <strong>Error extracting text:</strong> {str(e)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                    <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 0.75rem; border-radius: 5px; margin-top: 0.5rem;">
                        <p style="margin: 0; color: #856404;">
                            <i class="fas fa-lightbulb"></i> <strong>Tip:</strong> Make sure your file isn't password-protected or corrupted.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            # Show helpful message when no file uploaded
            st.markdown("""
                <div class="upload-section">
                    <p style="text-align: center; color: #666; margin: 1rem;">
                        <i class="fas fa-cloud-upload-alt" style="font-size: 2rem; display: block; margin-bottom: 0.5rem;"></i>
                        <strong>Drag and drop your CV here</strong><br>
                        <small>or click to browse</small>
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="section-header"><i class="fas fa-clipboard-list fa-icon"></i>Job Description</p>', unsafe_allow_html=True)
        
        jd_text = st.text_area(
            "Paste the job description here",
            height=300,
            placeholder="""Paste the complete job description here...

Example:
- Job Title: Software Engineer
- Required Skills: Python, Django, AWS
- Experience: 3+ years
- Responsibilities: ...
            """,
            label_visibility="collapsed"
        )
        
        if jd_text:
            char_count = len(jd_text)
            word_count = len(jd_text.split())
            st.caption(f"<i class='fas fa-chart-line'></i> {char_count} characters • {word_count} words")
    
    # Analysis button
    st.markdown("---")
    
    # Show analysis status
    if cv_text and jd_text:
        col_status1, col_status2, col_status3 = st.columns([1, 2, 1])
        with col_status2:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                            padding: 1rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                    <p style="margin: 0; color: #2e7d32; font-weight: 600;">
                        <i class="fas fa-check-circle"></i> Ready to analyze! Click the button below.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    elif not cv_text and not jd_text:
        st.markdown("""
            <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 1rem; border-radius: 5px;">
                <p style="margin: 0; color: #0c5460;">
                    <i class="fas fa-hand-point-up"></i> <strong>Please upload a CV and paste a Job Description to get started</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif not cv_text:
        st.markdown("""
            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px;">
                <p style="margin: 0; color: #856404;">
                    <i class="fas fa-exclamation-triangle"></i> <strong>Please upload a CV file</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px;">
                <p style="margin: 0; color: #856404;">
                    <i class="fas fa-exclamation-triangle"></i> <strong>Please paste a Job Description</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        analyze_button = st.button(
            "🔍 Analyze CV Now",
            use_container_width=True,
            type="primary",
            disabled=not (cv_text and jd_text)
        )
    
    # Perform analysis
    if analyze_button:
        if not cv_text:
            st.error("⚠️ Please upload a CV first!")
        elif not jd_text:
            st.error("⚠️ Please enter a job description!")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Start analysis
                status_text.text("🔍 Analyzing CV structure...")
                progress_bar.progress(25)
                
                # Step 2: Extract keywords
                status_text.text("🔎 Extracting keywords and skills...")
                progress_bar.progress(50)
                
                # Step 3: Calculate match
                status_text.text("📊 Calculating match score...")
                progress_bar.progress(75)
                
                # Perform analysis
                result = st.session_state.analyzer.analyze_cv_vs_jd(
                    cv_text=cv_text,
                    jd_text=jd_text,
                    use_llm=use_llm
                )
                
                st.session_state.analysis_result = result
                
                # Step 4: Complete
                status_text.text("✅ Analysis complete!")
                progress_bar.progress(100)
                
                # Save to database if enabled
                if use_rag and save_to_db and st.session_state.db:
                    st.session_state.db.add_cv(cv_text, {'source': uploaded_file.name if uploaded_file else 'unknown'})
                    st.session_state.db.add_jd(jd_text, {'timestamp': 'now'})
                    st.markdown("""
                        <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem; border-radius: 5px;">
                            <p style="margin: 0; color: #155724;">
                                <i class="fas fa-save"></i> <strong>Saved to database!</strong>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Clear progress indicators
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.markdown(f"""
                    <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 5px;">
                        <p style="margin: 0; color: #721c24;">
                            <i class="fas fa-times-circle"></i> <strong>Error during analysis:</strong> {str(e)}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown("""
                    <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 1rem; border-radius: 5px; margin-top: 0.5rem;">
                        <p style="margin: 0; color: #0c5460;">
                            <i class="fas fa-lightbulb"></i> <strong>Troubleshooting:</strong><br>
                            • Make sure your CV contains text<br>
                            • Check that the job description is complete<br>
                            • Try with a different file format
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                progress_bar.empty()
                status_text.empty()
    
    # Display results
    if st.session_state.analysis_result:
        st.markdown("---")
        st.markdown('<p class="section-header"><i class="fas fa-chart-pie fa-icon"></i>Analysis Results</p>', unsafe_allow_html=True)
        
        result = st.session_state.analysis_result
        
        # Overall score with visual appeal
        st.markdown('<h3><i class="fas fa-crosshairs fa-icon"></i>Overall Match Score</h3>', unsafe_allow_html=True)
        render_score_box(result['overall_score'])
        
        # Interpretation message
        score = result['overall_score']
        if score >= 70:
            st.markdown("""
                <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 5px;">
                    <p style="margin: 0; color: #155724;">
                        <i class="fas fa-star"></i> <strong>Excellent Match!</strong> This candidate shows strong alignment with the job requirements.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        elif score >= 50:
            st.markdown("""
                <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 1rem; border-radius: 5px;">
                    <p style="margin: 0; color: #0c5460;">
                        <i class="fas fa-thumbs-up"></i> <strong>Good Match!</strong> The candidate has several relevant qualifications worth considering.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px;">
                    <p style="margin: 0; color: #856404;">
                        <i class="fas fa-exclamation-triangle"></i> <strong>Moderate Match.</strong> The candidate may need additional skills or experience for this role.
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Detailed scores in a nice grid
        st.markdown('<h3><i class="fas fa-chart-line fa-icon"></i>Detailed Breakdown</h3>', unsafe_allow_html=True)
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric(
                "💼 Skill Match", 
                f"{result['skill_score']}%",
                help="Technical skills alignment"
            )
        with col_s2:
            st.metric(
                "⏱️ Experience", 
                f"{result['experience_score']}%",
                help="Years of experience match"
            )
        with col_s3:
            st.metric(
                "📄 Text Similarity", 
                f"{result['similarity_score']}%",
                help="Overall document similarity (TF-IDF)"
            )
        with col_s4:
            st.metric(
                "🔤 Keyword Match", 
                f"{result['keyword_score']}%",
                help="How well CV keywords match the job description"
            )
        
        # Experience details in a nice info box
        if 'cv_years_experience' in result and result['cv_years_experience'] > 0:
            st.markdown('<h3><i class="fas fa-briefcase fa-icon"></i>Experience Analysis</h3>', unsafe_allow_html=True)
            
            exp_col1, exp_col2 = st.columns([2, 1])
            
            with exp_col1:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                                padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1976d2;">
                        <h4 style="margin: 0 0 0.5rem 0; color: #1565c0;"><i class="fas fa-chart-bar"></i> Experience Summary</h4>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                            <strong>Total Experience:</strong> {result['cv_years_experience']} years ({result['cv_jobs_found']} positions found)
                        </p>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                            <strong>Required Experience:</strong> {result['jd_required_years']} years
                        </p>
                        <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                            <strong>Status:</strong> {result['experience_match']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with exp_col2:
                # Visual indicator
                cv_exp = result['cv_years_experience']
                req_exp = result['jd_required_years']
                
                if req_exp > 0:
                    if cv_exp >= req_exp:
                        st.markdown(f"""
                            <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 5px; text-align: center;">
                                <p style="margin: 0; color: #155724;">
                                    <i class="fas fa-check-circle"></i> <strong>Meets Requirement</strong>
                                </p>
                                <p style="margin: 0.5rem 0 0 0; color: #155724; font-size: 1.2rem; font-weight: 600;">
                                    {cv_exp} ≥ {req_exp} years
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    elif cv_exp >= req_exp * 0.7:
                        st.markdown(f"""
                            <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px; text-align: center;">
                                <p style="margin: 0; color: #856404;">
                                    <i class="fas fa-exclamation-triangle"></i> <strong>Close Match</strong>
                                </p>
                                <p style="margin: 0.5rem 0 0 0; color: #856404; font-size: 1.2rem; font-weight: 600;">
                                    {cv_exp} vs {req_exp} years
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 5px; text-align: center;">
                                <p style="margin: 0; color: #721c24;">
                                    <i class="fas fa-times-circle"></i> <strong>Below Requirement</strong>
                                </p>
                                <p style="margin: 0.5rem 0 0 0; color: #721c24; font-size: 1.2rem; font-weight: 600;">
                                    {cv_exp} < {req_exp} years
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="background-color: #d1ecf1; border-left: 4px solid #0c5460; padding: 1rem; border-radius: 5px; text-align: center;">
                            <p style="margin: 0; color: #0c5460;">
                                <i class="fas fa-info-circle"></i> <strong>No Requirement</strong>
                            </p>
                            <p style="margin: 0.5rem 0 0 0; color: #0c5460; font-size: 1.2rem; font-weight: 600;">
                                Candidate: {cv_exp} years
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Matched and Missing Skills with better visuals
        st.markdown('<h3><i class="fas fa-tasks fa-icon"></i>Skills Analysis</h3>', unsafe_allow_html=True)
        
        col_match1, col_match2 = st.columns(2)
        
        with col_match1:
            st.markdown('<p style="font-size: 1.2rem; font-weight: 600; color: #28a745;"><i class="fas fa-check-circle"></i> Matched Skills</p>', 
                       unsafe_allow_html=True)
            
            if result['matched_skills']:
                # Show count
                st.caption(f"Found {len(result['matched_skills'])} matching skills")
                
                # Display skills as badges
                skills_html = "".join([
                    f'<span class="skill-badge"><i class="fas fa-check"></i> {skill}</span>' 
                    for skill in result['matched_skills'][:20]
                ])
                st.markdown(f'<div style="margin: 1rem 0;">{skills_html}</div>', unsafe_allow_html=True)
                
                # Show more if needed
                if len(result['matched_skills']) > 20:
                    with st.expander(f"Show all {len(result['matched_skills'])} matched skills"):
                        st.write(", ".join(result['matched_skills']))
            else:
                st.info("No specific technical skills matched. Consider highlighting more technical keywords in the CV.")
        
        with col_match2:
            st.markdown('<p style="font-size: 1.2rem; font-weight: 600; color: #dc3545;"><i class="fas fa-times-circle"></i> Missing Skills</p>', 
                       unsafe_allow_html=True)
            
            if result['missing_skills']:
                # Show count
                st.caption(f"Missing {len(result['missing_skills'])} skills from JD")
                
                # Display missing skills as badges
                skills_html = "".join([
                    f'<span class="missing-badge"><i class="fas fa-times"></i> {skill}</span>' 
                    for skill in result['missing_skills'][:20]
                ])
                st.markdown(f'<div style="margin: 1rem 0;">{skills_html}</div>', unsafe_allow_html=True)
                
                # Show more if needed
                if len(result['missing_skills']) > 20:
                    with st.expander(f"Show all {len(result['missing_skills'])} missing skills"):
                        st.write(", ".join(result['missing_skills']))
            else:
                st.markdown("""
                    <div style="background-color: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 5px;">
                        <p style="margin: 0; color: #155724;">
                            <i class="fas fa-trophy"></i> <strong>No missing skills! Excellent coverage!</strong>
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Matched and Missing Keywords
        with st.expander("🔍 Detailed Keyword Analysis", expanded=False):
            col_k1, col_k2 = st.columns(2)
            
            with col_k1:
                st.markdown("**<i class='fas fa-check-circle' style='color: #28a745;'></i> Matched Keywords:**")
                if result['matched_keywords']:
                    # Show in organized way
                    keywords_per_row = 5
                    keywords_list = result['matched_keywords'][:30]
                    
                    keywords_html = ""
                    for i, kw in enumerate(keywords_list):
                        keywords_html += f'<span style="background: #e3f2fd; padding: 0.3rem 0.7rem; border-radius: 5px; margin: 0.2rem; display: inline-block; font-size: 0.9rem;">{kw}</span>'
                        if (i + 1) % keywords_per_row == 0:
                            keywords_html += "<br>"
                    
                    st.markdown(keywords_html, unsafe_allow_html=True)
                    st.caption(f"Showing top {len(keywords_list)} keywords")
                else:
                    st.write("None")
            
            with col_k2:
                st.markdown("**<i class='fas fa-times-circle' style='color: #dc3545;'></i> Missing Keywords:**")
                if result['missing_keywords']:
                    # Show in organized way
                    keywords_per_row = 5
                    keywords_list = result['missing_keywords'][:30]
                    
                    keywords_html = ""
                    for i, kw in enumerate(keywords_list):
                        keywords_html += f'<span style="background: #ffebee; padding: 0.3rem 0.7rem; border-radius: 5px; margin: 0.2rem; display: inline-block; font-size: 0.9rem;">{kw}</span>'
                        if (i + 1) % keywords_per_row == 0:
                            keywords_html += "<br>"
                    
                    st.markdown(keywords_html, unsafe_allow_html=True)
                    st.caption(f"Showing top {len(keywords_list)} keywords")
                else:
                    st.write("None")
        
        # LLM Analysis - showcase it prominently
        if result.get('llm_analysis'):
            st.markdown("---")
            st.markdown('<h3><i class="fas fa-robot fa-icon"></i>Expert Analysis & Recommendations</h3>', unsafe_allow_html=True)
            
            # Create a nice box for the analysis
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4;">
                    {result['llm_analysis'].replace('\n', '<br>')}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("---")
            st.markdown('<h3><i class="fas fa-file-alt fa-icon"></i>Analysis Summary</h3>', unsafe_allow_html=True)
            st.markdown("""
                <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 5px;">
                    <p style="margin: 0; color: #856404;">
                        <i class="fas fa-lightbulb"></i> <strong>Tip:</strong> Add your OpenAI API key in the sidebar to get AI-powered recommendations and detailed insights!
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Download report with better formatting
        st.markdown("---")
        st.markdown('<h3><i class="fas fa-download fa-icon"></i>Export Results</h3>', unsafe_allow_html=True)
        
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        
        with col_dl2:
            # Create comprehensive report
            report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    CV ANALYSIS REPORT                            ║
║                  Generated by Smart CV Filter                     ║
╚══════════════════════════════════════════════════════════════════╝

📊 MATCH SCORES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Match Score:      {result['overall_score']}%
Skill Match:              {result['skill_score']}%
Experience Match:         {result['experience_score']}%
Text Similarity:          {result['similarity_score']}%
Keyword Match:            {result['keyword_score']}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💼 EXPERIENCE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Candidate Total Experience: {result.get('cv_years_experience', 0)} years
Positions Found:            {result.get('cv_jobs_found', 0)}
Required Experience:        {result.get('jd_required_years', 0)} years
Experience Match Status:    {result.get('experience_match', 'Not specified')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ MATCHED SKILLS ({len(result['matched_skills'])})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{', '.join(result['matched_skills']) if result['matched_skills'] else 'None'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ MISSING SKILLS ({len(result['missing_skills'])})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{', '.join(result['missing_skills']) if result['missing_skills'] else 'None'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 MATCHED KEYWORDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{', '.join(result['matched_keywords'][:50]) if result['matched_keywords'] else 'None'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🤖 EXPERT ANALYSIS & RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{result.get('llm_analysis', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{'✅ Recommended for Interview' if result['overall_score'] >= 70 else '⚖️ Consider with Reservations' if result['overall_score'] >= 50 else '❌ Needs Improvement'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generated by Smart CV Filter - AI-Powered CV Analysis
Report Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            st.download_button(
                label="📥 Download Complete Report",
                data=report,
                file_name=f"cv_analysis_report_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            st.markdown("""
                <p style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 0.5rem;">
                    <i class="fas fa-lightbulb"></i> Download includes full analysis with scores, skills, keywords, and recommendations
                </p>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0 1rem 0;">
            <p style="margin: 0;">
                Made with <i class="fas fa-heart" style="color: #e74c3c;"></i> by <strong>Smart CV Filter</strong> | 
                <a href="https://github.com/Atik1000/Smart-CV-Filter" target="_blank" style="color: #1f77b4; text-decoration: none;">
                    <i class="fab fa-github"></i> GitHub
                </a>
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                <i class="fas fa-lightbulb" style="color: #ffd700;"></i> <strong>Tip:</strong> Works great without OpenAI API key! Add it for advanced AI insights.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
