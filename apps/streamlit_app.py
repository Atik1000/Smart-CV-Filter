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
from utils.embedding_db import CVEmbeddingDB


# Page configuration
st.set_page_config(
    page_title="Smart CV Filter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .score-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .score-high {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .score-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .score-low {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .skill-badge {
        background-color: #e3f2fd;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
    }
    .missing-badge {
        background-color: #ffebee;
        padding: 0.3rem 0.6rem;
        border-radius: 5px;
        margin: 0.2rem;
        display: inline-block;
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
        emoji = "🎉"
        label = "Excellent Match"
    elif score >= 50:
        css_class = "score-medium"
        emoji = "👍"
        label = "Good Match"
    else:
        css_class = "score-low"
        emoji = "⚠️"
        label = "Needs Improvement"
    
    st.markdown(f"""
        <div class="score-box {css_class}">
            <h2>{emoji} {score}%</h2>
            <p style="margin: 0; font-size: 1.2rem;">{label}</p>
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application logic."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📄 Smart CV Filter</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered CV Analysis using RAG + LLM</p>', 
                unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OpenAI API Key
        openai_key = st.text_input(
            "OpenAI API Key (Optional)",
            type="password",
            help="Enter your OpenAI API key for LLM analysis. Leave empty for basic analysis."
        )
        
        # Use LLM toggle
        use_llm = st.checkbox(
            "Use LLM Analysis",
            value=bool(openai_key),
            help="Enable AI-powered insights and suggestions",
            disabled=not openai_key
        )
        
        # RAG settings
        st.subheader("🔍 RAG Settings")
        use_rag = st.checkbox(
            "Enable RAG (Vector DB)",
            value=False,
            help="Store and retrieve similar CVs/JDs for better context"
        )
        
        save_to_db = st.checkbox(
            "Save CV & JD to Database",
            value=False,
            help="Store current CV and JD for future comparisons",
            disabled=not use_rag
        )
        
        # Initialize analyzer
        if openai_key:
            st.session_state.analyzer = CVAnalyzer(openai_api_key=openai_key)
        elif st.session_state.analyzer is None:
            st.session_state.analyzer = CVAnalyzer()
        
        # Initialize database
        if use_rag and st.session_state.db is None:
            st.session_state.db = CVEmbeddingDB()
            
        # Database stats
        if use_rag and st.session_state.db:
            st.subheader("📊 Database Stats")
            stats = st.session_state.db.get_collection_stats()
            st.write(f"Total Documents: {stats['total_documents']}")
            st.write(f"CVs: {stats['cvs']}")
            st.write(f"JDs: {stats['jds']}")
            
            if st.button("Clear Database"):
                st.session_state.db.clear_collection()
                st.success("Database cleared!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload CV")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload CV in PDF, DOCX, or TXT format"
        )
        
        cv_text = ""
        if uploaded_file:
            try:
                with st.spinner("Extracting text from CV..."):
                    cv_text = extract_text(uploaded_file)
                st.success(f"✅ Extracted {len(cv_text)} characters from CV")
                
                with st.expander("Preview CV Text"):
                    st.text_area("CV Content", cv_text[:1000] + "...", height=200)
            except Exception as e:
                st.error(f"Error extracting text: {str(e)}")
    
    with col2:
        st.subheader("📝 Job Description")
        jd_text = st.text_area(
            "Paste job description here",
            height=300,
            placeholder="Paste the job description text here..."
        )
    
    # Analysis button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        analyze_button = st.button(
            "🔍 Analyze CV",
            use_container_width=True,
            type="primary"
        )
    
    # Perform analysis
    if analyze_button:
        if not cv_text:
            st.error("⚠️ Please upload a CV first!")
        elif not jd_text:
            st.error("⚠️ Please enter a job description!")
        else:
            with st.spinner("Analyzing CV against Job Description..."):
                try:
                    # Perform analysis
                    result = st.session_state.analyzer.analyze_cv_vs_jd(
                        cv_text=cv_text,
                        jd_text=jd_text,
                        use_llm=use_llm
                    )
                    
                    st.session_state.analysis_result = result
                    
                    # Save to database if enabled
                    if use_rag and save_to_db and st.session_state.db:
                        st.session_state.db.add_cv(cv_text, {'source': uploaded_file.name if uploaded_file else 'unknown'})
                        st.session_state.db.add_jd(jd_text, {'timestamp': 'now'})
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    # Display results
    if st.session_state.analysis_result:
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        result = st.session_state.analysis_result
        
        # Overall score
        st.subheader("Overall Match Score")
        render_score_box(result['overall_score'])
        
        # Detailed scores
        st.subheader("Detailed Scores")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric("Keyword Match", f"{result['keyword_score']}%")
        with col_s2:
            st.metric("Skill Match", f"{result['skill_score']}%")
        with col_s3:
            st.metric("Text Similarity", f"{result['similarity_score']}%")
        
        # Matched and Missing Skills
        col_match1, col_match2 = st.columns(2)
        
        with col_match1:
            st.subheader("✅ Matched Skills")
            if result['matched_skills']:
                skills_html = "".join([
                    f'<span class="skill-badge">{skill}</span>' 
                    for skill in result['matched_skills'][:15]
                ])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No specific skills matched")
        
        with col_match2:
            st.subheader("❌ Missing Skills")
            if result['missing_skills']:
                skills_html = "".join([
                    f'<span class="missing-badge">{skill}</span>' 
                    for skill in result['missing_skills'][:15]
                ])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.success("No missing skills!")
        
        # Matched and Missing Keywords
        with st.expander("🔍 Keyword Analysis"):
            col_k1, col_k2 = st.columns(2)
            
            with col_k1:
                st.write("**Matched Keywords:**")
                if result['matched_keywords']:
                    st.write(", ".join(result['matched_keywords'][:20]))
                else:
                    st.write("None")
            
            with col_k2:
                st.write("**Missing Keywords:**")
                if result['missing_keywords']:
                    st.write(", ".join(result['missing_keywords'][:20]))
                else:
                    st.write("None")
        
        # LLM Analysis
        if result.get('llm_analysis'):
            st.subheader("🤖 AI Analysis & Recommendations")
            st.markdown(result['llm_analysis'])
        
        # Download report
        st.markdown("---")
        report = f"""
CV ANALYSIS REPORT
==================

Overall Match Score: {result['overall_score']}%
Keyword Match: {result['keyword_score']}%
Skill Match: {result['skill_score']}%
Text Similarity: {result['similarity_score']}%

Matched Skills ({len(result['matched_skills'])}):
{', '.join(result['matched_skills'])}

Missing Skills ({len(result['missing_skills'])}):
{', '.join(result['missing_skills'])}

AI Analysis:
{result.get('llm_analysis', 'N/A')}

---
Generated by Smart CV Filter
"""
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="cv_analysis_report.txt",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()
