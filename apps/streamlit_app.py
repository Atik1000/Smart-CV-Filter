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

# Optional RAG support
try:
    from utils.embedding_db import CVEmbeddingDB
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    print(f"⚠️ ChromaDB not available - RAG features disabled (Reason: {type(e).__name__})")


# Page configuration
st.set_page_config(
    page_title="Smart CV Filter - AI-Powered CV Analysis",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced UI
st.markdown("""
    <style>
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
    st.markdown('<h1 class="main-header">🎯 Smart CV Filter</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent CV Analysis - Match CVs with Job Descriptions Instantly!</p>', 
                unsafe_allow_html=True)
    
    # Welcome Message
    st.markdown("""
        <div class="info-box">
            <h3>👋 Welcome! How to Use:</h3>
            <p style="margin: 0.5rem 0;">
                ✅ <strong>Step 1:</strong> Upload your CV (PDF, DOCX, or TXT)<br>
                ✅ <strong>Step 2:</strong> Paste the Job Description<br>
                ✅ <strong>Step 3:</strong> Click "Analyze CV" and get instant results!<br>
                💡 <strong>Pro Tip:</strong> OpenAI API key is optional - the app works great without it!
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.markdown("---")
        
        # Info about no API key needed
        st.info("ℹ️ **No API Key? No Problem!**\n\nThe app works perfectly without OpenAI. You'll get:\n- Match scores\n- Skill analysis\n- Keyword matching\n- Detailed recommendations")
        
        st.markdown("---")
        
        # Advanced Settings
        with st.expander("🚀 Advanced Settings (Optional)", expanded=False):
            # OpenAI API Key
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Optional: Add for AI-powered insights. The app works great without it!"
            )
            
            if openai_key:
                st.success("✅ API Key added! AI features enabled.")
            
            # Use LLM toggle
            use_llm = st.checkbox(
                "Enable AI-Powered Analysis",
                value=bool(openai_key),
                help="Get advanced AI insights (requires API key)",
                disabled=not openai_key
            )
            
            st.markdown("---")
            
            # RAG settings
            st.subheader("🔍 Database Features")
            if not RAG_AVAILABLE:
                st.info("ℹ️ Vector database features unavailable (ChromaDB not installed). Core analysis works perfectly!")
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
        
        # Initialize analyzer
        if openai_key:
            st.session_state.analyzer = CVAnalyzer(openai_api_key=openai_key)
        elif st.session_state.analyzer is None:
            st.session_state.analyzer = CVAnalyzer()
        
        # Initialize database
        if use_rag and RAG_AVAILABLE and st.session_state.db is None:
            st.session_state.db = CVEmbeddingDB()
            
        # Database stats
        if use_rag and RAG_AVAILABLE and st.session_state.db:
            st.markdown("---")
            st.subheader("📊 Database Stats")
            stats = st.session_state.db.get_collection_stats()
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Total Docs", stats['total_documents'])
            with col_stat2:
                st.metric("CVs Stored", stats['cvs'])
            
            if st.button("🗑️ Clear Database", use_container_width=True):
                st.session_state.db.clear_collection()
                st.success("✅ Database cleared!")
                st.rerun()
        
        st.markdown("---")
        
        # Help Section
        with st.expander("❓ Need Help?"):
            st.markdown("""
                **Supported Formats:**
                - PDF files
                - DOCX files  
                - TXT files
                
                **Tips for Best Results:**
                - Use well-formatted CVs
                - Paste complete job descriptions
                - Include technical skills in both
                
                **Scoring:**
                - 70-100%: Excellent match
                - 50-69%: Good match
                - 0-49%: Needs improvement
            """)
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<p class="section-header">📤 Upload Your CV</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose your CV file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your CV in PDF, DOCX, or TXT format (Max 10MB)",
            label_visibility="collapsed"
        )
        
        cv_text = ""
        if uploaded_file:
            try:
                with st.spinner("📄 Extracting text from your CV..."):
                    cv_text = extract_text(uploaded_file)
                
                # Success message with stats
                col_success1, col_success2 = st.columns([3, 1])
                with col_success1:
                    st.success(f"✅ Successfully extracted text!")
                with col_success2:
                    st.info(f"📊 {len(cv_text)} chars")
                
                with st.expander("👀 Preview CV Text (First 500 characters)"):
                    st.text_area(
                        "CV Content Preview",
                        cv_text[:500] + "..." if len(cv_text) > 500 else cv_text,
                        height=150,
                        label_visibility="collapsed"
                    )
            except Exception as e:
                st.error(f"❌ Error extracting text: {str(e)}")
                st.info("💡 **Tip:** Make sure your file isn't password-protected or corrupted.")
        else:
            # Show helpful message when no file uploaded
            st.markdown("""
                <div class="upload-section">
                    <p style="text-align: center; color: #666; margin: 1rem;">
                        📁 <strong>Drag and drop your CV here</strong><br>
                        <small>or click to browse</small>
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="section-header">📝 Job Description</p>', unsafe_allow_html=True)
        
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
            st.caption(f"📊 {char_count} characters • {word_count} words")
    
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
                        ✅ Ready to analyze! Click the button below.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    elif not cv_text and not jd_text:
        st.info("👆 Please upload a CV and paste a Job Description to get started")
    elif not cv_text:
        st.warning("⚠️ Please upload a CV file")
    else:
        st.warning("⚠️ Please paste a Job Description")
    
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
                    st.success("💾 Saved to database!")
                
                # Clear progress indicators
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 **Troubleshooting:**\n- Make sure your CV contains text\n- Check that the job description is complete\n- Try with a different file format")
                progress_bar.empty()
                status_text.empty()
    
    # Display results
    if st.session_state.analysis_result:
        st.markdown("---")
        st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)
        
        result = st.session_state.analysis_result
        
        # Overall score with visual appeal
        st.markdown("### 🎯 Overall Match Score")
        render_score_box(result['overall_score'])
        
        # Interpretation message
        score = result['overall_score']
        if score >= 70:
            st.success("🌟 **Excellent Match!** This candidate shows strong alignment with the job requirements.")
        elif score >= 50:
            st.info("👍 **Good Match!** The candidate has several relevant qualifications worth considering.")
        else:
            st.warning("⚠️ **Moderate Match.** The candidate may need additional skills or experience for this role.")
        
        # Detailed scores in a nice grid
        st.markdown("### 📈 Detailed Breakdown")
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.metric(
                "🔤 Keyword Match", 
                f"{result['keyword_score']}%",
                help="How well CV keywords match the job description"
            )
        with col_s2:
            st.metric(
                "💼 Skill Match", 
                f"{result['skill_score']}%",
                help="Technical skills alignment"
            )
        with col_s3:
            st.metric(
                "📄 Text Similarity", 
                f"{result['similarity_score']}%",
                help="Overall document similarity (TF-IDF)"
            )
        
        # Matched and Missing Skills with better visuals
        st.markdown("### 🎯 Skills Analysis")
        
        col_match1, col_match2 = st.columns(2)
        
        with col_match1:
            st.markdown('<p style="font-size: 1.2rem; font-weight: 600; color: #28a745;">✅ Matched Skills</p>', 
                       unsafe_allow_html=True)
            
            if result['matched_skills']:
                # Show count
                st.caption(f"Found {len(result['matched_skills'])} matching skills")
                
                # Display skills as badges
                skills_html = "".join([
                    f'<span class="skill-badge">✓ {skill}</span>' 
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
            st.markdown('<p style="font-size: 1.2rem; font-weight: 600; color: #dc3545;">❌ Missing Skills</p>', 
                       unsafe_allow_html=True)
            
            if result['missing_skills']:
                # Show count
                st.caption(f"Missing {len(result['missing_skills'])} skills from JD")
                
                # Display missing skills as badges
                skills_html = "".join([
                    f'<span class="missing-badge">✗ {skill}</span>' 
                    for skill in result['missing_skills'][:20]
                ])
                st.markdown(f'<div style="margin: 1rem 0;">{skills_html}</div>', unsafe_allow_html=True)
                
                # Show more if needed
                if len(result['missing_skills']) > 20:
                    with st.expander(f"Show all {len(result['missing_skills'])} missing skills"):
                        st.write(", ".join(result['missing_skills']))
            else:
                st.success("🎉 No missing skills! Excellent coverage!")
        
        # Matched and Missing Keywords
        with st.expander("🔍 Detailed Keyword Analysis", expanded=False):
            col_k1, col_k2 = st.columns(2)
            
            with col_k1:
                st.markdown("**✅ Matched Keywords:**")
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
                st.markdown("**❌ Missing Keywords:**")
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
            st.markdown("### 🤖 Expert Analysis & Recommendations")
            
            # Create a nice box for the analysis
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                            padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4;">
                    {result['llm_analysis'].replace('\n', '<br>')}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("---")
            st.markdown("### 📝 Analysis Summary")
            st.info("💡 **Tip:** Add your OpenAI API key in the sidebar to get AI-powered recommendations and detailed insights!")
        
        # Download report with better formatting
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
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
Keyword Match:            {result['keyword_score']}%
Skill Match:              {result['skill_score']}%
Text Similarity:          {result['similarity_score']}%

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
            
            st.caption("💡 Download includes full analysis with scores, skills, keywords, and recommendations")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0 1rem 0;">
            <p style="margin: 0;">
                Made with ❤️ by <strong>Smart CV Filter</strong> | 
                <a href="https://github.com/Atik1000/Smart-CV-Filter" target="_blank" style="color: #1f77b4; text-decoration: none;">GitHub</a>
            </p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
                💡 <strong>Tip:</strong> Works great without OpenAI API key! Add it for advanced AI insights.
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
