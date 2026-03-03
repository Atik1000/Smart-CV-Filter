# 📄 Smart CV Filter

An intelligent CV analysis tool that matches CVs against Job Descriptions using advanced NLP and optional AI features.

**✨ Works Perfectly WITHOUT OpenAI API Key! ✨**

## 🎯 Key Highlights

- 🚀 **No API Key Required** - Full functionality works out of the box
- 📊 **Intelligent Scoring** - TF-IDF, keyword matching, and similarity analysis
- 💼 **Skill Detection** - Automatic identification of technical skills
- 📈 **Match Score** - Get 0-100% compatibility scores instantly
- 📱 **Beautiful UI** - Modern, user-friendly Streamlit interface
- 🤖 **Optional AI** - Add OpenAI key for enhanced insights (not required)
- 💾 **Free & Private** - All processing happens locally

## 🌟 Features

### Core Features (No API Key Needed)
- **Multi-format Support**: Upload CVs in PDF, DOCX, or TXT format
- **Intelligent Analysis**: 
  - Keyword extraction and matching using TF-IDF
  - Technical skill identification (Python, AWS, Docker, etc.)
  - Match score calculation (0-100%)
  - Text similarity analysis
  - Professional recommendations
- **Beautiful UI**: Modern Streamlit interface with real-time analysis
- **Detailed Reports**: Downloadable analysis reports with full breakdown

### Optional AI Features (with OpenAI API Key)
- **AI-Powered Insights**: Natural language recommendations (OpenAI GPT-3.5)
- **RAG Integration**: 
  - Vector database (ChromaDB) for storing CVs and JDs
  - Contextual analysis using similar historical matches
  - Intelligent comparison across multiple candidates

## 📁 Project Structure

```
cv-filtering/
├── apps/
│   ├── __init__.py
│   └── streamlit_app.py          # Main Streamlit UI
├── utils/
│   ├── __init__.py
│   ├── extract_text.py            # Text extraction from PDF/DOCX/TXT
│   ├── analyze_cv.py              # CV analysis engine
│   └── embedding_db.py            # Vector database for RAG
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- **(Optional)** OpenAI API key - only if you want AI-powered insights

### Installation

1. **Clone or download this repository**
   ```bash
   cd cv-filtering
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **[OPTIONAL] Set up OpenAI API key** (skip if you don't have one)
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```
   
   **Note:** The app works great without this step!

### Running the Application

```bash
streamlit run apps/streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`

**🎉 That's it! No API key needed to start analyzing CVs!**

## 📖 Usage Guide

### Basic Usage

1. **Upload CV**: Click "Choose a file" and upload a CV (PDF, DOCX, or TXT)
2. **Enter Job Description**: Paste the job description in the text area
3. **Analyze**: Click "🔍 Analyze CV" button
4. **Review Results**:
   - Overall match score
   - Matched and missing skills
   - Keyword analysis
   - AI recommendations (if OpenAI key is provided)

### Advanced Features

#### Using LLM Analysis

1. Enter your OpenAI API key in the sidebar
2. Check "Use LLM Analysis"
3. Get AI-powered insights and recommendations

#### Using RAG (Vector Database)

1. Enable "Enable RAG (Vector DB)" in the sidebar
2. Check "Save CV & JD to Database" to store documents
3. The system will use similar historical CVs/JDs for better context

#### Database Management

- View database statistics in the sidebar
- Clear database using "Clear Database" button

### Example Workflow

```
1. Upload CV: software_engineer_resume.pdf
2. Paste JD: "Looking for Python developer with Django, AWS..."
3. Click Analyze
4. Get results:
   - Match Score: 75%
   - Matched Skills: python, django, git, docker
   - Missing Skills: aws, kubernetes
   - AI Recommendation: "Strong candidate. Consider highlighting..."
5. Download report for documentation
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (copy from `.env.example`):

```env
OPENAI_API_KEY=sk-your-key-here
CHROMA_DB_PATH=./chroma_db
DEBUG=False
```

### Streamlit Configuration

Edit `.streamlit/config.toml` to customize:
- Theme colors
- Server port
- CORS settings

## 🏗️ Architecture

### Text Extraction Layer
- `extract_text.py`: Handles PDF, DOCX, TXT file parsing
- Uses `pdfplumber` for PDFs, `docx2txt` for DOCX

### Analysis Engine
- `analyze_cv.py`: Core analysis logic
  - Keyword extraction using TF-IDF
  - Skill pattern matching
  - Match score calculation
  - LLM integration for insights

### RAG Layer
- `embedding_db.py`: Vector database management
  - ChromaDB for embeddings
  - Similarity search
  - Context retrieval for LLM

### UI Layer
- `streamlit_app.py`: Interactive web interface
  - File upload
  - Real-time analysis
  - Results visualization

## 📊 Scoring Algorithm

The overall match score is a weighted average of:

- **Keyword Match (30%)**: Percentage of JD keywords found in CV
- **Skill Match (40%)**: Percentage of technical skills matched
- **Text Similarity (30%)**: Cosine similarity of TF-IDF vectors

Score Interpretation:
- **70-100%**: Excellent match - Recommended for interview
- **50-69%**: Good match - Consider with reservations
- **0-49%**: Weak match - Significant gaps

## 🛠️ Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8 .
```

### Adding New Features

1. **New File Format Support**: 
   - Add extraction function in `utils/extract_text.py`
   - Update `extract_text()` dispatcher

2. **Custom Skill Patterns**:
   - Modify `extract_skills_and_technologies()` in `utils/analyze_cv.py`

3. **Alternative LLM**:
   - Update LLM integration in `utils/analyze_cv.py`
   - Add new API key to `.env`

## 📦 Dependencies

### Core
- `streamlit`: Web UI framework
- `pdfplumber`: PDF text extraction
- `docx2txt`: DOCX text extraction
- `scikit-learn`: TF-IDF and similarity
- `openai`: GPT-3.5 integration
- `chromadb`: Vector database
- `langchain`: LLM orchestration

See `requirements.txt` for complete list.

## 🔐 Security Notes

- **Never commit API keys** to version control
- Use `.env` file for sensitive configuration
- API keys are stored in session state (not persisted)
- Vector database is local by default

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Error extracting text from PDF"
- **Solution**: Ensure PDF is not password-protected or corrupted

**Issue**: "OpenAI API error"
- **Solution**: Check API key validity and account credits

**Issue**: "ChromaDB import error"
- **Solution**: Reinstall with `pip install chromadb --upgrade`

**Issue**: "Streamlit not found"
- **Solution**: Ensure virtual environment is activated

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- OpenAI for GPT-3.5 API
- Streamlit for the amazing UI framework
- ChromaDB for vector database
- All open-source contributors

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check existing documentation
- Review troubleshooting section

## 🚀 Future Enhancements

- [ ] Support for multiple LLM providers (Anthropic, Cohere, etc.)
- [ ] Batch CV processing
- [ ] Historical analytics dashboard
- [ ] Export to multiple formats (PDF, JSON)
- [ ] Advanced NLP with spaCy
- [ ] Resume builder based on JD
- [ ] Multi-language support
- [ ] API endpoint for integration

---

**Made with ❤️ for better hiring decisions**
