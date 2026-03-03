"""
Sample test file for CV analysis utilities.
Run with: pytest tests/
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.analyze_cv import CVAnalyzer


def test_keyword_extraction():
    """Test keyword extraction from text."""
    analyzer = CVAnalyzer()
    
    text = """
    Python developer with experience in Django, Flask, and FastAPI.
    Strong skills in machine learning using TensorFlow and PyTorch.
    Familiar with AWS, Docker, and Kubernetes.
    """
    
    keywords = analyzer.extract_keywords(text, top_n=10)
    
    assert isinstance(keywords, list)
    assert len(keywords) > 0
    assert 'python' in [k.lower() for k in keywords] or 'developer' in [k.lower() for k in keywords]


def test_skill_extraction():
    """Test technical skill extraction."""
    analyzer = CVAnalyzer()
    
    text = """
    Experienced in Python, JavaScript, and React.
    Worked with PostgreSQL and MongoDB databases.
    Deployed applications on AWS using Docker.
    """
    
    skills = analyzer.extract_skills_and_technologies(text)
    
    assert isinstance(skills, list)
    assert len(skills) > 0


def test_match_score_calculation():
    """Test CV vs JD matching score."""
    analyzer = CVAnalyzer()
    
    cv_text = """
    Software Engineer with 5 years of experience in Python development.
    Expertise in Django, Flask, and building RESTful APIs.
    Strong knowledge of PostgreSQL, Redis, and Docker.
    Experience with AWS cloud services and CI/CD pipelines.
    """
    
    jd_text = """
    We are looking for a Python Developer with:
    - 3+ years of Python experience
    - Django or Flask framework knowledge
    - Experience with PostgreSQL
    - AWS and Docker experience preferred
    - Strong API development skills
    """
    
    result = analyzer.calculate_match_score(cv_text, jd_text)
    
    assert 'overall_score' in result
    assert 'keyword_score' in result
    assert 'skill_score' in result
    assert 'matched_skills' in result
    assert 'missing_skills' in result
    assert 0 <= result['overall_score'] <= 100


def test_fallback_analysis():
    """Test fallback analysis without LLM."""
    analyzer = CVAnalyzer()
    
    match_data = {
        'overall_score': 75.5,
        'matched_skills': ['python', 'django', 'postgresql'],
        'missing_skills': ['aws', 'kubernetes']
    }
    
    analysis = analyzer._get_fallback_analysis(match_data)
    
    assert isinstance(analysis, str)
    assert len(analysis) > 0
    assert 'Assessment' in analysis or 'Recommendation' in analysis


def test_analyze_cv_vs_jd():
    """Test complete CV vs JD analysis."""
    analyzer = CVAnalyzer()
    
    cv_text = "Python developer with Django experience and PostgreSQL knowledge."
    jd_text = "Looking for Python developer with Django and database skills."
    
    result = analyzer.analyze_cv_vs_jd(cv_text, jd_text, use_llm=False)
    
    assert 'overall_score' in result
    assert 'llm_analysis' in result
    assert isinstance(result['matched_skills'], list)
    assert isinstance(result['missing_skills'], list)


if __name__ == "__main__":
    print("Running tests...")
    test_keyword_extraction()
    print("✅ test_keyword_extraction passed")
    
    test_skill_extraction()
    print("✅ test_skill_extraction passed")
    
    test_match_score_calculation()
    print("✅ test_match_score_calculation passed")
    
    test_fallback_analysis()
    print("✅ test_fallback_analysis passed")
    
    test_analyze_cv_vs_jd()
    print("✅ test_analyze_cv_vs_jd passed")
    
    print("\n✅ All tests passed!")
