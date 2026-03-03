"""
CV Analysis module - analyzes CV against Job Description.
Implements keyword extraction, matching, scoring, and LLM-based insights.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class CVAnalyzer:
    """Analyzes CV against Job Description using NLP and LLM."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize CV Analyzer.
        
        Args:
            openai_api_key: OpenAI API key for LLM analysis
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.openai_api_key and OpenAI:
            self.client = OpenAI(api_key=self.openai_api_key)
    
    def extract_keywords(self, text: str, top_n: int = 30) -> List[str]:
        """
        Extract important keywords from text using TF-IDF.
        
        Args:
            text: Input text
            top_n: Number of top keywords to extract
            
        Returns:
            List of keywords
        """
        # Common stop words to filter out
        stop_words = [
            'the', 'and', 'to', 'of', 'a', 'in', 'for', 'is', 'on', 'that', 'with',
            'as', 'be', 'at', 'by', 'this', 'from', 'or', 'an', 'are', 'will', 'can',
            'has', 'have', 'had', 'was', 'were', 'been', 'being', 'it', 'its', 'they',
            'their', 'them', 'we', 'our', 'you', 'your', 'he', 'she', 'his', 'her'
        ]
        
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stop words
        words = [w for w in words if w not in stop_words]
        
        # Count frequency
        word_counts = Counter(words)
        
        # Return top N keywords
        return [word for word, count in word_counts.most_common(top_n)]
    
    def extract_skills_and_technologies(self, text: str) -> List[str]:
        """
        Extract technical skills and technologies from text.
        
        Args:
            text: Input text
            
        Returns:
            List of identified skills
        """
        # Common skills/technologies patterns
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|ruby|go|rust|php|swift|kotlin|scala)\b',
            # Frameworks
            r'\b(react|angular|vue|django|flask|fastapi|spring|node\.?js|express|next\.?js)\b',
            # Databases
            r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb)\b',
            # Cloud & DevOps
            r'\b(aws|azure|gcp|docker|kubernetes|jenkins|terraform|ansible|ci/cd)\b',
            # Data Science & ML
            r'\b(machine learning|deep learning|tensorflow|pytorch|scikit-learn|pandas|numpy|nlp)\b',
            # Other tools
            r'\b(git|github|gitlab|jira|agile|scrum|rest api|graphql|microservices)\b'
        ]
        
        skills = set()
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                skills.add(match.group(0))
        
        return list(skills)
    
    def calculate_match_score(self, cv_text: str, jd_text: str) -> Dict:
        """
        Calculate how well CV matches JD.
        
        Args:
            cv_text: CV text content
            jd_text: Job Description text
            
        Returns:
            Dict containing score and details
        """
        # Extract keywords from both
        cv_keywords = set(self.extract_keywords(cv_text, top_n=50))
        jd_keywords = set(self.extract_keywords(jd_text, top_n=50))
        
        # Extract skills
        cv_skills = set(self.extract_skills_and_technologies(cv_text))
        jd_skills = set(self.extract_skills_and_technologies(jd_text))
        
        # Calculate keyword match
        matched_keywords = cv_keywords.intersection(jd_keywords)
        missing_keywords = jd_keywords - cv_keywords
        
        # Calculate skill match
        matched_skills = cv_skills.intersection(jd_skills)
        missing_skills = jd_skills - cv_skills
        
        # Calculate scores
        keyword_score = (len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords else 0
        skill_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
        
        # Calculate text similarity using TF-IDF
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([cv_text, jd_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            similarity_score = similarity * 100
        except:
            similarity_score = 0
        
        # Overall score (weighted average)
        overall_score = (keyword_score * 0.3 + skill_score * 0.4 + similarity_score * 0.3)
        
        return {
            'overall_score': round(overall_score, 2),
            'keyword_score': round(keyword_score, 2),
            'skill_score': round(skill_score, 2),
            'similarity_score': round(similarity_score, 2),
            'matched_keywords': list(matched_keywords),
            'missing_keywords': list(missing_keywords),
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'total_jd_keywords': len(jd_keywords),
            'total_jd_skills': len(jd_skills)
        }
    
    def get_llm_analysis(self, cv_text: str, jd_text: str, match_data: Dict) -> str:
        """
        Get LLM-based analysis and suggestions.
        
        Args:
            cv_text: CV text
            jd_text: Job Description text
            match_data: Matching analysis data
            
        Returns:
            LLM analysis and suggestions
        """
        if not self.client:
            return self._get_fallback_analysis(match_data)
        
        try:
            # Prepare prompt
            prompt = f"""You are an expert HR consultant and career advisor. Analyze the following CV against the Job Description and provide detailed feedback.

Job Description:
{jd_text[:2000]}

CV Content:
{cv_text[:2000]}

Match Analysis:
- Overall Match Score: {match_data['overall_score']}%
- Matched Skills: {', '.join(match_data['matched_skills'][:10]) if match_data['matched_skills'] else 'None'}
- Missing Skills: {', '.join(match_data['missing_skills'][:10]) if match_data['missing_skills'] else 'None'}

Please provide:
1. A brief assessment of how well this CV matches the job requirements
2. Key strengths of the candidate based on the CV
3. Main gaps or areas for improvement
4. Specific suggestions to improve the CV for this role
5. Whether you recommend the candidate for an interview (Yes/No/Maybe)

Keep the response concise (under 300 words).
"""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert HR consultant providing CV feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"LLM Analysis unavailable: {str(e)}\n\n" + self._get_fallback_analysis(match_data)
    
    def _get_fallback_analysis(self, match_data: Dict) -> str:
        """
        Generate basic analysis without LLM.
        
        Args:
            match_data: Matching analysis data
            
        Returns:
            Basic analysis text
        """
        score = match_data['overall_score']
        
        if score >= 70:
            assessment = "Strong match! This CV aligns well with the job requirements."
            recommendation = "Recommended for interview"
        elif score >= 50:
            assessment = "Moderate match. The CV shows some relevant qualifications."
            recommendation = "Consider for interview with reservations"
        else:
            assessment = "Weak match. The CV lacks many required qualifications."
            recommendation = "Not recommended at this time"
        
        matched_skills = match_data['matched_skills'][:5]
        missing_skills = match_data['missing_skills'][:5]
        
        analysis = f"""
**Assessment:** {assessment}

**Strengths:**
{f"- Demonstrates experience with: {', '.join(matched_skills)}" if matched_skills else "- Limited relevant skills identified"}

**Gaps:**
{f"- Missing key skills: {', '.join(missing_skills)}" if missing_skills else "- No major gaps identified"}

**Suggestions:**
1. Highlight relevant projects and achievements
2. {"Add experience with: " + ", ".join(missing_skills[:3]) if missing_skills else "Emphasize existing skills more prominently"}
3. Tailor CV language to match job description
4. Quantify achievements where possible

**Recommendation:** {recommendation}
"""
        
        return analysis.strip()
    
    def analyze_cv_vs_jd(self, cv_text: str, jd_text: str, use_llm: bool = True) -> Dict:
        """
        Complete analysis of CV against Job Description.
        
        Args:
            cv_text: CV text content
            jd_text: Job Description text
            use_llm: Whether to use LLM for analysis
            
        Returns:
            Complete analysis results
        """
        # Calculate match scores
        match_data = self.calculate_match_score(cv_text, jd_text)
        
        # Get LLM analysis if requested
        llm_analysis = ""
        if use_llm:
            llm_analysis = self.get_llm_analysis(cv_text, jd_text, match_data)
        else:
            llm_analysis = self._get_fallback_analysis(match_data)
        
        # Combine all results
        result = {
            **match_data,
            'llm_analysis': llm_analysis
        }
        
        return result
