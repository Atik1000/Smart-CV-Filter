"""
CV Analysis module - analyzes CV against Job Description.
Implements keyword extraction, matching, scoring, and LLM-based insights.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import Counter
import os
from datetime import datetime
from dateutil import parser as date_parser

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from utils.llm_provider import LLMProvider
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class CVAnalyzer:
    """Analyzes CV against Job Description using NLP and LLM."""
    
    def __init__(self, llm_provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize CV Analyzer.
        
        Args:
            llm_provider: LLM provider to use (openai, anthropic, google, groq, ollama)
            api_key: API key for the LLM provider
        """
        self.llm_provider_name = llm_provider
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.llm = None
        
        # Initialize LLM provider
        if LLM_AVAILABLE and (self.api_key or llm_provider == "ollama"):
            try:
                self.llm = LLMProvider(provider=llm_provider, api_key=self.api_key)
            except Exception as e:
                print(f"Could not initialize LLM provider {llm_provider}: {e}")
                self.llm = None
    
    def extract_years_of_experience(self, text: str) -> Dict:
        """
        Extract and calculate total years of experience from CV.
        
        Args:
            text: CV text content
            
        Returns:
            Dict with total years and job history details
        """
        # Common date patterns in CVs
        date_patterns = [
            # "Jan 2020 - Dec 2022", "January 2020 - December 2022"
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*[-–—to]+\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|Present|Current)',
            # "2020 - 2022", "2020-2022"
            r'(\d{4})\s*[-–—to]+\s*(\d{4}|Present|Current)',
            # "01/2020 - 12/2022"
            r'(\d{1,2}/\d{4})\s*[-–—to]+\s*(\d{1,2}/\d{4}|Present|Current)',
        ]
        
        jobs = []
        total_months = 0
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start_str = match.group(1)
                end_str = match.group(2)
                
                try:
                    # Parse start date
                    start_date = date_parser.parse(start_str, fuzzy=True)
                    
                    # Parse end date (handle "Present"/"Current")
                    if end_str.lower() in ['present', 'current']:
                        end_date = datetime.now()
                    else:
                        end_date = date_parser.parse(end_str, fuzzy=True)
                    
                    # Calculate duration in months
                    duration_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                    
                    if duration_months > 0:  # Valid duration
                        jobs.append({
                            'start': start_str,
                            'end': end_str,
                            'duration_months': duration_months
                        })
                        total_months += duration_months
                except:
                    continue
        
        total_years = round(total_months / 12, 1)
        
        return {
            'total_years': total_years,
            'total_months': total_months,
            'jobs_found': len(jobs),
            'job_details': jobs
        }
    
    def extract_required_experience(self, jd_text: str) -> Dict:
        """
        Extract required years of experience from job description.
        
        Args:
            jd_text: Job description text
            
        Returns:
            Dict with minimum and maximum years required
        """
        # Patterns for experience requirements
        exp_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
            r'experience\s*:?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'minimum\s+(\d+)\s+(?:years?|yrs?)',
            r'at\s+least\s+(\d+)\s+(?:years?|yrs?)',
            r'(\d+)[-–](\d+)\s+(?:years?|yrs?)\s+(?:of\s+)?experience',
        ]
        
        years_found = []
        
        for pattern in exp_patterns:
            matches = re.finditer(pattern, jd_text, re.IGNORECASE)
            for match in matches:
                if match.lastindex == 2:  # Range pattern (e.g., "3-5 years")
                    years_found.append(int(match.group(1)))
                    years_found.append(int(match.group(2)))
                else:
                    years_found.append(int(match.group(1)))
        
        if years_found:
            return {
                'min_years': min(years_found),
                'max_years': max(years_found),
                'found': True
            }
        else:
            return {
                'min_years': 0,
                'max_years': 0,
                'found': False
            }
    
    def extract_keywords(self, text: str, top_n: int = 30) -> List[str]:
        """
        Extract job-relevant keywords (skills and qualifications only).
        Focus on technical terms, not generic words.
        
        Args:
            text: Input text
            top_n: Number of top keywords to extract
            
        Returns:
            List of relevant keywords
        """
        # Extract only skills and technologies - no messy generic keywords
        keywords = self.extract_skills_and_technologies(text)
        
        # Add education/qualification keywords
        qualification_patterns = [
            r'\b(bachelor|master|phd|mba|degree|certification|certified)\b',
            r'\b(agile|scrum|devops|cicd|mlops)\b',
            r'\b(leadership|management|team lead|architect|senior|junior)\b',
        ]
        
        text_lower = text.lower()
        for pattern in qualification_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                keywords.append(match.group(0).lower())
        
        # Remove duplicates and return
        return list(set(keywords))[:top_n]
    
    def extract_skills_and_technologies(self, text: str) -> List[str]:
        """
        Extract technical skills and technologies from text.
        Only extracts specific, job-relevant skills - no generic words.
        
        Args:
            text: Input text
            
        Returns:
            List of identified skills
        """
        # Comprehensive skills/technologies patterns - ONLY technical terms
        skill_patterns = [
            # Programming languages
            r'\b(python|java|javascript|typescript|c\+\+|c#|c sharp|ruby|go|golang|rust|php|swift|kotlin|scala|r\b|matlab|perl|shell|bash)\b',
            # Web frameworks
            r'\b(react|angular|vue|django|flask|fastapi|spring|node\.?js|express|next\.?js|nuxt|svelte|blazor|asp\.?net)\b',
            # Mobile
            r'\b(android|ios|react native|flutter|xamarin|ionic)\b',
            # Databases
            r'\b(sql|mysql|postgresql|postgres|mongodb|mongo|redis|elasticsearch|elastic|cassandra|dynamodb|oracle|sql server|sqlite|mariadb|neo4j)\b',
            # Cloud platforms
            r'\b(aws|amazon web services|azure|microsoft azure|gcp|google cloud|heroku|digitalocean|ibm cloud|alibaba cloud)\b',
            # Cloud services
            r'\b(ec2|s3|lambda|rds|cloudformation|cloudwatch|eks|ecs|fargate|api gateway)\b',
            # DevOps & CI/CD
            r'\b(docker|kubernetes|k8s|jenkins|gitlab ci|github actions|circleci|travis ci|ansible|terraform|puppet|chef|vagrant)\b',
            r'\b(ci/cd|cicd|devops|gitops|infrastructure as code|iac)\b',
            # Data Science & ML
            r'\b(machine learning|ml|deep learning|dl|tensorflow|pytorch|keras|scikit-learn|sklearn|pandas|numpy|scipy|jupyter)\b',
            r'\b(nlp|natural language processing|computer vision|cv|neural network|cnn|rnn|lstm|transformer|bert|gpt)\b',
            r'\b(data science|data analysis|data engineering|big data|hadoop|spark|kafka|airflow|mlflow)\b',
            # Version control & collaboration
            r'\b(git|github|gitlab|bitbucket|svn|mercurial|jira|confluence|slack|trello)\b',
            # Methodologies
            r'\b(agile|scrum|kanban|waterfall|lean|safe|xp|tdd|bdd|test driven)\b',
            # API & Architecture
            r'\b(rest|restful|rest api|graphql|grpc|soap|microservices|mvc|mvvm|serverless|event driven)\b',
            # Testing
            r'\b(junit|pytest|jest|mocha|selenium|cypress|testng|unittest|integration test|unit test)\b',
            # Web technologies
            r'\b(html|html5|css|css3|sass|scss|less|webpack|babel|npm|yarn|tailwind|bootstrap|jquery)\b',
            # Backend
            r'\b(api|backend|server|web server|nginx|apache|tomcat|websocket|message queue|rabbitmq|celery)\b',
            # Security
            r'\b(oauth|jwt|ssl|tls|encryption|authentication|authorization|security|cybersecurity|penetration test)\b',
            # Business Intelligence
            r'\b(tableau|power bi|looker|qlik|sap|salesforce|erp|crm)\b',
            # Other important tools
            r'\b(linux|unix|windows server|bash|powershell|vim|vscode|intellij|eclipse|postman|swagger)\b'
        ]
        
        skills = set()
        text_lower = text.lower()
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                skill = match.group(0).strip()
                # Normalize some common variations
                if skill in ['k8s']:
                    skill = 'kubernetes'
                elif skill in ['postgres']:
                    skill = 'postgresql'
                skills.add(skill)
        
        return list(skills)
    
    def calculate_match_score(self, cv_text: str, jd_text: str) -> Dict:
        """
        Calculate how well CV matches JD.
        Now includes years of experience in the scoring.
        
        Args:
            cv_text: CV text content
            jd_text: Job Description text
            
        Returns:
            Dict containing score and details
        """
        # Extract keywords from both (now only skills/technologies)
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
        
        # Extract and compare years of experience
        cv_experience = self.extract_years_of_experience(cv_text)
        jd_experience = self.extract_required_experience(jd_text)
        
        # Calculate experience score
        experience_score = 0
        experience_match = "Not specified"
        
        if jd_experience['found']:
            cv_years = cv_experience['total_years']
            required_years = jd_experience['min_years']
            
            if cv_years >= required_years:
                # Candidate meets or exceeds requirement
                experience_score = 100
                experience_match = f"✓ Meets requirement ({cv_years} >= {required_years} years)"
            elif cv_years >= required_years * 0.7:  # Within 70% of requirement
                # Close to requirement
                experience_score = 70
                experience_match = f"⚠ Close to requirement ({cv_years} vs {required_years} years required)"
            else:
                # Below requirement
                experience_score = (cv_years / required_years) * 50  # Max 50% if below requirement
                experience_match = f"✗ Below requirement ({cv_years} vs {required_years} years required)"
        else:
            # No experience requirement in JD
            experience_score = 100  # Don't penalize if not required
            experience_match = f"No specific requirement (Candidate has {cv_experience['total_years']} years)"
        
        # Overall score (weighted average including experience)
        # Skills: 40%, Experience: 30%, Similarity: 20%, Keywords: 10%
        overall_score = (
            skill_score * 0.40 + 
            experience_score * 0.30 + 
            similarity_score * 0.20 + 
            keyword_score * 0.10
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'skill_score': round(skill_score, 2),
            'experience_score': round(experience_score, 2),
            'similarity_score': round(similarity_score, 2),
            'keyword_score': round(keyword_score, 2),
            'matched_skills': list(matched_skills),
            'missing_skills': list(missing_skills),
            'matched_keywords': list(matched_keywords),
            'missing_keywords': list(missing_keywords),
            'total_jd_skills': len(jd_skills),
            'total_jd_keywords': len(jd_keywords),
            'cv_years_experience': cv_experience['total_years'],
            'cv_jobs_found': cv_experience['jobs_found'],
            'jd_required_years': jd_experience['min_years'] if jd_experience['found'] else 0,
            'experience_match': experience_match,
            'experience_details': cv_experience
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
        if not self.llm:
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
- Skills Match: {match_data['skill_score']}%
- Experience Match: {match_data['experience_score']}%
- Candidate Experience: {match_data.get('cv_years_experience', 0)} years
- Required Experience: {match_data.get('jd_required_years', 0)} years
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
            
            # Use LLM provider
            response = self.llm.generate_analysis(prompt, max_tokens=500)
            return response
            
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
