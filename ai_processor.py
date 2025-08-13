import streamlit as st
import openai
from typing import Dict, Any
import json
import re

class AIProcessor:
    def __init__(self):
        self.client = openai.OpenAI(
            api_key=st.secrets["openai"]["api_key"]
        )
        self.model = st.secrets["openai"]["model"]
    
    def analyze_email(self, email: Dict[str, Any]) -> Dict[str, str]:
        """Analyze email using OpenAI to extract categorization and summary"""
        
        try:
            # Extract email content
            subject = email.get('subject', '')
            sender = email.get('from', {}).get('emailAddress', {}).get('name', '')
            sender_email = email.get('from', {}).get('emailAddress', {}).get('address', '')
            body_preview = email.get('bodyPreview', '')
            importance = email.get('importance', 'normal')
            
            # Create prompt for AI analysis[7][10][13]
            prompt = f"""
            Analyze the following email and provide a structured response in JSON format:

            Subject: {subject}
            From: {sender} ({sender_email})
            Body Preview: {body_preview}
            Outlook Importance: {importance}

            Please categorize this email and provide the following information in JSON format:

            {{
                "type": "Official/Personal/Spam/Advertisement",
                "importance": "High/Medium/Low/Not important", 
                "summary": "2-3 sentence summary of the email content",
                "response_required": "Yes/No/Maybe",
                "urgency": "Immediate/by today/Anytime/Not urgent"
            }}

            Guidelines:
            - Type: Official (work/business), Personal (friends/family), Spam (unwanted), Advertisement (marketing)
            - Importance: Based on content urgency and sender relevance
            - Summary: Concise but informative summary
            - Response Required: Whether the email needs a reply
            - Urgency: Time sensitivity of the email

            Respond only with valid JSON, no additional text.
            """
            
            # Call OpenAI API[19]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert email analyst. Provide accurate categorization and summaries in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse response
            ai_response = response.choices[0].message.content.strip()
            
            # Clean up response to ensure valid JSON
            ai_response = self._clean_json_response(ai_response)
            
            try:
                analysis = json.loads(ai_response)
                
                # Validate and set defaults if needed
                analysis = self._validate_analysis(analysis)
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return self._create_fallback_analysis(subject, sender_email, body_preview)
                
        except Exception as e:
            st.error(f"Error analyzing email: {str(e)}")
            return self._create_fallback_analysis(subject, sender_email, body_preview)
    
    def _clean_json_response(self, response: str) -> str:
        """Clean AI response to ensure valid JSON"""
        # Remove any text before the first {
        start_idx = response.find('{')
        if start_idx != -1:
            response = response[start_idx:]
        
        # Remove any text after the last }
        end_idx = response.rfind('}')
        if end_idx != -1:
            response = response[:end_idx + 1]
        
        return response
    
    def _validate_analysis(self, analysis: Dict[str, str]) -> Dict[str, str]:
        """Validate and correct AI analysis results"""
        
        # Valid options
        valid_types = ["Official", "Personal", "Spam", "Advertisement"]
        valid_importance = ["High", "Medium", "Low", "Not important"]
        valid_response = ["Yes", "No", "Maybe"]
        valid_urgency = ["Immediate", "by today", "Anytime", "Not urgent"]
        
        # Validate and correct values
        if analysis.get('type') not in valid_types:
            analysis['type'] = "Official"  # Default
        
        if analysis.get('importance') not in valid_importance:
            analysis['importance'] = "Medium"  # Default
        
        if analysis.get('response_required') not in valid_response:
            analysis['response_required'] = "Maybe"  # Default
        
        if analysis.get('urgency') not in valid_urgency:
            analysis['urgency'] = "Anytime"  # Default
        
        # Ensure summary exists
        if not analysis.get('summary'):
            analysis['summary'] = "No summary available"
        
        return analysis
    
    def _create_fallback_analysis(self, subject: str, sender_email: str, body_preview: str) -> Dict[str, str]:
        """Create fallback analysis when AI processing fails"""
        
        # Simple rule-based categorization
        email_type = "Official"
        if any(keyword in subject.lower() for keyword in ['sale', 'offer', 'discount', 'deal']):
            email_type = "Advertisement"
        elif any(domain in sender_email.lower() for domain in ['gmail.com', 'yahoo.com', 'hotmail.com']):
            email_type = "Personal"
        
        return {
            "type": email_type,
            "importance": "Medium",
            "summary": body_preview[:200] + "..." if len(body_preview) > 200 else body_preview,
            "response_required": "Maybe",
            "urgency": "Anytime"
        }
