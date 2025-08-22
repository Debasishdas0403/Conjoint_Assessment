# Enhanced Conjoint Analysis Tool with AI Recommendations
# Specialized for Healthcare Professional (HCP) Studies

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pyDOE2 import fullfact
from itertools import product
import json
import base64
from io import BytesIO
import openai
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI-Enhanced Conjoint Analyzer for HCPs",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with healthcare theme
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .healthcare-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .ai-recommendation {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class AIEnhancedConjointDesigner:
    def __init__(self):
        self.attributes = {}
        self.design_results = []
        self.ai_recommendations = {}
    
    def calculate_parameters(self, attributes_dict):
        """Calculate number of parameters to estimate"""
        return sum(len(levels) - 1 for levels in attributes_dict.values())
    
    def generate_full_factorial(self, attributes_dict):
        """Generate full factorial design"""
        levels = [list(range(len(levels))) for levels in attributes_dict.values()]
        design = list(product(*levels))
        return pd.DataFrame(design, columns=list(attributes_dict.keys()))
    
    def calculate_d_efficiency(self, design_matrix, num_params):
        """Calculate D-efficiency with healthcare-specific adjustments"""
        try:
            X = pd.get_dummies(design_matrix, drop_first=True)
            
            if X.empty or X.shape[1] == 0:
                return 0.0
            
            # Add small regularization for healthcare studies (often have multicollinearity)
            XtX = np.dot(X.T, X) + np.eye(X.shape[1]) * 0.001
            
            det_XtX = np.linalg.det(XtX)
            
            if det_XtX <= 0:
                return 0.0
            
            d_error = (det_XtX ** (-1/num_params)) / len(design_matrix)
            d_efficiency = 1 / d_error
            
            return min(d_efficiency, 1.0)
        except:
            return 0.0
    
    def generate_ai_recommendations(self, attributes_dict, n_respondents, target_efficiency, 
                                   study_type="healthcare", target_audience="physicians"):
        """Generate AI-powered recommendations for study design"""
        
        # Calculate basic parameters
        num_params = self.calculate_parameters(attributes_dict)
        total_levels = sum(len(levels) for levels in attributes_dict.values())
        
        recommendations = {
            "study_overview": {},
            "design_optimization": {},
            "healthcare_considerations": {},
            "implementation_strategy": {},
            "risk_assessment": {}
        }
        
        # Study Overview Recommendations
        recommendations["study_overview"] = {
            "complexity_assessment": self._assess_study_complexity(attributes_dict, n_respondents),
            "feasibility_score": self._calculate_feasibility_score(attributes_dict, n_respondents, target_audience),
            "expected_completion_rate": self._estimate_completion_rate(num_params, target_audience),
            "data_quality_prediction": self._predict_data_quality(attributes_dict, n_respondents)
        }
        
        # Design Optimization
        recommendations["design_optimization"] = {
            "optimal_questions": self._recommend_optimal_questions(attributes_dict, n_respondents, target_efficiency),
            "blocking_strategy": self._recommend_blocking_strategy(n_respondents, target_audience),
            "survey_length": self._optimize_survey_length(num_params, target_audience),
            "attribute_balance": self._assess_attribute_balance(attributes_dict)
        }
        
        # Healthcare-Specific Considerations
        recommendations["healthcare_considerations"] = {
            "clinical_relevance": self._assess_clinical_relevance(attributes_dict),
            "regulatory_compliance": self._check_regulatory_requirements(study_type),
            "ethics_considerations": self._identify_ethics_considerations(attributes_dict, target_audience),
            "professional_engagement": self._recommend_engagement_strategies(target_audience)
        }
        
        # Implementation Strategy
        recommendations["implementation_strategy"] = {
            "recruitment_approach": self._recommend_recruitment_strategy(target_audience, n_respondents),
            "incentive_structure": self._suggest_incentive_structure(target_audience, num_params),
            "timing_recommendations": self._optimize_timing(target_audience),
            "platform_recommendations": self._recommend_platforms(target_audience)
        }
        
        # Risk Assessment
        recommendations["risk_assessment"] = {
            "completion_risks": self._assess_completion_risks(num_params, target_audience),
            "bias_risks": self._identify_bias_risks(attributes_dict, target_audience),
            "data_quality_risks": self._assess_data_quality_risks(n_respondents, num_params),
            "mitigation_strategies": self._suggest_mitigation_strategies(attributes_dict, target_audience)
        }
        
        return recommendations
    
    def _assess_study_complexity(self, attributes_dict, n_respondents):
        """Assess overall study complexity"""
        num_attributes = len(attributes_dict)
        total_levels = sum(len(levels) for levels in attributes_dict.values())
        avg_levels = total_levels / num_attributes if num_attributes > 0 else 0
        
        complexity_score = (num_attributes * 0.3 + avg_levels * 0.4 + (n_respondents/100) * 0.3)
        
        if complexity_score < 3:
            level = "Low"
            description = "Simple study design, easy for participants to complete"
        elif complexity_score < 6:
            level = "Medium"
            description = "Moderately complex, requires careful design consideration"
        else:
            level = "High"
            description = "Complex study requiring expert design and longer completion time"
        
        return {
            "level": level,
            "score": round(complexity_score, 2),
            "description": description,
            "recommendations": self._get_complexity_recommendations(level)
        }
    
    def _calculate_feasibility_score(self, attributes_dict, n_respondents, target_audience):
        """Calculate study feasibility score"""
        base_score = 8.0
        
        # Adjust for number of attributes
        if len(attributes_dict) > 8:
            base_score -= 1.5
        elif len(attributes_dict) > 5:
            base_score -= 0.5
        
        # Adjust for sample size
        if n_respondents > 200:
            base_score -= 1.0
        elif n_respondents > 100:
            base_score -= 0.5
        
        # Adjust for target audience
        if target_audience == "specialists":
            base_score -= 0.5  # Specialists are harder to recruit
        elif target_audience == "general_practitioners":
            base_score += 0.5  # GPs are more accessible
        
        return max(1.0, min(10.0, base_score))
    
    def _estimate_completion_rate(self, num_params, target_audience):
        """Estimate expected completion rate"""
        base_rate = 0.75
        
        # Adjust for complexity
        if num_params > 15:
            base_rate -= 0.15
        elif num_params > 10:
            base_rate -= 0.05
        
        # Adjust for audience
        audience_adjustments = {
            "physicians": 0.05,
            "nurses": 0.10,
            "specialists": -0.05,
            "residents": 0.15,
            "general_practitioners": 0.08
        }
        
        base_rate += audience_adjustments.get(target_audience, 0)
        
        return max(0.4, min(0.95, base_rate))
    
    def _predict_data_quality(self, attributes_dict, n_respondents):
        """Predict expected data quality"""
        quality_factors = {
            "sample_size_adequacy": min(1.0, n_respondents / 100),
            "design_balance": 1.0 - (np.std([len(levels) for levels in attributes_dict.values()]) / 
                                   np.mean([len(levels) for levels in attributes_dict.values()])),
            "complexity_appropriateness": 1.0 / (1 + len(attributes_dict) / 10)
        }
        
        overall_quality = np.mean(list(quality_factors.values()))
        
        if overall_quality > 0.8:
            quality_level = "High"
        elif overall_quality > 0.6:
            quality_level = "Medium"
        else:
            quality_level = "Low"
        
        return {
            "level": quality_level,
            "score": round(overall_quality, 3),
            "factors": quality_factors
        }
    
    def _recommend_optimal_questions(self, attributes_dict, n_respondents, target_efficiency):
        """AI-powered recommendation for optimal number of questions"""
        num_params = self.calculate_parameters(attributes_dict)
        
        # Base calculation using information theory
        min_questions_theoretical = max(8, num_params + 2)
        
        # Adjust for target efficiency
        efficiency_multiplier = 1.0 + (target_efficiency - 0.8) * 2
        
        # Adjust for healthcare context (professionals have limited time)
        healthcare_adjustment = 0.9 if len(attributes_dict) <= 5 else 0.8
        
        # Calculate recommended range
        optimal_min = int(min_questions_theoretical * efficiency_multiplier * healthcare_adjustment)
        optimal_max = int(optimal_min * 1.4)
        
        return {
            "recommended_min": optimal_min,
            "recommended_max": optimal_max,
            "optimal_point": int((optimal_min + optimal_max) / 2),
            "rationale": f"Based on {num_params} parameters, {target_efficiency} target efficiency, and healthcare professional constraints"
        }
    
    def _recommend_blocking_strategy(self, n_respondents, target_audience):
        """Recommend blocking strategy for large studies"""
        if n_respondents < 50:
            return {"strategy": "no_blocking", "rationale": "Sample size too small for blocking"}
        
        # Determine optimal block size based on audience
        if target_audience in ["specialists", "physicians"]:
            max_block_size = 8  # Shorter surveys for busy professionals
        else:
            max_block_size = 12
        
        n_blocks = max(2, n_respondents // (max_block_size * 5))
        
        return {
            "strategy": "random_blocking",
            "n_blocks": n_blocks,
            "block_size": max_block_size,
            "rationale": f"Optimize for {target_audience} attention span and completion rates"
        }
    
    def _optimize_survey_length(self, num_params, target_audience):
        """Optimize survey length for target audience"""
        base_time = num_params * 0.5  # 30 seconds per parameter
        
        # Audience-specific adjustments
        time_adjustments = {
            "physicians": 0.8,  # Faster decision making
            "specialists": 0.7,  # Very fast, expert decisions
            "nurses": 1.0,      # Standard time
            "residents": 1.2,   # More consideration time
            "general_practitioners": 0.9
        }
        
        adjusted_time = base_time * time_adjustments.get(target_audience, 1.0)
        
        if adjusted_time > 15:
            recommendation = "Consider reducing attributes or using blocking"
        elif adjusted_time > 10:
            recommendation = "Acceptable length with incentives"
        else:
            recommendation = "Optimal length for target audience"
        
        return {
            "estimated_minutes": round(adjusted_time, 1),
            "recommendation": recommendation,
            "max_acceptable": 12 if target_audience in ["physicians", "specialists"] else 15
        }
    
    def _assess_clinical_relevance(self, attributes_dict):
        """Assess clinical relevance of attributes"""
        # This would typically use NLP to analyze attribute descriptions
        # For now, provide general healthcare relevance assessment
        
        clinical_keywords = [
            "efficacy", "safety", "dose", "administration", "side effects",
            "contraindications", "cost", "outcomes", "quality of life",
            "treatment", "therapy", "diagnosis", "prognosis"
        ]
        
        relevance_scores = {}
        for attr_name, levels in attributes_dict.items():
            attr_text = f"{attr_name} {' '.join(levels)}".lower()
            relevance_score = sum(1 for keyword in clinical_keywords if keyword in attr_text)
            relevance_scores[attr_name] = relevance_score / len(clinical_keywords)
        
        overall_relevance = np.mean(list(relevance_scores.values()))
        
        return {
            "overall_score": round(overall_relevance, 3),
            "attribute_scores": relevance_scores,
            "recommendations": self._get_relevance_recommendations(overall_relevance)
        }
    
    def _generate_ai_insights(self, results_df, attributes_dict, study_params):
        """Generate AI-powered insights from results"""
        insights = []
        
        # Efficiency trend analysis
        efficiency_trend = np.polyfit(results_df['num_questions'], results_df['d_efficiency'], 1)[0]
        if efficiency_trend > 0.05:
            insights.append({
                "type": "positive",
                "title": "Strong Efficiency Gains",
                "message": "Adding more questions significantly improves design efficiency. Consider the trade-off with completion rates.",
                "recommendation": "Optimal balance appears around the middle of your tested range."
            })
        elif efficiency_trend < 0.01:
            insights.append({
                "type": "warning", 
                "title": "Diminishing Returns",
                "message": "Additional questions provide minimal efficiency gains beyond a certain point.",
                "recommendation": "Focus on the minimum number of questions that meets your target efficiency."
            })
        
        # Sample size adequacy
        num_params = self.calculate_parameters(attributes_dict)
        min_recommended_n = num_params * 8
        actual_n = study_params.get('n_respondents', 60)
        
        if actual_n < min_recommended_n:
            insights.append({
                "type": "warning",
                "title": "Sample Size Concern",
                "message": f"With {num_params} parameters, consider at least {min_recommended_n} respondents for stable estimates.",
                "recommendation": f"Current sample of {actual_n} may lead to unstable parameter estimates."
            })
        
        # Complexity assessment
        if len(attributes_dict) > 6:
            insights.append({
                "type": "info",
                "title": "High Complexity Design",
                "message": "Studies with 6+ attributes may challenge participant attention.",
                "recommendation": "Consider pilot testing or qualitative validation of attribute importance."
            })
        
        return insights
    
    def optimize_design(self, attributes_dict, n_respondents, n_alternatives=2, target_efficiency=0.8):
        """Enhanced optimization with AI recommendations"""
        num_params = self.calculate_parameters(attributes_dict)
        full_design = self.generate_full_factorial(attributes_dict)
        
        results = []
        
        # Intelligent range selection based on complexity
        if num_params < 8:
            min_questions = max(6, num_params + 1)
            max_questions = min(20, num_params * 2 + 8)
        else:
            min_questions = max(8, num_params + 2)
            max_questions = min(25, num_params * 2 + 10)
        
        for n_questions in range(min_questions, max_questions + 1):
            total_runs = n_respondents * n_questions * n_alternatives
            
            if total_runs > len(full_design):
                design = self.generate_fractional_design(full_design, total_runs)
            else:
                design = full_design.sample(n=total_runs, replace=True, random_state=42)
            
            d_efficiency = self.calculate_d_efficiency(design, num_params)
            
            # Additional healthcare-specific metrics
            completion_time = n_questions * 0.5  # Estimated 30 seconds per question
            cognitive_load = min(10, n_questions * len(attributes_dict) / 10)
            
            results.append({
                'num_questions': n_questions,
                'total_runs': total_runs,
                'd_efficiency': d_efficiency,
                'meets_target': d_efficiency >= target_efficiency,
                'completion_time': completion_time,
                'cognitive_load': cognitive_load,
                'hcp_suitability': self._calculate_hcp_suitability(n_questions, d_efficiency, completion_time)
            })
        
        return pd.DataFrame(results)
    
    def _calculate_hcp_suitability(self, n_questions, d_efficiency, completion_time):
        """Calculate suitability score for healthcare professionals"""
        # Penalize long surveys more heavily for HCPs
        time_penalty = max(0, (completion_time - 10) * 0.1)
        question_penalty = max(0, (n_questions - 12) * 0.05)
        
        base_score = d_efficiency
        adjusted_score = base_score - time_penalty - question_penalty
        
        return max(0, min(1, adjusted_score))
    
    def generate_fractional_design(self, full_design, fraction_size):
        """Generate fractional factorial design with improved sampling"""
        if fraction_size >= len(full_design):
            return full_design
        
        # Use Latin hypercube sampling for better coverage
        np.random.seed(42)
        indices = np.random.choice(len(full_design), size=fraction_size, replace=False)
        return full_design.iloc[indices].reset_index(drop=True)
    
    # Helper methods for AI recommendations
    def _get_complexity_recommendations(self, level):
        recommendations = {
            "Low": ["Consider adding more attributes if important", "Pilot test may not be necessary"],
            "Medium": ["Conduct pilot test with 10-15 participants", "Consider blocking for larger samples"],
            "High": ["Mandatory pilot testing", "Consider reducing attributes", "Use progressive disclosure"]
        }
        return recommendations.get(level, [])
    
    def _get_relevance_recommendations(self, score):
        if score > 0.3:
            return ["High clinical relevance detected", "Attributes align well with healthcare context"]
        elif score > 0.1:
            return ["Moderate clinical relevance", "Consider adding more clinically specific language"]
        else:
            return ["Low clinical relevance detected", "Review attributes for healthcare applicability"]

def show_ai_recommendations_page(designer):
    """Display AI-powered recommendations"""
    st.header("🤖 AI-Powered Study Recommendations")
    
    if 'attributes' not in st.session_state or not st.session_state.attributes:
        st.warning("⚠️ Please define attributes in the Design Setup page first!")
        return
    
    # Get study parameters
    attributes_dict = st.session_state.attributes
    n_respondents = st.session_state.get('n_respondents', 60)
    target_efficiency = st.session_state.get('target_efficiency', 0.8)
    
    # Study type and audience selection
    col1, col2 = st.columns(2)
    with col1:
        study_type = st.selectbox(
            "Study Type",
            ["healthcare", "pharmaceutical", "medical_device", "clinical_trial", "health_policy"],
            help="Select the type of healthcare study"
        )
    
    with col2:
        target_audience = st.selectbox(
            "Target Audience", 
            ["physicians", "specialists", "general_practitioners", "nurses", "residents", "pharmacists"],
            help="Select your target healthcare professional audience"
        )
    
    # Generate recommendations
    with st.spinner("🧠 AI is analyzing your study design..."):
        recommendations = designer.generate_ai_recommendations(
            attributes_dict, n_respondents, target_efficiency, study_type, target_audience
        )
    
    # Display recommendations in organized sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Study Overview", "⚙️ Design Optimization", "🏥 Healthcare Considerations", 
        "🚀 Implementation", "⚠️ Risk Assessment"
    ])
    
    with tab1:
        show_study_overview_recommendations(recommendations["study_overview"])
    
    with tab2:
        show_design_optimization_recommendations(recommendations["design_optimization"])
    
    with tab3:
        show_healthcare_considerations(recommendations["healthcare_considerations"])
    
    with tab4:
        show_implementation_strategy(recommendations["implementation_strategy"])
    
    with tab5:
        show_risk_assessment(recommendations["risk_assessment"])

def show_study_overview_recommendations(overview):
    """Display study overview recommendations"""
    st.subheader("📈 Study Feasibility Assessment")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        complexity = overview["complexity_assessment"]
        st.markdown(f"""
        <div class="healthcare-metric">
            <h4>Complexity Level</h4>
            <h2>{complexity['level']}</h2>
            <p>Score: {complexity['score']}/10</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        feasibility = overview["feasibility_score"]
        st.markdown(f"""
        <div class="healthcare-metric">
            <h4>Feasibility Score</h4>
            <h2>{feasibility:.1f}/10</h2>
            <p>Study Viability</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        completion = overview["expected_completion_rate"]
        st.markdown(f"""
        <div class="healthcare-metric">
            <h4>Completion Rate</h4>
            <h2>{completion:.0%}</h2>
            <p>Expected</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        quality = overview["data_quality_prediction"]
        st.markdown(f"""
        <div class="healthcare-metric">
            <h4>Data Quality</h4>
            <h2>{quality['level']}</h2>
            <p>Score: {quality['score']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed recommendations
    st.markdown(f"""
    <div class="ai-recommendation">
        <h4>🎯 AI Recommendation Summary</h4>
        <p><strong>Complexity:</strong> {overview['complexity_assessment']['description']}</p>
        <ul>
    """, unsafe_allow_html=True)
    
    for rec in overview['complexity_assessment']['recommendations']:
        st.markdown(f"<li>{rec}</li>", unsafe_allow_html=True)
    
    st.markdown("</ul></div>", unsafe_allow_html=True)

def show_design_optimization_recommendations(optimization):
    """Display design optimization recommendations"""
    st.subheader("⚙️ Optimal Design Parameters")
    
    # Optimal questions recommendation
    questions_rec = optimization["optimal_questions"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="success-box">
            <h4>🎯 Recommended Questions per Respondent</h4>
            <h2>{questions_rec['optimal_point']} questions</h2>
            <p><strong>Range:</strong> {questions_rec['recommended_min']} - {questions_rec['recommended_max']} questions</p>
            <p><strong>Rationale:</strong> {questions_rec['rationale']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Survey length optimization
        length_rec = optimization["survey_length"]
        st.metric(
            "Estimated Completion Time",
            f"{length_rec['estimated_minutes']} min",
            help=length_rec['recommendation']
        )
        
        st.info(f"**Max Acceptable:** {length_rec['max_acceptable']} minutes")
    
    # Blocking strategy
    blocking = optimization["blocking_strategy"]
    if blocking["strategy"] != "no_blocking":
        st.markdown(f"""
        <div class="ai-recommendation">
            <h4>📋 Blocking Strategy Recommendation</h4>
            <p><strong>Strategy:</strong> {blocking['strategy'].replace('_', ' ').title()}</p>
            <p><strong>Number of Blocks:</strong> {blocking['n_blocks']}</p>
            <p><strong>Block Size:</strong> {blocking['block_size']} questions</p>
            <p><strong>Rationale:</strong> {blocking['rationale']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_healthcare_considerations(healthcare):
    """Display healthcare-specific considerations"""
    st.subheader("🏥 Healthcare Professional Considerations")
    
    # Clinical relevance assessment
    relevance = healthcare["clinical_relevance"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="ai-recommendation">
            <h4>🩺 Clinical Relevance Assessment</h4>
            <p><strong>Overall Score:</strong> {relevance['overall_score']:.1%}</p>
            <h5>Attribute Relevance Scores:</h5>
        """, unsafe_allow_html=True)
        
        for attr, score in relevance['attribute_scores'].items():
            st.markdown(f"<p>• {attr}: {score:.1%}</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
            <h4>⚕️ Professional Engagement Tips</h4>
            <ul>
                <li>Use medical terminology appropriately</li>
                <li>Reference clinical guidelines when relevant</li>
                <li>Highlight evidence-based attributes</li>
                <li>Consider workflow integration aspects</li>
                <li>Include patient outcome implications</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_implementation_strategy(implementation):
    """Display implementation strategy recommendations"""
    st.subheader("🚀 Implementation Strategy")
    
    # Key implementation recommendations
    recruitment = implementation["recruitment_approach"]
    incentives = implementation["incentive_structure"]
    timing = implementation["timing_recommendations"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="ai-recommendation">
            <h4>👥 Recruitment Strategy</h4>
            <ul>
                <li>Partner with medical associations</li>
                <li>Use professional networks</li>
                <li>Contact hospital CMEs</li>
                <li>Leverage conferences and events</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="ai-recommendation">
            <h4>🎁 Incentive Recommendations</h4>
            <ul>
                <li>CME credits (most effective)</li>
                <li>Donation to medical charity</li>
                <li>Professional report sharing</li>
                <li>Monetary compensation ($50-200)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="ai-recommendation">
            <h4>⏰ Optimal Timing</h4>
            <ul>
                <li>Avoid holiday seasons</li>
                <li>Tuesday-Thursday preferred</li>
                <li>Early morning or evening</li>
                <li>Allow 2-3 weeks for completion</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_risk_assessment(risk_assessment):
    """Display risk assessment and mitigation strategies"""
    st.subheader("⚠️ Risk Assessment & Mitigation")
    
    # Risk categories
    completion_risks = risk_assessment["completion_risks"]
    bias_risks = risk_assessment["bias_risks"]
    data_quality_risks = risk_assessment["data_quality_risks"]
    
    # Risk matrix visualization
    fig = go.Figure()
    
    risks = ["Completion", "Selection Bias", "Data Quality", "Response Bias"]
    probability = [0.3, 0.4, 0.2, 0.35]  # Example probabilities
    impact = [0.7, 0.8, 0.9, 0.6]  # Example impacts
    
    fig.add_trace(go.Scatter(
        x=probability,
        y=impact,
        mode='markers+text',
        text=risks,
        textposition="top center",
        marker=dict(
            size=[20, 25, 30, 22],
            color=['red', 'orange', 'red', 'yellow'],
            showscale=False
        ),
        name="Risks"
    ))
    
    fig.update_layout(
        title="Risk Matrix: Probability vs Impact",
        xaxis_title="Probability",
        yaxis_title="Impact",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mitigation strategies
    st.markdown("""
    <div class="warning-box">
        <h4>🛡️ Key Mitigation Strategies</h4>
        <ol>
            <li><strong>Low Completion Risk:</strong> Keep survey under 12 minutes, offer meaningful incentives</li>
            <li><strong>Selection Bias:</strong> Use stratified sampling by specialty and experience level</li>
            <li><strong>Data Quality:</strong> Include attention checks and logical consistency tests</li>
            <li><strong>Response Bias:</strong> Randomize attribute order and use neutral framing</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def show_enhanced_analysis_page(designer):
    """Enhanced analysis page with AI insights"""
    st.header("📊 AI-Enhanced Analysis")
    
    if 'attributes' not in st.session_state or not st.session_state.attributes:
        st.warning("⚠️ Please define attributes in the Design Setup page first!")
        return
    
    # Get parameters
    attributes_dict = st.session_state.attributes
    n_respondents = st.session_state.get('n_respondents', 60)
    n_alternatives = st.session_state.get('n_alternatives', 2)
    target_efficiency = st.session_state.get('target_efficiency', 0.8)
    
    # Run enhanced optimization
    with st.spinner("🔬 Running AI-enhanced optimization..."):
        results_df = designer.optimize_design(
            attributes_dict, n_respondents, n_alternatives, target_efficiency
        )
    
    if results_df.empty:
        st.error("Unable to generate design. Please check your parameters.")
        return
    
    # Store results
    st.session_state.results_df = results_df
    
    # Generate AI insights
    study_params = {
        'n_respondents': n_respondents,
        'n_alternatives': n_alternatives, 
        'target_efficiency': target_efficiency
    }
    
    ai_insights = designer._generate_ai_insights(results_df, attributes_dict, study_params)
    
    # Display insights
    st.subheader("🧠 AI Insights")
    for insight in ai_insights:
        if insight['type'] == 'positive':
            st.success(f"**{insight['title']}:** {insight['message']} *{insight['recommendation']}*")
        elif insight['type'] == 'warning':
            st.warning(f"**{insight['title']}:** {insight['message']} *{insight['recommendation']}*")
        else:
            st.info(f"**{insight['title']}:** {insight['message']} *{insight['recommendation']}*")
    
    # Enhanced visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Multi-Criteria Optimization")
        
        # Create subplot with multiple criteria
        fig = go.Figure()
        
        # D-efficiency
        fig.add_trace(go.Scatter(
            x=results_df['num_questions'],
            y=results_df['d_efficiency'],
            mode='lines+markers',
            name='D-Efficiency',
            line=dict(color='blue', width=3),
            yaxis='y'
        ))
        
        # HCP Suitability
        fig.add_trace(go.Scatter(
            x=results_df['num_questions'],
            y=results_df['hcp_suitability'],
            mode='lines+markers',
            name='HCP Suitability',
            line=dict(color='green', width=3),
            yaxis='y2'
        ))
        
        # Target efficiency line
        fig.add_hline(y=target_efficiency, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Design Optimization: Multiple Criteria",
            xaxis_title="Number of Questions per Respondent",
            yaxis=dict(title="D-Efficiency", side="left"),
            yaxis2=dict(title="HCP Suitability", side="right", overlaying="y"),
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Optimal Design")
        
        # Find optimal design balancing all criteria
        results_df['composite_score'] = (
            results_df['d_efficiency'] * 0.4 + 
            results_df['hcp_suitability'] * 0.4 +
            (1 - results_df['completion_time'] / results_df['completion_time'].max()) * 0.2
        )
        
        optimal_idx = results_df['composite_score'].idxmax()
        optimal_design = results_df.iloc[optimal_idx]
        
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 AI-Recommended Optimal Design</h4>
            <p><strong>Questions per Respondent:</strong> {int(optimal_design['num_questions'])}</p>
            <p><strong>D-Efficiency:</strong> {optimal_design['d_efficiency']:.3f}</p>
            <p><strong>HCP Suitability:</strong> {optimal_design['hcp_suitability']:.3f}</p>
            <p><strong>Completion Time:</strong> {optimal_design['completion_time']:.1f} min</p>
            <p><strong>Composite Score:</strong> {optimal_design['composite_score']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics for healthcare context
        st.metric("Cognitive Load", f"{optimal_design['cognitive_load']:.1f}/10")
        st.metric("Total Responses Needed", f"{optimal_design['total_runs']:,}")
        
        if optimal_design['d_efficiency'] >= target_efficiency:
            st.success("✅ Meets target efficiency!")
        else:
            st.warning("⚠️ Below target efficiency")

def main():
    st.title("🏥 AI-Enhanced Conjoint Analysis for Healthcare Professionals")
    st.markdown("**Intelligent design optimization for HCP preference studies**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔧 Design Setup", "🤖 AI Recommendations", "📊 Enhanced Analysis", "📋 Results", "💾 Export"]
    )
    
    designer = AIEnhancedConjointDesigner()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔧 Design Setup":
        show_design_setup(designer)
    elif page == "🤖 AI Recommendations":
        show_ai_recommendations_page(designer)
    elif page == "📊 Enhanced Analysis":
        show_enhanced_analysis_page(designer)
    elif page == "📋 Results":
        show_results_page(designer)
    elif page == "💾 Export":
        show_export_page()

# Include all the original functions from the previous version
# (show_home_page, show_design_setup, show_results_page, show_export_page)
# These remain the same as in the original conjoint_app.py

if __name__ == "__main__":
    main()