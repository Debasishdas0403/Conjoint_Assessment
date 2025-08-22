import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import openai
from scipy.optimize import minimize
import warnings
import math
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Conjoint Demand Estimation Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🎯 Conjoint Demand Estimation Analysis Tool")
st.markdown("""
This tool helps you design and analyze conjoint choice experiments for demand estimation.
It calculates D-efficiency and provides GPT-powered recommendations for optimal survey design.
""")

# Initialize OpenAI client from secrets
def get_openai_client():
    """Get OpenAI client using secrets"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return openai.OpenAI(api_key=api_key)
    except KeyError:
        return None
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

# Check if OpenAI is available
openai_available = get_openai_client() is not None

# Sidebar for inputs
st.sidebar.header("📋 Survey Design Parameters")

class ConjointAnalyzer:
    def __init__(self):
        self.attributes = {}
        self.profiles = None
        self.num_params = 0
        
    def generate_profiles(self, attributes):
        """Generate all possible product profiles"""
        self.attributes = attributes
        attribute_names = list(attributes.keys())
        level_combinations = list(product(*attributes.values()))
        
        profiles_df = pd.DataFrame(level_combinations, columns=attribute_names)
        self.profiles = profiles_df
        
        # Calculate number of parameters (sum of levels - 1 for each attribute)
        self.num_params = sum(len(levels) - 1 for levels in attributes.values())
        
        return profiles_df
    
    def calculate_d_efficiency(self, n_questions, n_respondents=60, n_alternatives=2):
        """
        More realistic D-efficiency calculation based on conjoint analysis principles
        """
        if n_questions == 0:
            return 0, float('inf')
        
        # Total choice observations
        total_choices = n_questions * n_respondents
        
        # For conjoint analysis, we need sufficient observations per parameter
        # Rule of thumb: at least 10-20 observations per parameter for stable estimates
        observations_per_param = total_choices / self.num_params if self.num_params > 0 else 0
        
        # Calculate D-error based on Fisher Information Matrix approximation
        # This is a simplified but more realistic approach
        if observations_per_param <= 1:
            # Very low efficiency when insufficient data
            d_efficiency = 0.1 * observations_per_param
        elif observations_per_param <= 5:
            # Low efficiency, steep improvement curve
            d_efficiency = 0.2 + 0.15 * (observations_per_param - 1)
        elif observations_per_param <= 10:
            # Medium efficiency, moderate improvement
            d_efficiency = 0.8 + 0.04 * (observations_per_param - 5)
        elif observations_per_param <= 20:
            # High efficiency, diminishing returns
            d_efficiency = 1.0 - 0.2 * np.exp(-(observations_per_param - 10) / 5)
        else:
            # Asymptotic approach to 1.0
            d_efficiency = 1.0 - 0.1 * np.exp(-(observations_per_param - 20) / 10)
        
        # Add complexity penalty for designs with many alternatives
        complexity_factor = 1 - (n_alternatives - 2) * 0.05
        d_efficiency = d_efficiency * max(complexity_factor, 0.5)
        
        # Ensure bounds
        d_efficiency = max(0, min(d_efficiency, 1.0))
        
        # Calculate corresponding D-error
        d_error = (1 / d_efficiency) ** (1/self.num_params) if d_efficiency > 0 else float('inf')
        
        return d_efficiency, d_error
    
    def find_optimal_questions(self, n_respondents=60, target_efficiency=0.8):
        """Find optimal number of questions with dynamic range"""
        
        # Calculate recommended minimum questions (rule of thumb: num_params + 2)
        recommended_min = max(1, self.num_params + 2)
        
        # Dynamic range: start from 1, go up to 1.5 times recommended minimum
        min_questions = 1
        max_questions = max(recommended_min, math.ceil(recommended_min * 1.5))
        
        results = []
        
        for n_questions in range(min_questions, max_questions + 1):
            d_eff, d_err = self.calculate_d_efficiency(n_questions, n_respondents)
            results.append({
                'num_questions': n_questions,
                'd_efficiency': d_eff,
                'd_error': d_err
            })
        
        return pd.DataFrame(results), recommended_min, max_questions

# Initialize analyzer
analyzer = ConjointAnalyzer()

# Dynamic attribute input
st.sidebar.subheader("🔧 Define Attributes and Levels")

if 'num_attributes' not in st.session_state:
    st.session_state.num_attributes = 3

num_attributes = st.sidebar.number_input(
    "Number of Attributes", 
    min_value=2, 
    max_value=10, 
    value=st.session_state.num_attributes
)

attributes = {}
for i in range(num_attributes):
    attr_name = st.sidebar.text_input(f"Attribute {i+1} Name", value=f"Attribute_{i+1}")
    
    num_levels = st.sidebar.number_input(
        f"Number of levels for {attr_name}", 
        min_value=2, 
        max_value=6, 
        value=3,
        key=f"levels_{i}"
    )
    
    levels = []
    for j in range(num_levels):
        level = st.sidebar.text_input(
            f"Level {j+1} for {attr_name}", 
            value=f"Level_{j+1}",
            key=f"level_{i}_{j}"
        )
        levels.append(level)
    
    attributes[attr_name] = levels

# Survey parameters
st.sidebar.subheader("📊 Survey Parameters")
n_respondents = st.sidebar.number_input("Number of Respondents", min_value=10, max_value=1000, value=60)
n_alternatives = st.sidebar.selectbox("Alternatives per Question", [2, 3, 4], index=0)
target_efficiency = st.sidebar.slider("Target D-Efficiency", 0.5, 1.0, 0.8, 0.05)

# Main analysis
if st.button("🚀 Run Conjoint Analysis", type="primary"):
    with st.spinner("Generating profiles and calculating efficiency..."):
        
        # Generate profiles
        profiles_df = analyzer.generate_profiles(attributes)
        
        # Calculate optimal questions with dynamic range
        results_df, recommended_min, max_questions_tested = analyzer.find_optimal_questions(
            n_respondents=n_respondents,
            target_efficiency=target_efficiency
        )
        
        # Store results in session state
        st.session_state.profiles = profiles_df
        st.session_state.results = results_df
        st.session_state.attributes = attributes
        st.session_state.num_params = analyzer.num_params
        st.session_state.recommended_min = recommended_min
        st.session_state.max_questions_tested = max_questions_tested

# Display results if available
if 'results' in st.session_state:
    results_df = st.session_state.results
    profiles_df = st.session_state.profiles
    recommended_min = st.session_state.recommended_min
    max_questions_tested = st.session_state.max_questions_tested
    
    # Main results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Profiles", len(profiles_df))
    
    with col2:
        st.metric("Parameters to Estimate", st.session_state.num_params)
    
    with col3:
        optimal_questions = results_df[results_df['d_efficiency'] >= target_efficiency]
        min_questions = optimal_questions['num_questions'].min() if len(optimal_questions) > 0 else "N/A"
        st.metric("Minimum Questions", min_questions)
    
    with col4:
        max_efficiency = results_df['d_efficiency'].max()
        st.metric("Maximum D-Efficiency", f"{max_efficiency:.4f}")
    
    # Show analysis range info
    st.info(f"📊 **Analysis Range:** Testing 1 to {max_questions_tested} questions (1.5× recommended minimum of {recommended_min})")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Efficiency Curve", "📋 Detailed Results", "🎯 Product Profiles", "🤖 GPT Recommendations"])
    
    with tab1:
        st.subheader("D-Efficiency vs Number of Questions")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results_df['num_questions'],
            y=results_df['d_efficiency'],
            mode='lines+markers',
            name='D-Efficiency',
            line=dict(color='darkgreen', width=3),
            marker=dict(size=8)
        ))
        
        # Add target efficiency line
        fig.add_hline(
            y=target_efficiency,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Target Efficiency ({target_efficiency})"
        )
        
        # Add recommended minimum line
        fig.add_vline(
            x=recommended_min,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Recommended Min ({recommended_min})"
        )
        
        fig.update_layout(
            title="Conjoint Design Efficiency Analysis",
            xaxis_title="Number of Questions per Respondent",
            yaxis_title="D-Efficiency",
            hovermode='x unified',
            template='plotly_white',
            xaxis=dict(
                range=[0.5, max_questions_tested + 0.5],
                tick0=1,
                dtick=1
            ),
            yaxis=dict(range=[0, 1.1])  # Set Y-axis range from 0 to 1.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("**📋 Chart Insights:**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"• **Red dashed line:** Target D-efficiency ({target_efficiency})")
            st.markdown(f"• **Blue dotted line:** Recommended minimum ({recommended_min} questions)")
        with col_b:
            if min_questions != "N/A":
                st.markdown(f"• **Optimal point:** {min_questions} questions achieves target")
            st.markdown(f"• **Range tested:** 1 to {max_questions_tested} questions")
        
        # Show calculation methodology
        with st.expander("📚 D-Efficiency Calculation Methodology"):
            st.markdown(f"""
            **Calculation Method:**
            - **Observations per parameter:** Total choices ÷ Parameters to estimate
            - **Current study:** {st.session_state.num_params} parameters need estimation
            - **With 60 respondents and varying questions:** More questions = more observations = higher efficiency
            
            **Efficiency Scale:**
            - **0-1 obs/param:** Very low efficiency (0.0-0.1)
            - **1-5 obs/param:** Low efficiency (0.1-0.8)
            - **5-10 obs/param:** Good efficiency (0.8-1.0)
            - **10+ obs/param:** Excellent efficiency (approaching 1.0)
            
            **Note:** This is a simplified approximation. Real D-efficiency requires complex matrix calculations.
            """)
    
    with tab2:
        st.subheader("Detailed Efficiency Results")
        
        # Add a note about the range
        st.markdown(f"**Range:** 1 to {max_questions_tested} questions (recommended minimum: {recommended_min})")
        
        # Add observations per parameter column for better understanding
        results_display = results_df.copy()
        results_display['obs_per_param'] = (results_display['num_questions'] * n_respondents) / st.session_state.num_params
        results_display['obs_per_param'] = results_display['obs_per_param'].round(2)
        
        # Reorder columns
        results_display = results_display[['num_questions', 'obs_per_param', 'd_efficiency', 'd_error']]
        
        # Highlight optimal rows
        optimal_mask = results_display['d_efficiency'] >= target_efficiency
        
        st.dataframe(
            results_display.style.apply(
                lambda x: ['background-color: lightgreen' if optimal_mask[x.name] else '' for _ in x], 
                axis=1
            ),
            use_container_width=True
        )
        
        # Download button for results
        csv = results_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="conjoint_efficiency_results.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.subheader("Product Profiles")
        st.write(f"**Total possible profiles:** {len(profiles_df)}")
        
        # Show sample profiles
        if len(profiles_df) > 20:
            st.write("**Sample profiles (first 20):**")
            st.dataframe(profiles_df.head(20))
        else:
            st.dataframe(profiles_df)
        
        # Download profiles
        profiles_csv = profiles_df.to_csv(index=False)
        st.download_button(
            label="📥 Download All Profiles as CSV",
            data=profiles_csv,
            file_name="product_profiles.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.subheader("🤖 GPT-Powered Recommendations")
        
        if openai_available:
            if st.button("Generate GPT Recommendations"):
                with st.spinner("Generating AI recommendations..."):
                    try:
                        client = get_openai_client()
                        
                        # Prepare context for GPT
                        context = f"""
                        Conjoint Analysis Results Summary:
                        - Total Attributes: {len(st.session_state.attributes)}
                        - Attributes: {list(st.session_state.attributes.keys())}
                        - Total Profiles: {len(profiles_df)}
                        - Parameters to Estimate: {st.session_state.num_params}
                        - Respondents: {n_respondents}
                        - Target D-Efficiency: {target_efficiency}
                        - Maximum Achieved D-Efficiency: {max_efficiency:.4f}
                        - Recommended Minimum Questions: {recommended_min}
                        - Range Tested: 1 to {max_questions_tested} questions
                        - Optimal Questions: {min_questions if min_questions != "N/A" else "Could not achieve target"}
                        
                        Please provide recommendations for this conjoint study design including:
                        1. Survey design optimization
                        2. Sample size recommendations
                        3. Potential challenges and solutions
                        4. Implementation best practices
                        """
                        
                        response = client.chat.completions.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "You are a market research expert specializing in conjoint analysis and survey design."},
                                {"role": "user", "content": context}
                            ],
                            max_tokens=1500,
                            temperature=0.7
                        )
                        
                        st.success("GPT Recommendations Generated!")
                        st.markdown(response.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"Error generating GPT recommendations: {str(e)}")
        else:
            st.info("🔑 OpenAI API key not configured in secrets. Contact administrator to enable GPT recommendations.")
            
            # Provide basic recommendations
            st.markdown("""
            ### Basic Recommendations:
            
            **Survey Design:**
            - Ensure your D-efficiency meets the target threshold (typically 0.8+)
            - Balance survey length with statistical power
            - Consider using blocked designs for large attribute sets
            
            **Sample Size:**
            - Minimum 200+ respondents for stable results
            - Increase sample size if you have many segments to analyze
            - Consider using quotas to ensure representative sampling
            
            **Implementation:**
            - Pre-test your survey with a small sample
            - Randomize choice task order
            - Include attention checks and quality controls
            """)

# Summary section
if 'results' in st.session_state:
    st.markdown("---")
    st.subheader("📊 Summary")
    
    optimal_questions = results_df[results_df['d_efficiency'] >= target_efficiency]
    
    if len(optimal_questions) > 0:
        min_q = optimal_questions['num_questions'].min()
        eff_at_min = optimal_questions[optimal_questions['num_questions'] == min_q]['d_efficiency'].iloc[0]
        obs_per_param_at_min = (min_q * n_respondents) / st.session_state.num_params
        
        st.success(f"""
        **Optimal Design Found!**
        - Minimum questions per respondent: **{min_q}**
        - D-efficiency at minimum: **{eff_at_min:.4f}**
        - Observations per parameter: **{obs_per_param_at_min:.1f}**
        - This design meets your target efficiency of {target_efficiency}
        - Recommended minimum was: **{recommended_min}** questions
        """)
    else:
        st.warning(f"""
        **Target efficiency not achieved in tested range**
        - Consider increasing the number of questions beyond {max_questions_tested}
        - Or reducing the target D-efficiency threshold
        - Maximum achieved efficiency: **{max_efficiency:.4f}**
        - Recommended minimum: **{recommended_min}** questions
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
<p><strong>Conjoint Demand Estimation Tool</strong><br>
Built with Streamlit | Powered by Python & OpenAI</p>
</div>
""", unsafe_allow_html=True)
