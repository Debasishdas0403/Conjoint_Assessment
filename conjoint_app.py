import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from advanced_utils import AdvancedConjointAnalyzer

# Page configuration
st.set_page_config(
    page_title="Conjoint Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'attributes' not in st.session_state:
    st.session_state.attributes = {}
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = AdvancedConjointAnalyzer()

def main():
    st.markdown('<h1 class="main-header">📊 Advanced Conjoint Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Optimize your conjoint study design with statistical precision**")
    
    # Sidebar
    st.sidebar.title("🎯 Study Configuration")
    
    # Attributes section
    st.sidebar.subheader("📝 Attributes & Levels")
    
    with st.sidebar.form("add_attribute_form"):
        attr_name = st.text_input("Attribute Name", placeholder="e.g., Price, Brand")
        attr_levels = st.text_area(
            "Levels (one per line)", 
            placeholder="e.g.,\n\$10\n\$15\n\$20",
            height=80
        )
        
        add_attr = st.form_submit_button("➕ Add Attribute")
        
        if add_attr:
            if attr_name and attr_levels:
                levels = [level.strip() for level in attr_levels.split('\n') if level.strip()]
                if len(levels) >= 2:
                    st.session_state.attributes[attr_name] = levels
                    st.success(f"Added '{attr_name}' with {len(levels)} levels")
                    st.rerun()
                else:
                    st.error("Please provide at least 2 levels")
            else:
                st.error("Please provide both attribute name and levels")
    
    # Display current attributes
    if st.session_state.attributes:
        st.sidebar.subheader("📋 Current Attributes")
        for attr_name, levels in st.session_state.attributes.items():
            with st.sidebar.expander(f"**{attr_name}** ({len(levels)} levels)"):
                for i, level in enumerate(levels, 1):
                    st.write(f"{i}. {level}")
                
                if st.button(f"🗑️ Remove {attr_name}", key=f"remove_{attr_name}"):
                    del st.session_state.attributes[attr_name]
                    st.rerun()
    
    # Survey parameters
    st.sidebar.subheader("📊 Survey Parameters")
    n_respondents = st.sidebar.number_input("Number of Respondents", min_value=10, max_value=1000, value=100)
    n_alternatives = st.sidebar.selectbox("Alternatives per Question", [2, 3, 4], index=0)
    target_efficiency = st.sidebar.slider("Target D-Efficiency", 0.5, 1.0, 0.8, 0.05)
    
    # Main content
    if not st.session_state.attributes:
        st.info("👆 Please add attributes and levels using the sidebar to get started!")
        return
    
    # Study summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Attributes", len(st.session_state.attributes))
    
    with col2:
        total_levels = sum(len(levels) for levels in st.session_state.attributes.values())
        st.metric("Total Levels", total_levels)
    
    with col3:
        num_params = sum(len(levels) - 1 for levels in st.session_state.attributes.values()) + 1
        st.metric("Parameters", num_params)
    
    with col4:
        full_factorial = np.prod([len(levels) for levels in st.session_state.attributes.values()])
        st.metric("Full Factorial", f"{full_factorial:,}")
    
    # Analysis button
    if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
        run_analysis(n_respondents, n_alternatives, target_efficiency, num_params)

def run_analysis(n_respondents, n_alternatives, target_efficiency, num_params):
    """Run the complete conjoint analysis"""
    
    with st.spinner("🔄 Running advanced D-efficiency optimization..."):
        
        analyzer = st.session_state.analyzer
        
        # Find optimal design
        results_df, theoretical_min, max_questions_tested = analyzer.find_optimal_questions_advanced(
            st.session_state.attributes, 
            n_respondents, 
            target_efficiency
        )
        
        # Store results
        st.session_state.results = results_df
        st.session_state.theoretical_min = theoretical_min
        st.session_state.max_questions_tested = max_questions_tested
        st.session_state.num_params = num_params
    
    # Display results
    display_results(results_df, theoretical_min, target_efficiency, n_respondents)

def display_results(results_df, theoretical_min, target_efficiency, n_respondents):
    """Display comprehensive analysis results"""
    
    st.markdown("---")
    st.subheader("🎯 Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Efficiency Chart", "📊 Detailed Results", "🎯 Recommendations", "📋 Design Summary"])
    
    with tab1:
        display_efficiency_chart(results_df, theoretical_min, target_efficiency)
    
    with tab2:
        display_detailed_results(results_df, target_efficiency, n_respondents, theoretical_min)
    
    with tab3:
        display_recommendations(results_df, target_efficiency, theoretical_min)
    
    with tab4:
        display_design_summary()

def display_efficiency_chart(results_df, theoretical_min, target_efficiency):
    """Display the main efficiency chart"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main efficiency plot
        fig = go.Figure()
        
        # D-efficiency line
        fig.add_trace(go.Scatter(
            x=results_df['num_questions'],
            y=results_df['d_efficiency'],
            mode='lines+markers',
            name='D-Efficiency',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Target line
        fig.add_hline(
            y=target_efficiency, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Target: {target_efficiency}"
        )
        
        # Theoretical minimum vertical line
        fig.add_vline(
            x=theoretical_min,
            line_dash="dot",
            line_color="orange",
            annotation_text=f"Theoretical Min: {theoretical_min}"
        )
        
        # Highlight optimal point
        meets_target = results_df[results_df['d_efficiency'] >= target_efficiency]
        if not meets_target.empty:
            optimal_row = meets_target.iloc[0]
            fig.add_trace(go.Scatter(
                x=[optimal_row['num_questions']],
                y=[optimal_row['d_efficiency']],
                mode='markers',
                name='Optimal Design',
                marker=dict(size=15, color='red', symbol='star')
            ))
        
        fig.update_layout(
            title="D-Efficiency vs Number of Questions per Respondent",
            xaxis_title="Number of Questions per Respondent",
            yaxis_title="D-Efficiency",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Key Findings")
        
        if not meets_target.empty:
            optimal = meets_target.iloc[0]
            st.success("✅ Target achieved!")
            st.metric("Optimal Questions", int(optimal['num_questions']))
            st.metric("D-Efficiency", f"{optimal['d_efficiency']:.3f}")
            st.metric("Total Observations", f"{optimal['total_runs']:,}")
        else:
            max_eff_row = results_df.loc[results_df['d_efficiency'].idxmax()]
            st.warning("⚠️ Target not achieved")
            st.metric("Best Questions", int(max_eff_row['num_questions']))
            st.metric("Max D-Efficiency", f"{max_eff_row['d_efficiency']:.3f}")
            st.info(f"Consider increasing questions beyond {results_df['num_questions'].max()}")

def display_detailed_results(results_df, target_efficiency, n_respondents, theoretical_min):
    """Display detailed results table"""
    
    # Prepare display dataframe
    display_df = results_df.copy()
    display_df['Status'] = ''
    
    # Add status indicators
    optimal_questions = results_df[results_df['d_efficiency'] >= target_efficiency]
    if not optimal_questions.empty:
        optimal_min = optimal_questions['num_questions'].min()
        display_df.loc[display_df['num_questions'] == optimal_min, 'Status'] = '🎯 OPTIMAL'
    
    display_df.loc[display_df['num_questions'] == theoretical_min, 'Status'] += ' 📏 THEORETICAL'
    display_df.loc[display_df['d_efficiency'] >= target_efficiency, 'Status'] += ' ✅ MEETS TARGET'
    
    # Format columns
    display_df['d_efficiency'] = display_df['d_efficiency'].round(4)
    display_df['d_error'] = display_df['d_error'].round(4)
    display_df['avg_std_error'] = display_df['avg_std_error'].round(4)
    display_df['obs_per_param'] = display_df['obs_per_param'].round(1)
    
    # Reorder columns
    columns = ['num_questions', 'Status', 'total_runs', 'd_efficiency', 'd_error', 'avg_std_error', 'obs_per_param']
    display_df = display_df[columns]
    
    # Style the dataframe
    def highlight_optimal(s):
        return ['background-color: lightgreen' if '🎯 OPTIMAL' in str(val) else '' for val in s]
    
    st.dataframe(
        display_df.style.apply(highlight_optimal, subset=['Status']),
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="conjoint_efficiency_results.csv",
        mime="text/csv"
    )

def display_recommendations(results_df, target_efficiency, theoretical_min):
    """Display actionable recommendations"""
    
    optimal_questions = results_df[results_df['d_efficiency'] >= target_efficiency]
    
    if not optimal_questions.empty:
        optimal = optimal_questions.iloc[0]
        
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Recommended Design</h3>
            <p><strong>Use {int(optimal['num_questions'])} questions per respondent</strong></p>
            <ul>
                <li>Achieves D-efficiency of {optimal['d_efficiency']:.3f}</li>
                <li>Total observations: {optimal['total_runs']:,}</li>
                <li>Observations per parameter: {optimal['obs_per_param']:.1f}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        max_eff_row = results_df.loc[results_df['d_efficiency'].idxmax()]
        
        st.markdown(f"""
        <div class="warning-box">
            <h3>⚠️ Target Not Achieved</h3>
            <p><strong>Best option: {int(max_eff_row['num_questions'])} questions</strong></p>
            <p>Maximum D-efficiency: {max_eff_row['d_efficiency']:.3f}</p>
            
            <h4>Options to improve efficiency:</h4>
            <ol>
                <li>Increase number of questions beyond {results_df['num_questions'].max()}</li>
                <li>Increase number of respondents</li>
                <li>Reduce number of attribute levels</li>
                <li>Accept lower target efficiency</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Efficiency trend analysis
    st.subheader("📊 Efficiency Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Marginal gains
        results_df['marginal_gain'] = results_df['d_efficiency'].diff()
        
        fig_marginal = go.Figure()
        fig_marginal.add_trace(go.Bar(
            x=results_df['num_questions'][1:],
            y=results_df['marginal_gain'][1:],
            name='Marginal D-Efficiency Gain'
        ))
        
        fig_marginal.update_layout(
            title="Marginal Efficiency Gains",
            xaxis_title="Questions per Respondent",
            yaxis_title="Marginal Gain",
            height=300
        )
        
        st.plotly_chart(fig_marginal, use_container_width=True)
    
    with col2:
        # Cost-benefit analysis
        results_df['cost_benefit'] = results_df['d_efficiency'] / results_df['num_questions']
        
        fig_cb = go.Figure()
        fig_cb.add_trace(go.Scatter(
            x=results_df['num_questions'],
            y=results_df['cost_benefit'],
            mode='lines+markers',
            name='Efficiency per Question'
        ))
        
        fig_cb.update_layout(
            title="Efficiency per Question (Cost-Benefit)",
            xaxis_title="Questions per Respondent",
            yaxis_title="D-Efficiency / Questions",
            height=300
        )
        
        st.plotly_chart(fig_cb, use_container_width=True)

def display_design_summary():
    """Display comprehensive design summary"""
