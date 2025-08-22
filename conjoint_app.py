# Conjoint Analysis D-Efficiency Tool
# A Streamlit application for designing and analyzing conjoint experiments

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pyDOE3 import fullfact
from itertools import product
import json
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Conjoint D-Efficiency Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ConjointDesigner:
    def __init__(self):
        self.attributes = {}
        self.design_results = []
    
    def calculate_parameters(self, attributes_dict):
        """Calculate number of parameters to estimate"""
        return sum(len(levels) - 1 for levels in attributes_dict.values())
    
    def generate_full_factorial(self, attributes_dict):
        """Generate full factorial design"""
        levels = [list(range(len(levels))) for levels in attributes_dict.values()]
        design = list(product(*levels))
        return pd.DataFrame(design, columns=list(attributes_dict.keys()))
    
    def calculate_d_efficiency(self, design_matrix, num_params):
        """
        Calculate D-efficiency for a given design
        This is a simplified calculation - in practice, you'd need the full information matrix
        """
        try:
            # Create dummy variables for categorical attributes
            X = pd.get_dummies(design_matrix, drop_first=True)
            
            # Calculate information matrix (X'X)
            XtX = np.dot(X.T, X)
            
            # Calculate determinant
            det_XtX = np.linalg.det(XtX)
            
            if det_XtX <= 0:
                return 0.0
            
            # Calculate D-efficiency
            n_runs = len(design_matrix)
            d_error = (det_XtX / (n_runs ** num_params)) ** (-1/num_params)
            d_efficiency = 1 / d_error
            
            return min(d_efficiency, 1.0)
        except:
            return 0.0
    
    def generate_fractional_design(self, full_design, fraction_size):
        """Generate fractional factorial design"""
        if fraction_size >= len(full_design):
            return full_design
        
        # Simple random sampling - in practice, you'd use optimal selection
        np.random.seed(42)
        indices = np.random.choice(len(full_design), size=fraction_size, replace=False)
        return full_design.iloc[indices].reset_index(drop=True)
    
    def optimize_design(self, attributes_dict, n_respondents, n_alternatives=2, target_efficiency=0.8):
        """
        Optimize conjoint design for different numbers of questions
        """
        num_params = self.calculate_parameters(attributes_dict)
        full_design = self.generate_full_factorial(attributes_dict)
        
        results = []
        
        # Test different numbers of questions
        min_questions = max(num_params + 2, 8)
        max_questions = min(30, len(full_design) // 2)
        
        for n_questions in range(min_questions, max_questions + 1):
            total_runs = n_respondents * n_questions * n_alternatives
            
            if total_runs > len(full_design):
                # Use fractional design
                design = self.generate_fractional_design(full_design, total_runs)
            else:
                design = full_design.sample(n=total_runs, replace=True, random_state=42)
            
            d_efficiency = self.calculate_d_efficiency(design, num_params)
            
            results.append({
                'num_questions': n_questions,
                'total_runs': total_runs,
                'd_efficiency': d_efficiency,
                'meets_target': d_efficiency >= target_efficiency
            })
        
        return pd.DataFrame(results)

def main():
    st.title("🎯 Conjoint Analysis D-Efficiency Tool")
    st.markdown("**Design and optimize conjoint experiments for market research**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔧 Design Setup", "📊 Analysis", "📋 Results", "💾 Export"]
    )
    
    designer = ConjointDesigner()
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔧 Design Setup":
        show_design_setup(designer)
    elif page == "📊 Analysis":
        show_analysis_page(designer)
    elif page == "📋 Results":
        show_results_page(designer)
    elif page == "💾 Export":
        show_export_page()

def show_home_page():
    st.header("Welcome to the Conjoint Analysis D-Efficiency Tool")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This tool helps you design optimal conjoint experiments by:
        
        ### 🎯 **Key Features**
        - **Attribute Definition**: Define your product attributes and levels
        - **D-Efficiency Calculation**: Automatically calculate design efficiency
        - **Optimization**: Find the minimum number of questions needed
        - **Visualization**: Interactive charts showing efficiency curves
        - **Export Options**: Download designs and results
        
        ### 📈 **What is D-Efficiency?**
        D-efficiency measures how well a conjoint design can estimate main effects and interactions. 
        Higher D-efficiency (closer to 1.0) means more precise parameter estimates with fewer observations.
        
        ### 🚀 **Getting Started**
        1. Go to **Design Setup** to define your attributes
        2. Use **Analysis** to optimize your design
        3. View **Results** to see efficiency curves
        4. **Export** your final design
        """)
    
    with col2:
        st.info("""
        **Quick Example:**
        - 5 attributes with 2-4 levels each
        - 60 respondents
        - Target: 80% D-efficiency
        - Result: ~12 questions per respondent
        """)
        
        st.success("""
        **Benefits:**
        - Reduce survey length
        - Improve response quality
        - Optimize data collection costs
        - Maintain statistical power
        """)

def show_design_setup(designer):
    st.header("🔧 Design Setup")
    st.markdown("Define your conjoint experiment attributes and levels")
    
    # Initialize session state for attributes
    if 'attributes' not in st.session_state:
        st.session_state.attributes = {
            'A1': ['Level 1', 'Level 2', 'Level 3'],
            'A2': ['Option 1', 'Option 2']
        }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Add New Attribute")
        
        with st.form("add_attribute"):
            attr_name = st.text_input("Attribute Name", placeholder="e.g., Price, Brand, Features")
            attr_levels = st.text_area(
                "Levels (one per line)", 
                placeholder="Level 1\nLevel 2\nLevel 3",
                height=100
            )
            
            if st.form_submit_button("Add Attribute", type="primary"):
                if attr_name and attr_levels:
                    levels_list = [level.strip() for level in attr_levels.split('\n') if level.strip()]
                    if len(levels_list) >= 2:
                        st.session_state.attributes[attr_name] = levels_list
                        st.success(f"Added attribute '{attr_name}' with {len(levels_list)} levels")
                        st.rerun()
                    else:
                        st.error("Each attribute must have at least 2 levels")
                else:
                    st.error("Please provide both attribute name and levels")
    
    with col2:
        st.subheader("Current Attributes")
        
        if st.session_state.attributes:
            for attr_name, levels in st.session_state.attributes.items():
                with st.expander(f"**{attr_name}** ({len(levels)} levels)", expanded=False):
                    for i, level in enumerate(levels, 1):
                        st.write(f"{i}. {level}")
                    
                    if st.button(f"Remove {attr_name}", key=f"remove_{attr_name}"):
                        del st.session_state.attributes[attr_name]
                        st.rerun()
        else:
            st.info("No attributes defined yet. Add some attributes to get started!")
    
    # Study parameters
    st.subheader("Study Parameters")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        n_respondents = st.number_input(
            "Number of Respondents", 
            min_value=10, 
            max_value=1000, 
            value=60,
            help="Total number of survey respondents"
        )
    
    with col4:
        n_alternatives = st.number_input(
            "Alternatives per Choice Set", 
            min_value=2, 
            max_value=5, 
            value=2,
            help="Number of product options shown per question"
        )
    
    with col5:
        target_efficiency = st.slider(
            "Target D-Efficiency", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.8, 
            step=0.05,
            help="Minimum acceptable D-efficiency (0.8 is commonly used)"
        )
    
    # Save parameters to session state
    st.session_state.update({
        'n_respondents': n_respondents,
        'n_alternatives': n_alternatives,
        'target_efficiency': target_efficiency
    })
    
    # Summary
    if st.session_state.attributes:
        st.subheader("Design Summary")
        
        total_levels = sum(len(levels) for levels in st.session_state.attributes.values())
        num_params = sum(len(levels) - 1 for levels in st.session_state.attributes.values())
        full_factorial_size = np.prod([len(levels) for levels in st.session_state.attributes.values()])
        
        col6, col7, col8, col9 = st.columns(4)
        
        with col6:
            st.metric("Attributes", len(st.session_state.attributes))
        with col7:
            st.metric("Total Levels", total_levels)
        with col8:
            st.metric("Parameters to Estimate", num_params)
        with col9:
            st.metric("Full Factorial Size", f"{full_factorial_size:,}")

def show_analysis_page(designer):
    st.header("📊 Analysis")
    
    if 'attributes' not in st.session_state or not st.session_state.attributes:
        st.warning("⚠️ Please define attributes in the Design Setup page first!")
        return
    
    st.markdown("Optimize your conjoint design for maximum efficiency")
    
    # Get parameters from session state
    attributes_dict = st.session_state.attributes
    n_respondents = st.session_state.get('n_respondents', 60)
    n_alternatives = st.session_state.get('n_alternatives', 2)
    target_efficiency = st.session_state.get('target_efficiency', 0.8)
    
    # Run optimization
    with st.spinner("Optimizing design... This may take a moment."):
        results_df = designer.optimize_design(
            attributes_dict, n_respondents, n_alternatives, target_efficiency
        )
    
    if results_df.empty:
        st.error("Unable to generate design. Please check your parameters.")
        return
    
    # Store results in session state
    st.session_state.results_df = results_df
    
    # Find optimal design
    meets_target = results_df[results_df['meets_target'] == True]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 D-Efficiency Curve")
        
        # Create interactive plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_df['num_questions'],
            y=results_df['d_efficiency'],
            mode='lines+markers',
            name='D-Efficiency',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        # Add target line
        fig.add_hline(
            y=target_efficiency, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Target: {target_efficiency}"
        )
        
        # Highlight optimal point
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
            title="D-Efficiency vs Number of Questions",
            xaxis_title="Number of Questions per Respondent",
            yaxis_title="D-Efficiency",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Optimization Results")
        
        if not meets_target.empty:
            optimal = meets_target.iloc[0]
            
            st.success("✅ Target efficiency achieved!")
            
            st.metric(
                "Optimal Questions per Respondent",
                int(optimal['num_questions'])
            )
            st.metric(
                "Achieved D-Efficiency",
                f"{optimal['d_efficiency']:.3f}"
            )
            st.metric(
                "Total Survey Responses",
                f"{optimal['total_runs']:,}"
            )
            
            # Calculate survey statistics
            total_questions = int(optimal['num_questions']) * n_respondents
            completion_time = int(optimal['num_questions']) * 0.5  # Assume 30 seconds per question
            
            st.metric("Total Questions", f"{total_questions:,}")
            st.metric("Est. Completion Time", f"{completion_time:.0f} min")
            
        else:
            st.warning("⚠️ Target efficiency not achieved in tested range")
            best = results_df.loc[results_df['d_efficiency'].idxmax()]
            
            st.metric(
                "Best Achievable Questions",
                int(best['num_questions'])
            )
            st.metric(
                "Best D-Efficiency",
                f"{best['d_efficiency']:.3f}"
            )
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    display_df = results_df.copy()
    display_df['d_efficiency'] = display_df['d_efficiency'].round(4)
    display_df['meets_target'] = display_df['meets_target'].map({True: '✅', False: '❌'})
    
    st.dataframe(
        display_df.rename(columns={
            'num_questions': 'Questions per Respondent',
            'total_runs': 'Total Responses',
            'd_efficiency': 'D-Efficiency',
            'meets_target': 'Meets Target'
        }),
        use_container_width=True,
        hide_index=True
    )

def show_results_page(designer):
    st.header("📋 Results Summary")
    
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please run the analysis first!")
        return
    
    results_df = st.session_state.results_df
    attributes_dict = st.session_state.attributes
    n_respondents = st.session_state.get('n_respondents', 60)
    target_efficiency = st.session_state.get('target_efficiency', 0.8)
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    meets_target = results_df[results_df['meets_target'] == True]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not meets_target.empty:
            optimal_questions = int(meets_target.iloc[0]['num_questions'])
            st.metric(
                "Recommended Questions", 
                optimal_questions,
                help="Minimum questions per respondent to achieve target efficiency"
            )
        else:
            st.metric("Recommended Questions", "Not achieved")
    
    with col2:
        max_efficiency = results_df['d_efficiency'].max()
        st.metric(
            "Maximum D-Efficiency", 
            f"{max_efficiency:.3f}",
            help="Best achievable efficiency in tested range"
        )
    
    with col3:
        efficiency_range = results_df['d_efficiency'].max() - results_df['d_efficiency'].min()
        st.metric(
            "Efficiency Range", 
            f"{efficiency_range:.3f}",
            help="Difference between best and worst efficiency"
        )
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if not meets_target.empty:
        optimal_row = meets_target.iloc[0]
        
        st.success(f"""
        **Recommended Design:**
        - **{int(optimal_row['num_questions'])} questions** per respondent
        - **{n_respondents} respondents** total
        - **D-efficiency: {optimal_row['d_efficiency']:.3f}** (exceeds target of {target_efficiency})
        - **Total responses needed: {optimal_row['total_runs']:,}**
        """)
        
        # Survey logistics
        completion_time = int(optimal_row['num_questions']) * 0.5
        total_time = completion_time * n_respondents / 60
        
        st.info(f"""
        **Survey Logistics:**
        - Estimated completion time: **{completion_time:.0f} minutes** per respondent
        - Total data collection time: **{total_time:.1f} hours** (if sequential)
        - Consider offering incentives for surveys longer than 10 minutes
        """)
        
    else:
        st.warning(f"""
        **Target not achieved** with current parameters.
        
        **Options to consider:**
        1. **Reduce target efficiency** to {target_efficiency - 0.1:.1f} or lower
        2. **Increase respondent count** to improve precision
        3. **Simplify attribute structure** by reducing levels
        4. **Accept longer survey** with more questions
        """)
    
    # Attribute summary
    st.subheader("🏷️ Attribute Summary")
    
    attr_summary = []
    for attr_name, levels in attributes_dict.items():
        attr_summary.append({
            'Attribute': attr_name,
            'Levels': len(levels),
            'Level Names': ', '.join(levels[:3]) + ('...' if len(levels) > 3 else ''),
            'Parameters': len(levels) - 1
        })
    
    st.dataframe(pd.DataFrame(attr_summary), use_container_width=True, hide_index=True)
    
    # Efficiency trend analysis
    st.subheader("📊 Efficiency Trend Analysis")
    
    col4, col5 = st.columns(2)
    
    with col4:
        # Marginal efficiency gains
        results_df['marginal_gain'] = results_df['d_efficiency'].diff()
        
        fig_marginal = go.Figure()
        fig_marginal.add_trace(go.Bar(
            x=results_df['num_questions'][1:],
            y=results_df['marginal_gain'][1:],
            name='Marginal Efficiency Gain'
        ))
        
        fig_marginal.update_layout(
            title="Marginal Efficiency Gains",
            xaxis_title="Questions per Respondent",
            yaxis_title="Efficiency Gain",
            height=300
        )
        
        st.plotly_chart(fig_marginal, use_container_width=True)
    
    with col5:
        # Cost-benefit analysis
        cost_per_question = st.number_input(
            "Cost per Question ($)", 
            min_value=0.01, 
            max_value=10.0, 
            value=0.5,
            step=0.1,
            help="Estimated cost per question per respondent"
        )
        
        results_df['total_cost'] = results_df['num_questions'] * n_respondents * cost_per_question
        results_df['efficiency_per_dollar'] = results_df['d_efficiency'] / results_df['total_cost']
        
        # Find most cost-effective design
        best_value_idx = results_df['efficiency_per_dollar'].idxmax()
        best_value_row = results_df.iloc[best_value_idx]
        
        st.metric(
            "Most Cost-Effective",
            f"{int(best_value_row['num_questions'])} questions",
            f"${best_value_row['total_cost']:,.0f} total cost"
        )

def show_export_page():
    st.header("💾 Export Results")
    
    if 'results_df' not in st.session_state or 'attributes' not in st.session_state:
        st.warning("⚠️ No results to export. Please complete the analysis first!")
        return
    
    st.markdown("Download your conjoint design and analysis results")
    
    results_df = st.session_state.results_df
    attributes_dict = st.session_state.attributes
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Export Options")
        
        # Create Excel file with multiple sheets
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Results sheet
            results_df.to_excel(writer, sheet_name='Analysis Results', index=False)
            
            # Attributes sheet
            attr_df = pd.DataFrame([
                {'Attribute': k, 'Level': level, 'Level_Code': i+1}
                for k, v in attributes_dict.items()
                for i, level in enumerate(v)
            ])
            attr_df.to_excel(writer, sheet_name='Attributes', index=False)
            
            # Parameters sheet
            params_df = pd.DataFrame([{
                'Parameter': 'Number of Respondents',
                'Value': st.session_state.get('n_respondents', 60)
            }, {
                'Parameter': 'Alternatives per Choice Set',
                'Value': st.session_state.get('n_alternatives', 2)
            }, {
                'Parameter': 'Target D-Efficiency',
                'Value': st.session_state.get('target_efficiency', 0.8)
            }])
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
        
        excel_data = output.getvalue()
        
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name="conjoint_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # CSV export
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📋 Download CSV Results",
            data=csv_data,
            file_name="conjoint_results.csv",
            mime="text/csv"
        )
        
        # JSON export
        export_data = {
            'attributes': attributes_dict,
            'parameters': {
                'n_respondents': st.session_state.get('n_respondents', 60),
                'n_alternatives': st.session_state.get('n_alternatives', 2),
                'target_efficiency': st.session_state.get('target_efficiency', 0.8)
            },
            'results': results_df.to_dict('records')
        }
        
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            label="🔧 Download JSON Configuration",
            data=json_data,
            file_name="conjoint_config.json",
            mime="application/json"
        )
    
    with col2:
        st.subheader("📋 Export Preview")
        
        # Show sample of results
        st.markdown("**Analysis Results Preview:**")
        st.dataframe(results_df.head(), use_container_width=True)
        
        # Show attributes
        st.markdown("**Attributes Preview:**")
        attr_preview = pd.DataFrame([
            {'Attribute': k, 'Levels': len(v), 'Sample Levels': ', '.join(v[:2]) + ('...' if len(v) > 2 else '')}
            for k, v in attributes_dict.items()
        ])
        st.dataframe(attr_preview, use_container_width=True, hide_index=True)
    
    # Import functionality
    st.subheader("📥 Import Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload existing configuration",
        type=['json'],
        help="Upload a previously exported JSON configuration file"
    )
    
    if uploaded_file is not None:
        try:
            config_data = json.load(uploaded_file)
            
            if st.button("Import Configuration"):
                st.session_state.attributes = config_data.get('attributes', {})
                st.session_state.update(config_data.get('parameters', {}))
                
                st.success("✅ Configuration imported successfully!")
                st.info("Go to the Analysis page to run the imported configuration.")
                
        except Exception as e:
            st.error(f"Error importing file: {str(e)}")

# Helper functions for file download
def get_download_link(data, filename, text):
    """Generate download link for data"""
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'

if __name__ == "__main__":
    main()
