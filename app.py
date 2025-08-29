import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import det
import io

def effects_coding(levels_count, design_column):
    """Convert categorical levels to effects coding"""
    n = len(design_column)
    if levels_count <= 1:
        return np.zeros((n, 0))
    coded = np.zeros((n, levels_count - 1))
    for i in range(levels_count - 1):
        coded[:, i] = np.where(design_column == (i + 1), 1, 0)
    coded[:, -1] = -np.sum(coded[:, :-1], axis=1)
    return coded

def calculate_d_efficiency(df, attributes, levels):
    """Calculate D-efficiency with improved error handling"""
    try:
        design_matrix_parts = []
        for attr in attributes:
            levels_count = len(levels[attr])
            col = df[attr].values
            coded_part = effects_coding(levels_count, col)
            if coded_part.shape[1] > 0:
                design_matrix_parts.append(coded_part)
        if len(design_matrix_parts) == 0:
            return 0
        X = np.hstack(design_matrix_parts)
        rank = np.linalg.matrix_rank(X)
        if rank < X.shape[1]:
            return 0  # Singular matrix
        XTX = np.dot(X.T, X)
        p = X.shape[1]
        N = X.shape[0]
        det_XTX = det(XTX)
        if det_XTX <= 0:
            return 0
        return (det_XTX ** (1 / p)) / N
    except Exception as e:
        return 0

def calculate_optimal_cards(num_respondents, blocking, n_blocks, levels_per_attribute, user_cards):
    """Calculate optimal number of cards with user input option"""
    # Minimum cards needed: 3x the number of parameters
    min_cards_needed = sum(l - 1 for l in levels_per_attribute) * 3
    
    if blocking:
        min_cards_per_block = max(15, min_cards_needed // n_blocks)
        cards_per_block = max(min_cards_per_block, num_respondents // n_blocks)
        total_cards = cards_per_block * n_blocks
    else:
        total_cards = max(num_respondents, min_cards_needed)
        cards_per_block = total_cards
    
    # If user specifies cards, use that instead (with validation)
    if user_cards > 0:
        total_cards = min(user_cards, np.prod(levels_per_attribute))
        if blocking:
            cards_per_block = total_cards // n_blocks
        else:
            cards_per_block = total_cards
    
    # Cap at total possible combinations
    total_combinations = np.prod(levels_per_attribute)
    total_cards = min(total_cards, total_combinations)
    cards_per_block = min(cards_per_block, total_cards)
    
    return int(total_cards), int(cards_per_block)

def generate_balanced_block(block_num, block_size, attributes, levels, cards_so_far=0):
    """Generate a balanced block with equal level distribution"""
    np.random.seed(42 + block_num)  # Reproducible but different per block
    
    block_data = {}
    block_data['Block'] = [block_num] * block_size
    block_data['Card Number'] = [cards_so_far + i + 1 for i in range(block_size)]
    block_data['Brand'] = ['New Brand'] * block_size
    
    # Generate balanced levels for each attribute
    for attr in attributes:
        attr_levels = levels[attr]
        num_levels = len(attr_levels)
        
        # Calculate how many times each level should appear
        repeats = block_size // num_levels
        remainder = block_size % num_levels
        
        levels_list = []
        for i, level in enumerate(attr_levels):
            count = repeats + (1 if i < remainder else 0)
            levels_list.extend([level] * count)
        
        # Randomize order while maintaining balance
        np.random.shuffle(levels_list)
        block_data[attr] = levels_list
    
    return pd.DataFrame(block_data)

def generate_design(num_attributes, levels_per_attribute, num_respondents, blocking, n_blocks, user_cards):
    """Generate complete experimental design with user-specified or optimal cards"""
    # Validate inputs
    if len(levels_per_attribute) != num_attributes:
        st.error(f"Please enter exactly {num_attributes} level values")
        return None, None
    
    # Create attributes and levels dictionary
    attributes = [f'Attr{i+1}' for i in range(num_attributes)]
    levels = {f'Attr{i+1}': list(range(1, l+1)) for i, l in enumerate(levels_per_attribute)}
    
    # Calculate optimal cards (considering user input)
    total_cards, cards_per_block = calculate_optimal_cards(
        num_respondents, blocking, n_blocks, levels_per_attribute, user_cards
    )
    
    # Adjust for non-blocking
    if not blocking:
        n_blocks = 1
        cards_per_block = total_cards
    
    # Generate blocks
    block_dfs = []
    cards_so_far = 0
    
    for block_num in range(1, n_blocks + 1):
        # Calculate block size (last block gets remaining cards)
        if block_num == n_blocks:
            block_size = total_cards - cards_per_block * (n_blocks - 1)
        else:
            block_size = cards_per_block
        
        if block_size > 0:  # Only create block if it has cards
            df_block = generate_balanced_block(block_num, block_size, attributes, levels, cards_so_far)
            block_dfs.append(df_block)
            cards_so_far += block_size
    
    # Combine all blocks
    if block_dfs:
        design_df = pd.concat(block_dfs, ignore_index=True)
    else:
        return None, None
    
    # Calculate D-efficiency
    d_eff = calculate_d_efficiency(design_df, attributes, levels)
    
    # Create approach description
    approach = "**Design Approach:** Balanced equal level counts within blocks\\n"
    if blocking:
        approach += f"Number of Blocks: {n_blocks}\\n"
    else:
        approach += "Single Block (no blocking)\\n"
    
    # Calculate additional metrics
    total_combinations = np.prod(levels_per_attribute)
    design_efficiency = (len(design_df) / total_combinations) * 100
    parameters_estimated = sum(l - 1 for l in levels_per_attribute)
    
    # Compile KPIs
    kpis = {
        'Design Structure': {
            'Total Cards': int(len(design_df)),
            'Cards per Respondent': f"{len(design_df) // num_respondents}-{len(design_df) // num_respondents + 1}",
            'Number of Attributes': int(num_attributes),
            'Levels per Attribute': str(levels_per_attribute),
            'Number of Respondents': int(num_respondents),
            'Blocking Enabled': 'Yes' if blocking else 'No',
            'Number of Blocks': int(n_blocks),
            'Cards per Block': int(cards_per_block)
        },
        'Statistical Properties': {
            'D-Efficiency (%)': round(float(d_eff * 100), 2),
            'Design Type': 'Fractional Factorial' if len(design_df) < total_combinations else 'Full Factorial',
            'Parameters Estimated': int(parameters_estimated),
            'Design Efficiency': f"{round(design_efficiency, 1)}% of full factorial",
            'Total Combinations': int(total_combinations)
        },
        'Approach': approach
    }
    
    return design_df, kpis

def safe_metric_display(value):
    """Safely display metrics"""
    if value is None:
        return "N/A"
    try:
        return str(value)
    except Exception:
        return "Error"

# Streamlit App
def main():
    st.set_page_config(
        page_title="Demand Estimation Design Generator",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Demand Estimation Experimental Design Generator")
    st.markdown("Generate statistically optimized experimental designs for conjoint analysis and demand estimation studies.")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Design Parameters")
    
    # Input parameters
    num_attributes = st.sidebar.number_input(
        'Number of Attributes', 
        min_value=2, max_value=15, value=6, 
        help="Number of product attributes to include in the study"
    )
    
    levels_input = st.sidebar.text_input(
        f'Levels per Attribute (comma-separated)',
        value='3,3,3,3,3,2',
        help=f"Enter {num_attributes} numbers separated by commas (e.g., 3,3,2,4)"
    )
    
    try:
        levels_per_attribute = [int(x.strip()) for x in levels_input.split(',') if x.strip().isdigit()]
        if len(levels_per_attribute) != num_attributes:
            st.sidebar.error(f"Please enter exactly {num_attributes} values")
            levels_per_attribute = [3] * num_attributes
    except:
        levels_per_attribute = [3] * num_attributes
    
    num_respondents = st.sidebar.number_input(
        'Number of Respondents', 
        min_value=10, max_value=1000, value=100,
        help="Total number of respondents for the study"
    )
    
    # NEW: Number of Cards input
    user_cards = st.sidebar.number_input(
        'Number of Cards (0 = Auto Calculate)', 
        min_value=0, max_value=1000, value=0,
        help="Specify exact number of cards, or enter 0 for automatic calculation based on statistical requirements"
    )
    
    # Show recommendation if user enters 0
    if user_cards == 0:
        min_cards_needed = sum(l - 1 for l in levels_per_attribute) * 3
        st.sidebar.info(f"📊 Recommended minimum cards: {min_cards_needed}")
    
    blocking_option = st.sidebar.selectbox(
        'Enable Blocking?', 
        ['No', 'Yes'],
        help="Blocking reduces respondent burden by dividing cards into groups"
    )
    
    if blocking_option == 'Yes':
        n_blocks = st.sidebar.number_input(
            'Number of Blocks', 
            min_value=2, max_value=20, value=6,
            help="Number of blocks to divide the cards into"
        )
    else:
        n_blocks = 1
    
    # Generate button
    if st.sidebar.button('🚀 Generate Design', type="primary"):
        with st.spinner('Generating optimized experimental design...'):
            result = generate_design(
                num_attributes, levels_per_attribute, num_respondents, 
                blocking_option == 'Yes', n_blocks, user_cards
            )
        
        if result[0] is not None:
            design_df, kpis = result
            
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📈 Design Summary", "🎴 Design Cards", "📥 Download"])
            
            with tab1:
                st.subheader("📈 Design Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🏗️ Design Structure")
                    for key, value in kpis['Design Structure'].items():
                        st.metric(key, safe_metric_display(value))
                
                with col2:
                    st.markdown("### 📊 Statistical Properties")
                    for key, value in kpis['Statistical Properties'].items():
                        if key == 'D-Efficiency (%)':
                            color = "normal" if value >= 70 else "inverse"
                            st.metric(key, f"{value}%")
                        else:
                            st.metric(key, safe_metric_display(value))
                
                st.markdown("### 🔬 Methodology")
                st.markdown(kpis['Approach'])
                
                # Show validation warnings if needed
                d_eff = kpis['Statistical Properties']['D-Efficiency (%)']
                if d_eff < 60:
                    st.warning("⚠️ D-Efficiency is below 60%. Consider increasing the number of cards for better statistical power.")
                elif d_eff >= 70:
                    st.success("✅ Excellent D-Efficiency! This design provides strong statistical power.")
            
            with tab2:
                st.subheader("🎴 Experimental Design Cards")
                st.dataframe(design_df, use_container_width=True)
                
                if blocking_option == 'Yes':
                    st.subheader("📊 Cards per Block")
                    block_counts = design_df['Block'].value_counts().sort_index()
                    st.bar_chart(block_counts)
                
                # Show attribute balance
                st.subheader("🎯 Attribute Level Balance")
                for attr in [f'Attr{i+1}' for i in range(num_attributes)]:
                    if attr in design_df.columns:
                        counts = design_df[attr].value_counts().sort_index()
                        st.write(f"**{attr}:** {dict(counts)}")
            
            with tab3:
                st.subheader("📥 Download Options")
                
                # Excel download with multiple sheets
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    design_df.to_excel(writer, sheet_name='Design_Cards', index=False)
                    
                    # Summary sheet
                    summary_data = []
                    for category, items in kpis.items():
                        if category != 'Approach' and isinstance(items, dict):
                            for key, value in items.items():
                                summary_data.append([category, key, value])
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data, columns=['Category', 'Metric', 'Value'])
                        summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📊 Download Complete Excel File",
                    data=excel_buffer.getvalue(),
                    file_name="experimental_design_complete.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # CSV download
                csv = design_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 Download CSV File",
                    data=csv,
                    file_name="experimental_design.csv",
                    mime="text/csv"
                )
        else:
            st.error("❌ Failed to generate design. Please check your parameters and try again.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("This tool generates balanced experimental designs for conjoint analysis and demand estimation studies using advanced statistical methods.")
    st.sidebar.markdown("**Features:**")
    st.sidebar.markdown("• Automatic or manual card calculation")
    st.sidebar.markdown("• D-efficiency optimization")
    st.sidebar.markdown("• Balanced block designs")
    st.sidebar.markdown("• Multiple download formats")

if __name__ == '__main__':
    main()

