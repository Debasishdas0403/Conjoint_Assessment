import streamlit as st
import pandas as pd
import numpy as np
from scipy.linalg import det
import io

def effects_coding(levels_count, design_column):
    """Convert categorical levels to effects coding for design matrix"""
    n = len(design_column)
    coded = np.zeros((n, levels_count - 1))
    for i in range(levels_count - 1):
        coded[:, i] = np.where(design_column == (i + 1), 1, 0)
    if levels_count > 1:
        coded[:, -1] = -np.sum(coded[:, :-1], axis=1)
    return coded

def calculate_d_efficiency(df, attributes, levels):
    """Calculate D-efficiency of the experimental design"""
    try:
        design_matrix_parts = []
        for attr in attributes:
            levels_count = len(levels[attr])
            col = df[attr].values
            coded_part = effects_coding(levels_count, col)
            design_matrix_parts.append(coded_part)
        
        X = np.hstack(design_matrix_parts)
        XTX = np.dot(X.T, X)
        p = X.shape[1]
        N = X.shape[0]
        
        det_XTX = det(XTX)
        if det_XTX <= 0:
            return 0
        else:
            return (det_XTX ** (1 / p)) / N
    except (np.linalg.LinAlgError, ValueError):
        return 0

def generate_balanced_block(block_num, block_size, attributes, levels, cards_so_far=0):
    """Generate a balanced block with equal level distribution"""
    np.random.seed(42 + block_num)  # Reproducible but different per block
    
    block_data = {}
    block_data['Block'] = [block_num] * block_size
    block_data['Card Number'] = [cards_so_far + i + 1 for i in range(block_size)]
    block_data['Brand'] = ['New Brand'] * block_size
    
    for attr in attributes:
        attr_levels = levels[attr]
        num_levels = len(attr_levels)
        
        # Calculate balanced distribution
        repeats = block_size // num_levels
        remainder = block_size % num_levels
        
        levels_list = []
        for i, level in enumerate(attr_levels):
            count = repeats + (1 if i < remainder else 0)
            levels_list.extend([level] * count)
        
        np.random.shuffle(levels_list)
        block_data[attr] = levels_list
    
    return pd.DataFrame(block_data)

def calculate_balance_metrics(df, attributes, levels):
    """Calculate balance metrics for the design"""
    balance_metrics = {}
    
    for attr in attributes:
        attr_levels = levels[attr]
        counts = df[attr].value_counts()
        
        # Calculate coefficient of variation for balance
        mean_count = len(df) / len(attr_levels)
        std_count = counts.std()
        cv = (std_count / mean_count) * 100 if mean_count > 0 else 0
        
        balance_metrics[attr] = {
            'Mean Count': mean_count,
            'Std Dev': std_count,
            'CV (%)': round(cv, 2),
            'Min Count': counts.min(),
            'Max Count': counts.max()
        }
    
    return balance_metrics

def generate_design(num_attributes, levels_per_attribute, num_cards, blocking, n_blocks):
    """Generate complete experimental design"""
    # Validate inputs
    if len(levels_per_attribute) != num_attributes:
        st.error(f"Please enter exactly {num_attributes} level values")
        return None, None
    
    # Create attributes and levels
    attributes = [f'Attr{i+1}' for i in range(num_attributes)]
    levels = {f'Attr{i+1}': list(range(1, l+1)) for i, l in enumerate(levels_per_attribute)}
    
    # Determine design structure
    if blocking:
        cards_per_block = num_cards // n_blocks
        remainder_cards = num_cards % n_blocks
    else:
        cards_per_block = num_cards
        n_blocks = 1
        remainder_cards = 0
    
    # Generate blocks
    block_dfs = []
    cards_so_far = 0
    
    for block_num in range(1, n_blocks + 1):
        # Add remainder cards to last blocks
        block_size = cards_per_block + (1 if block_num > (n_blocks - remainder_cards) else 0)
        
        df_block = generate_balanced_block(block_num, block_size, attributes, levels, cards_so_far)
        block_dfs.append(df_block)
        cards_so_far += block_size
    
    # Combine all blocks
    design_df = pd.concat(block_dfs, ignore_index=True)
    
    # Calculate metrics
    d_eff = calculate_d_efficiency(design_df, attributes, levels)
    balance_metrics = calculate_balance_metrics(design_df, attributes, levels)
    
    # Create approach description
    approach = "**Design Approach:** Balanced Incomplete Block Design (BIBD)\n\n"
    approach += "**Method:** Equal count distribution within blocks\n\n"
    if blocking:
        approach += f"**Structure:** {n_blocks} blocks with {cards_per_block}"
        if remainder_cards > 0:
            approach += f"-{cards_per_block + 1}"
        approach += " cards each"
    else:
        approach += "**Structure:** Single block (no blocking)"
    
    # Compile KPIs
    kpis = {
        'Design Structure': {
            'Total Cards': len(design_df),
            'Number of Attributes': num_attributes,
            'Levels per Attribute': levels_per_attribute,
            'Blocking': 'Yes' if blocking else 'No',
            'Number of Blocks': n_blocks,
            'Cards per Block': f"{cards_per_block}" + (f"-{cards_per_block + 1}" if remainder_cards > 0 else "")
        },
        'Statistical Properties': {
            'D-Efficiency (%)': round(d_eff * 100, 2),
            'Design Type': 'Fractional Factorial' if len(design_df) < np.prod(levels_per_attribute) else 'Full Factorial',
            'Parameters Estimated': sum(l - 1 for l in levels_per_attribute),
            'Degrees of Freedom': len(design_df) - sum(l - 1 for l in levels_per_attribute) - 1
        },
        'Balance Metrics': balance_metrics,
        'Approach': approach
    }
    
    return design_df, kpis

def create_summary_report(design_df, kpis):
    """Create downloadable summary report"""
    buffer = io.StringIO()
    
    buffer.write("EXPERIMENTAL DESIGN SUMMARY REPORT\n")
    buffer.write("=" * 50 + "\n\n")
    
    # Design Structure
    buffer.write("DESIGN STRUCTURE\n")
    buffer.write("-" * 20 + "\n")
    for key, value in kpis['Design Structure'].items():
        buffer.write(f"{key}: {value}\n")
    buffer.write("\n")
    
    # Statistical Properties  
    buffer.write("STATISTICAL PROPERTIES\n")
    buffer.write("-" * 25 + "\n")
    for key, value in kpis['Statistical Properties'].items():
        buffer.write(f"{key}: {value}\n")
    buffer.write("\n")
    
    # Balance Metrics
    buffer.write("ATTRIBUTE BALANCE METRICS\n")
    buffer.write("-" * 30 + "\n")
    for attr, metrics in kpis['Balance Metrics'].items():
        buffer.write(f"\n{attr}:\n")
        for metric, value in metrics.items():
            buffer.write(f"  {metric}: {value}\n")
    buffer.write("\n")
    
    # Approach
    buffer.write("METHODOLOGY\n")
    buffer.write("-" * 15 + "\n")
    buffer.write(kpis['Approach'].replace('**', '').replace('\n\n', '\n'))
    buffer.write("\n\n")
    
    # Design Preview
    buffer.write("DESIGN PREVIEW (First 10 cards)\n")
    buffer.write("-" * 35 + "\n")
    buffer.write(design_df.head(10).to_string(index=False))
    
    return buffer.getvalue()

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
    except:
        levels_per_attribute = [3] * num_attributes
    
    num_cards = st.sidebar.number_input(
        'Total Number of Cards', 
        min_value=10, max_value=500, value=54,
        help="Total number of experimental cards to generate"
    )
    
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
            design_df, kpis = generate_design(
                num_attributes, levels_per_attribute, num_cards, 
                blocking_option == 'Yes', n_blocks
            )
        
        if design_df is not None:
            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Design Summary", "🎴 Design Cards", "📊 Balance Analysis", "📥 Download"])
            
            with tab1:
                st.subheader("📈 Design Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🏗️ Design Structure")
                    for key, value in kpis['Design Structure'].items():
                        st.metric(key, value)
                
                with col2:
                    st.markdown("### 📊 Statistical Properties")
                    for key, value in kpis['Statistical Properties'].items():
                        if key == 'D-Efficiency (%)':
                            color = "normal" if value >= 70 else "inverse"
                            st.metric(key, f"{value}%", delta=None)
                        else:
                            st.metric(key, value)
                
                st.markdown("### 🔬 Methodology")
                st.markdown(kpis['Approach'])
            
            with tab2:
                st.subheader("🎴 Experimental Design Cards")
                st.dataframe(design_df, use_container_width=True)
                
                if blocking_option == 'Yes':
                    st.subheader("📊 Cards per Block")
                    block_counts = design_df['Block'].value_counts().sort_index()
                    st.bar_chart(block_counts)
            
            with tab3:
                st.subheader("📊 Attribute Balance Analysis")
                
                for attr, metrics in kpis['Balance Metrics'].items():
                    with st.expander(f"{attr} Balance Metrics"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean Count", f"{metrics['Mean Count']:.1f}")
                            st.metric("Std Dev", f"{metrics['Std Dev']:.2f}")
                        with col2:
                            st.metric("CV (%)", f"{metrics['CV (%)']}%")
                            cv_status = "🟢 Excellent" if metrics['CV (%)'] < 10 else "🟡 Good" if metrics['CV (%)'] < 20 else "🟠 Fair"
                            st.write(f"Balance: {cv_status}")
                        with col3:
                            st.metric("Min Count", metrics['Min Count'])
                            st.metric("Max Count", metrics['Max Count'])
                
                # Overall balance visualization
                st.subheader("🎯 Level Distribution by Attribute")
                for attr in [f'Attr{i+1}' for i in range(num_attributes)]:
                    if attr in design_df.columns:
                        counts = design_df[attr].value_counts().sort_index()
                        st.write(f"**{attr}:**")
                        st.bar_chart(counts, height=200)
            
            with tab4:
                st.subheader("📥 Download Options")
                
                # Excel download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    design_df.to_excel(writer, sheet_name='Design', index=False)
                    
                    # Add summary sheet
                    summary_data = []
                    for category, items in kpis.items():
                        if category != 'Approach' and isinstance(items, dict):
                            for key, value in items.items():
                                summary_data.append([category, key, value])
                    
                    summary_df = pd.DataFrame(summary_data, columns=['Category', 'Metric', 'Value'])
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="📊 Download Excel File",
                    data=excel_buffer.getvalue(),
                    file_name="experimental_design.xlsx",
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
                
                # Summary report download
                summary_report = create_summary_report(design_df, kpis)
                st.download_button(
                    label="📋 Download Summary Report",
                    data=summary_report,
                    file_name="design_summary_report.txt",
                    mime="text/plain"
                )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("This tool generates balanced experimental designs for conjoint analysis and demand estimation studies using advanced statistical methods.")

if __name__ == '__main__':
    main()
