import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
from outlook_manager import OutlookManager
from ai_processor import AIProcessor

def main():
    st.set_page_config(
        page_title="AI Email Assistant",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("🤖 AI Email Assistant for Outlook")
    st.markdown("Automatically categorize and summarize your unread emails")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'processed_emails' not in st.session_state:
        st.session_state.processed_emails = []
    
    # Sidebar for authentication and settings
    with st.sidebar:
        st.header("🔐 Authentication")
        
        if not st.session_state.authenticated:
            if st.button("Login to Outlook", type="primary"):
                try:
                    outlook_manager = OutlookManager()
                    if outlook_manager.authenticate():
                        st.session_state.authenticated = True
                        st.session_state.outlook_manager = outlook_manager
                        st.success("Successfully authenticated!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
        else:
            st.success("✅ Authenticated")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.processed_emails = []
                st.rerun()
    
    if st.session_state.authenticated:
        # Main interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("📅 Time Period Selection")
            
            # Date range selection
            start_date = st.date_input(
                "Start Date",
                value=datetime.now().date() - timedelta(days=7),
                max_value=datetime.now().date()
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )
            
            # Quick date options
            st.subheader("Quick Options")
            if st.button("Last 24 hours"):
                start_date = datetime.now().date() - timedelta(days=1)
                end_date = datetime.now().date()
            
            if st.button("Last 7 days"):
                start_date = datetime.now().date() - timedelta(days=7)
                end_date = datetime.now().date()
            
            if st.button("Last 30 days"):
                start_date = datetime.now().date() - timedelta(days=30)
                end_date = datetime.now().date()
            
            # Process emails button
            if st.button("🔍 Process Unread Emails", type="primary"):
                process_emails(start_date, end_date)
        
        with col2:
            st.header("📧 Email Summary")
            display_email_results()
    
    else:
        st.info("Please authenticate with your Outlook account to begin processing emails.")
        
        # Display features
        st.header("🌟 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Smart Categorization**
            - Official/Personal/Spam/Advertisement
            - AI-powered classification
            """)
        
        with col2:
            st.markdown("""
            **⚡ Priority Assessment**
            - High/Medium/Low/Not Important
            - Urgency evaluation
            """)
        
        with col3:
            st.markdown("""
            **📝 Auto Summary**
            - 2-3 sentence summaries
            - Response requirement analysis
            """)

def process_emails(start_date, end_date):
    """Process unread emails for the specified date range"""
    
    with st.spinner("Fetching unread emails..."):
        try:
            outlook_manager = st.session_state.outlook_manager
            ai_processor = AIProcessor()
            
            # Get unread emails from Outlook[3][4]
            unread_emails = outlook_manager.get_unread_emails(start_date, end_date)
            
            if not unread_emails:
                st.warning("No unread emails found in the specified date range.")
                return
            
            processed_results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, email in enumerate(unread_emails):
                status_text.text(f"Processing email {i+1} of {len(unread_emails)}")
                
                # Process email with AI[7][10]
                email_analysis = ai_processor.analyze_email(email)
                
                processed_email = {
                    'mail_from': email.get('from', {}).get('emailAddress', {}).get('name', 'Unknown'),
                    'mail_type': email_analysis.get('type', 'Unknown'),
                    'mail_importance': email_analysis.get('importance', 'Medium'),
                    'mail_subject': email.get('subject', 'No Subject'),
                    'mail_summary': email_analysis.get('summary', 'No summary available'),
                    'required_response': email_analysis.get('response_required', 'Maybe'),
                    'urgency': email_analysis.get('urgency', 'Anytime'),
                    'received_date': email.get('receivedDateTime', ''),
                    'email_id': email.get('id', '')
                }
                
                processed_results.append(processed_email)
                progress_bar.progress((i + 1) / len(unread_emails))
            
            st.session_state.processed_emails = processed_results
            st.success(f"Successfully processed {len(processed_results)} emails!")
            
        except Exception as e:
            st.error(f"Error processing emails: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

def display_email_results():
    """Display processed email results"""
    
    if not st.session_state.processed_emails:
        st.info("No processed emails to display. Please process emails first.")
        return
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(st.session_state.processed_emails)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", len(df))
    
    with col2:
        high_priority = len(df[df['mail_importance'] == 'High'])
        st.metric("High Priority", high_priority)
    
    with col3:
        urgent = len(df[df['urgency'] == 'Immediate'])
        st.metric("Urgent", urgent)
    
    with col4:
        response_required = len(df[df['required_response'] == 'Yes'])
        st.metric("Response Required", response_required)
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        type_filter = st.selectbox(
            "Mail Type", 
            ['All'] + list(df['mail_type'].unique())
        )
    
    with col2:
        importance_filter = st.selectbox(
            "Importance", 
            ['All'] + list(df['mail_importance'].unique())
        )
    
    with col3:
        urgency_filter = st.selectbox(
            "Urgency", 
            ['All'] + list(df['urgency'].unique())
        )
    
    # Apply filters
    filtered_df = df.copy()
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['mail_type'] == type_filter]
    if importance_filter != 'All':
        filtered_df = filtered_df[filtered_df['mail_importance'] == importance_filter]
    if urgency_filter != 'All':
        filtered_df = filtered_df[filtered_df['urgency'] == urgency_filter]
    
    # Display results
    st.subheader(f"📧 Email Details ({len(filtered_df)} emails)")
    
    for idx, email in filtered_df.iterrows():
        with st.expander(f"📧 {email['mail_subject'][:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Mail From:** {email['mail_from']}")
                st.markdown(f"**Mail Type:** {email['mail_type']}")
                st.markdown(f"**Mail Importance:** {email['mail_importance']}")
                st.markdown(f"**Required Response:** {email['required_response']}")
            
            with col2:
                st.markdown(f"**Mail Subject:** {email['mail_subject']}")
                st.markdown(f"**Urgency:** {email['urgency']}")
                st.markdown(f"**Received:** {email['received_date'][:10]}")
            
            st.markdown(f"**Mail Summary:** {email['mail_summary']}")
    
    # Export functionality
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"email_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download as JSON",
            data=json_data,
            file_name=f"email_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
