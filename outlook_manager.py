import streamlit as st
import requests
import msal
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import webbrowser
import urllib.parse

class OutlookManager:
    def __init__(self):
        self.client_id = st.secrets["microsoft"]["client_id"]
        self.client_secret = st.secrets["microsoft"]["client_secret"]
        self.tenant_id = st.secrets["microsoft"]["tenant_id"]
        self.redirect_uri = "http://localhost:8501"
        
        # Microsoft Graph API endpoints
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        # FIXED: Use delegated permissions for user email access
        self.scope = [
            "https://graph.microsoft.com/Mail.Read",
            "https://graph.microsoft.com/User.Read"
        ]
        self.graph_url = "https://graph.microsoft.com/v1.0"
        
        self.access_token = None
        self.app = None
    
    def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API using Authorization Code Flow"""
        try:
            # Create MSAL app for authorization code flow
            self.app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=self.authority,
                client_credential=self.client_secret
            )
            
            # Try to get token from cache first
            accounts = self.app.get_accounts()
            if accounts:
                result = self.app.acquire_token_silent(self.scope, account=accounts[0])
                if result and "access_token" in result:
                    self.access_token = result["access_token"]
                    return True
            
            # If no cached token, use device code flow for easier authentication
            flow = self.app.initiate_device_flow(scopes=self.scope)
            
            if "user_code" not in flow:
                raise ValueError("Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))
            
            # Display the device code to user
            st.info(f"**Authentication Required**")
            st.markdown(f"1. Go to: **{flow['verification_uri']}**")
            st.markdown(f"2. Enter this code: **{flow['user_code']}**")
            st.markdown("3. Complete the sign-in process in your browser")
            st.markdown("4. Click 'Complete Authentication' button below after signing in")
            
            # Store flow in session state for completion
            st.session_state.auth_flow = flow
            
            return False  # Return False to show completion button
            
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
            return False
    
    def complete_authentication(self) -> bool:
        """Complete the device flow authentication"""
        try:
            if 'auth_flow' not in st.session_state:
                st.error("No authentication flow found. Please start authentication again.")
                return False
            
            flow = st.session_state.auth_flow
            
            # Complete the device flow
            result = self.app.acquire_token_by_device_flow(flow)
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                # Clear the auth flow from session state
                del st.session_state.auth_flow
                return True
            else:
                error_msg = result.get('error_description', 'Unknown error')
                st.error(f"Authentication failed: {error_msg}")
                return False
                
        except Exception as e:
            st.error(f"Authentication completion error: {str(e)}")
            return False
    
    def is_token_valid(self) -> bool:
        """Check if the current access token is valid"""
        if not self.access_token:
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            # Test the token with a simple API call
            url = f"{self.graph_url}/me"
            response = requests.get(url, headers=headers)
            return response.status_code == 200
            
        except Exception:
            return False

    def refresh_token_if_needed(self) -> bool:
        """Refresh token if it's expired"""
        if not self.is_token_valid():
            # Try to get a new token silently
            if self.app:
                accounts = self.app.get_accounts()
                if accounts:
                    result = self.app.acquire_token_silent(self.scope, account=accounts[0])
                    if result and "access_token" in result:
                        self.access_token = result["access_token"]
                        return True
            return False
        return True
    
    def get_unread_emails(self, start_date: datetime.date, end_date: datetime.date) -> List[Dict]:
        """Fetch unread emails from Outlook for specified date range"""
        if not self.access_token:
            raise Exception("Not authenticated. Please authenticate first.")
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            # Format dates for Microsoft Graph API
            start_datetime = datetime.combine(start_date, datetime.min.time()).isoformat() + 'Z'
            end_datetime = datetime.combine(end_date, datetime.max.time()).isoformat() + 'Z'
            
            # Build filter query for unread emails in date range
            filter_query = f"isRead eq false and receivedDateTime ge {start_datetime} and receivedDateTime le {end_datetime}"
            
            # API endpoint with filter
            url = f"{self.graph_url}/me/messages"
            params = {
                '$filter': filter_query,
                '$select': 'id,subject,from,receivedDateTime,bodyPreview,importance,isRead',
                '$orderby': 'receivedDateTime desc',
                '$top': 50  # Limit to 50 emails per request
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            emails = data.get('value', [])
            
            # Handle pagination if there are more emails
            while '@odata.nextLink' in data and len(emails) < 200:  # Limit total to 200
                response = requests.get(data['@odata.nextLink'], headers=headers)
                response.raise_for_status()
                data = response.json()
                emails.extend(data.get('value', []))
            
            return emails
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching emails: {str(e)}")
    
    def mark_email_as_read(self, email_id: str) -> bool:
        """Mark an email as read"""
        if not self.access_token:
            return False
        
        try:
            headers = {
                'Authorization': f'Bearer {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            url = f"{self.graph_url}/me/messages/{email_id}"
            data = {'isRead': True}
            
            response = requests.patch(url, headers=headers, json=data)
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException:
            return False
