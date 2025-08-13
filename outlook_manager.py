import streamlit as st
import requests
import msal
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class OutlookManager:
    def __init__(self):
        self.client_id = st.secrets["microsoft"]["client_id"]
        self.client_secret = st.secrets["microsoft"]["client_secret"]
        self.tenant_id = st.secrets["microsoft"]["tenant_id"]
        
        # Microsoft Graph API endpoints
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        # FIXED: Changed scope to use .default format for client credentials flow
        self.scope = ["https://graph.microsoft.com/.default"]
        self.graph_url = "https://graph.microsoft.com/v1.0"
        
        self.access_token = None
        self.app = None
    
    def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API using Client Credentials Flow"""
        try:
            # Create MSAL app for client credentials flow
            self.app = msal.ConfidentialClientApplication(
                self.client_id,
                authority=self.authority,
                client_credential=self.client_secret
            )
            
            # Use acquire_token_for_client for client credentials flow
            result = self.app.acquire_token_for_client(scopes=self.scope)
            
            if "access_token" in result:
                self.access_token = result["access_token"]
                return True
            else:
                error_msg = result.get('error_description', 'Unknown error')
                st.error(f"Authentication failed: {error_msg}")
                return False
                
        except Exception as e:
            st.error(f"Authentication error: {str(e)}")
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
            return self.authenticate()
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
