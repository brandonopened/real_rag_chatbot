#!/usr/bin/env python3
"""
Command Line Interface for the RAG Chatbot API
Usage: python cli_chat.py --username <user> --password <pass> --query "Your question"
"""

import argparse
import requests
import json
import sys
from typing import Optional

API_BASE_URL = "http://localhost:8000"

class ChatbotCLI:
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.token: Optional[str] = None
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate with the API and store the token"""
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password}
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                return True
            else:
                print(f"Login failed: {response.json().get('detail', 'Unknown error')}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return False
    
    def chat(self, query: str, profile_name: Optional[str] = None) -> dict:
        """Send a chat query to the API"""
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"query": query}
        if profile_name:
            payload["profile_name"] = profile_name
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers=headers
            )
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                raise RuntimeError(f"Chat request failed: {error_detail}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Connection error: {e}")
    
    def list_profiles(self) -> list:
        """Get list of available profiles"""
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")
        
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            response = requests.get(f"{self.base_url}/profiles", headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                raise RuntimeError(f"Failed to get profiles: {error_detail}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Connection error: {e}")

def main():
    parser = argparse.ArgumentParser(description="CLI for RAG Chatbot API")
    parser.add_argument("--username", "-u", required=True, help="Username for authentication")
    parser.add_argument("--password", "-p", required=True, help="Password for authentication")
    parser.add_argument("--query", "-q", help="Query to send to the chatbot")
    parser.add_argument("--profile", help="Profile name to use (optional)")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--url", default=API_BASE_URL, help="API base URL")
    
    args = parser.parse_args()
    
    # Initialize CLI client
    cli = ChatbotCLI(args.url)
    
    # Login
    print(f"Authenticating as {args.username}...")
    if not cli.login(args.username, args.password):
        sys.exit(1)
    print("Authentication successful!")
    
    # List profiles if requested
    if args.list_profiles:
        try:
            profiles = cli.list_profiles()
            print("\nAvailable profiles:")
            for profile in profiles:
                print(f"  - {profile['name']}: {profile.get('description', 'No description')}")
            return
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                query = input("\nYour question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                result = cli.chat(query, args.profile)
                print(f"\n[Profile: {result['profile_used']}]")
                print(f"Response: {result['response']}")
                if result['sources']:
                    print(f"Sources: {', '.join(result['sources'])}")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # Single query mode
    elif args.query:
        try:
            result = cli.chat(args.query, args.profile)
            print(f"\n[Profile: {result['profile_used']}]")
            print(f"Response: {result['response']}")
            if result['sources']:
                print(f"Sources: {', '.join(result['sources'])}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    else:
        print("Please provide either --query or --interactive flag. Use --help for more options.")

if __name__ == "__main__":
    main()