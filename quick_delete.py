"""
Quick Delete - Simple one-file script to delete a ticket
Usage: python quick_delete.py TKT-2024-001
"""

import requests
import sys

if len(sys.argv) < 2:
    print("❌ Usage: python quick_delete.py <ticket_id>")
    print("Example: python quick_delete.py TKT-2024-001")
    sys.exit(1)

ticket_id = sys.argv[1]
print(f"🗑️  Deleting ticket: {ticket_id}")

try:
    response = requests.delete(f"http://localhost:8000/api/v1/tickets/{ticket_id}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {result['message']}")
        print(f"   Title: {result['deleted_ticket']['title']}")
    elif response.status_code == 404:
        print(f"❌ Ticket '{ticket_id}' not found")
    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API. Is FastAPI running?")
    print("   Run: python app_fastapi.py")
except Exception as e:
    print(f"❌ Error: {e}")
