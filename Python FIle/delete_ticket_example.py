"""
Test Script: Delete Ticket by ticket_id
This script demonstrates how to delete a ticket from Weaviate using the DELETE endpoint
"""

import requests
import json

# ===================== CONFIGURATION =====================
API_BASE_URL = "http://localhost:8000"

# ===================== DELETE TICKET FUNCTION =====================
def delete_ticket(ticket_id: str):
    """
    Delete a ticket by its ticket_id
    
    Args:
        ticket_id: The unique ticket identifier (e.g., "TKT-2024-001")
    """
    print("=" * 60)
    print(f"🗑️  DELETING TICKET: {ticket_id}")
    print("=" * 60)
    print()
    
    try:
        # Send DELETE request
        print(f"📤 Sending DELETE request for ticket: {ticket_id}")
        response = requests.delete(
            f"{API_BASE_URL}/api/v1/tickets/{ticket_id}",
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Ticket deleted")
            print()
            print("📊 Deletion Details:")
            print(json.dumps(result, indent=2))
            print()
            return True
            
        elif response.status_code == 404:
            print(f"❌ ERROR: Ticket '{ticket_id}' not found")
            print(f"Response: {response.json()}")
            return False
            
        else:
            print(f"❌ ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to API")
        print("💡 Make sure FastAPI is running: python app_fastapi.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# ===================== CHECK IF TICKET EXISTS =====================
def check_ticket_exists(ticket_id: str):
    """Check if a ticket exists before deleting"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/tickets/{ticket_id}",
            timeout=5
        )
        
        if response.status_code == 200:
            ticket_data = response.json()["ticket"]
            print(f"✅ Ticket found:")
            print(f"   ID: {ticket_data['ticket_id']}")
            print(f"   Title: {ticket_data['title']}")
            print(f"   Status: {ticket_data['status']}")
            print()
            return True
        else:
            print(f"❌ Ticket '{ticket_id}' not found")
            return False
    except Exception as e:
        print(f"⚠️ Could not check ticket: {e}")
        return False

# ===================== GET ALL TICKETS =====================
def list_all_tickets():
    """List all available tickets to choose from"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/tickets?limit=10",
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            tickets = result.get("tickets", [])
            
            if len(tickets) == 0:
                print("⚠️ No tickets found in database")
                return []
            
            print(f"📋 Found {result['total_count']} total ticket(s)")
            print()
            print("Available tickets:")
            for ticket in tickets:
                print(f"   • {ticket['ticket_id']}: {ticket['title']}")
            print()
            return tickets
        else:
            print("⚠️ Could not list tickets")
            return []
    except Exception as e:
        print(f"⚠️ Error listing tickets: {e}")
        return []

# ===================== VERIFY DELETION =====================
def verify_deletion(ticket_id: str):
    """Verify that the ticket was actually deleted"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/tickets/{ticket_id}",
            timeout=5
        )
        
        if response.status_code == 404:
            print("✅ Verified: Ticket no longer exists in database")
            return True
        else:
            print("⚠️ Warning: Ticket still exists in database")
            return False
    except Exception as e:
        print(f"⚠️ Could not verify deletion: {e}")
        return False

# ===================== MAIN EXECUTION =====================
def main():
    print()
    print("=" * 60)
    print("🗑️  TICKET DELETION TOOL")
    print("=" * 60)
    print()
    
    # Check API health
    try:
        health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is healthy")
            print()
        else:
            print("⚠️ API health check warning")
            print()
    except:
        print("❌ Cannot connect to API - make sure it's running!")
        print("Run: python app_fastapi.py")
        return
    
    # List available tickets
    print("Step 1: Checking available tickets...")
    tickets = list_all_tickets()
    
    if len(tickets) == 0:
        print()
        print("💡 No tickets to delete. Upload some tickets first:")
        print("   python upload_single_ticket_example.py")
        return
    
    # Get ticket ID to delete (you can change this)
    ticket_id_to_delete = input("\nEnter ticket_id to delete (or press Enter to use first ticket): ").strip()
    
    if not ticket_id_to_delete and len(tickets) > 0:
        ticket_id_to_delete = tickets[0]['ticket_id']
        print(f"Using first ticket: {ticket_id_to_delete}")
    
    print()
    
    # Step 2: Check if ticket exists
    print("Step 2: Verifying ticket exists...")
    if not check_ticket_exists(ticket_id_to_delete):
        print()
        print("❌ Cannot delete - ticket not found")
        return
    
    # Step 3: Confirm deletion
    print("⚠️  WARNING: This will permanently delete the ticket!")
    confirm = input("Type 'yes' to confirm deletion: ").strip().lower()
    
    if confirm != 'yes':
        print()
        print("❌ Deletion cancelled by user")
        return
    
    print()
    
    # Step 4: Delete ticket
    print("Step 3: Deleting ticket...")
    success = delete_ticket(ticket_id_to_delete)
    
    if not success:
        print()
        print("❌ Deletion failed")
        return
    
    # Step 5: Verify deletion
    print("Step 4: Verifying deletion...")
    verify_deletion(ticket_id_to_delete)
    
    print()
    print("=" * 60)
    print("✅ DELETION COMPLETE!")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("   1. Check remaining tickets: GET http://localhost:8000/api/v1/tickets")
    print("   2. Check count: GET http://localhost:8000/api/v1/collections/SupportTickets/count")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
