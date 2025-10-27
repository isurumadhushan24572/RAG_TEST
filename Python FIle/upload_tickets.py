"""
Upload Sample Tickets to Weaviate via FastAPI
This script loads tickets_sample.json and uploads them to the SupportTickets collection
"""

import requests
import json
import sys
from pathlib import Path

# ===================== CONFIGURATION =====================
API_BASE_URL = "http://localhost:8000"
SAMPLE_FILE = "tickets_sample.json"

# ===================== HELPER FUNCTIONS =====================

def check_api_health():
    """Check if FastAPI is running and Weaviate is connected"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("weaviate_connected"):
                print("✅ API is healthy and Weaviate is connected")
                return True
            else:
                print("❌ API is running but Weaviate is not connected")
                print("💡 Make sure Docker container is running: docker-compose up -d")
                return False
        else:
            print(f"❌ API health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to FastAPI")
        print("💡 Make sure FastAPI is running: python app_fastapi.py")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def load_tickets_from_file(filename):
    """Load tickets from JSON file"""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            tickets = json.load(f)
        
        print(f"✅ Loaded {len(tickets)} tickets from {filename}")
        return tickets
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {filename}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def upload_single_ticket(ticket):
    """Upload a single ticket to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/tickets",
            json=ticket,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_batch_tickets(tickets):
    """Upload multiple tickets in batch"""
    try:
        print(f"📦 Uploading {len(tickets)} tickets in batch...")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/tickets/batch",
            json=tickets,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch upload completed!")
            print(f"   📊 Total: {result['total_tickets']}")
            print(f"   ✅ Uploaded: {result['uploaded']}")
            print(f"   ❌ Failed: {result['failed']}")
            
            if result['failed'] > 0 and result.get('failed_tickets'):
                print(f"\n⚠️ Failed tickets:")
                for failed in result['failed_tickets']:
                    print(f"   - {failed['ticket_id']}: {failed['error']}")
            
            return True
        else:
            print(f"❌ Batch upload failed: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Batch upload error: {e}")
        return False

def get_collection_count():
    """Get current count of tickets in Weaviate"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/collections/SupportTickets/count",
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("document_count", 0)
        else:
            return None
    except Exception as e:
        print(f"⚠️ Could not get collection count: {e}")
        return None

def search_tickets(query, limit=3):
    """Test search functionality"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/tickets/search",
            params={"query": query, "limit": limit},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return None

# ===================== MAIN SCRIPT =====================

def main():
    """Main execution flow"""
    
    print("=" * 60)
    print("🎫 SUPPORT TICKETS UPLOADER")
    print("=" * 60)
    print()
    
    # Step 1: Check API health
    print("Step 1: Checking API health...")
    if not check_api_health():
        print("\n❌ Cannot proceed without healthy API connection")
        print("\n💡 Troubleshooting:")
        print("   1. Start Weaviate: docker-compose up -d")
        print("   2. Wait 30 seconds")
        print("   3. Start FastAPI: python app_fastapi.py")
        print("   4. Run this script again")
        sys.exit(1)
    
    print()
    
    # Step 2: Check current collection count
    print("Step 2: Checking current tickets in database...")
    current_count = get_collection_count()
    if current_count is not None:
        print(f"📊 Current ticket count: {current_count}")
        
        if current_count > 0:
            response = input(f"\n⚠️ Collection already has {current_count} tickets. Continue? (y/n): ")
            if response.lower() != 'y':
                print("❌ Upload cancelled by user")
                sys.exit(0)
    
    print()
    
    # Step 3: Load tickets from file
    print(f"Step 3: Loading tickets from {SAMPLE_FILE}...")
    tickets = load_tickets_from_file(SAMPLE_FILE)
    
    if tickets is None or len(tickets) == 0:
        print("❌ No tickets to upload")
        sys.exit(1)
    
    print()
    
    # Step 4: Upload tickets
    print("Step 4: Uploading tickets to Weaviate...")
    
    # Ask user for upload method
    print("\nChoose upload method:")
    print("  1. Batch upload (faster, recommended)")
    print("  2. Individual upload (slower, detailed progress)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    print()
    
    if choice == "1":
        # Batch upload
        success = upload_batch_tickets(tickets)
    else:
        # Individual upload
        print(f"📝 Uploading {len(tickets)} tickets individually...")
        success_count = 0
        failed_count = 0
        
        for i, ticket in enumerate(tickets, 1):
            ticket_id = ticket.get('ticket_id', 'Unknown')
            print(f"   [{i}/{len(tickets)}] Uploading {ticket_id}...", end=" ")
            
            success, result = upload_single_ticket(ticket)
            
            if success:
                print("✅")
                success_count += 1
            else:
                print(f"❌ {result.get('detail', 'Unknown error')}")
                failed_count += 1
        
        print(f"\n✅ Successfully uploaded: {success_count}")
        print(f"❌ Failed: {failed_count}")
        success = (success_count > 0)
    
    if not success:
        print("\n❌ Upload failed")
        sys.exit(1)
    
    print()
    
    # Step 5: Verify upload
    print("Step 5: Verifying upload...")
    new_count = get_collection_count()
    
    if new_count is not None:
        print(f"✅ New ticket count: {new_count}")
        if current_count is not None:
            added = new_count - current_count
            print(f"📈 Tickets added: {added}")
    
    print()
    
    # Step 6: Test search
    print("Step 6: Testing vector search...")
    test_query = "payment api error 500"
    print(f"🔍 Searching for: '{test_query}'")
    
    results = search_tickets(test_query, limit=3)
    
    if results and results.get('results'):
        print(f"✅ Found {results['results_count']} similar tickets:")
        for i, ticket in enumerate(results['results'], 1):
            score = ticket.get('similarity_score', 0)
            ticket_id = ticket.get('ticket_id', 'Unknown')
            title = ticket.get('title', 'No title')
            print(f"   {i}. {ticket_id}: {title}")
            print(f"      Similarity: {score:.2%}")
    else:
        print("⚠️ No search results (this is normal if vectorization is still in progress)")
    
    print()
    print("=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("   1. View all tickets: GET http://localhost:8000/api/v1/tickets")
    print("   2. Search tickets: GET http://localhost:8000/api/v1/tickets/search?query=your+query")
    print("   3. Get specific ticket: GET http://localhost:8000/api/v1/tickets/TKT-0001")
    print("   4. Check Swagger UI: http://localhost:8000/docs")
    print()

# ===================== RUN SCRIPT =====================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Upload cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
