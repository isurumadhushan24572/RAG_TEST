"""
Upload Sample Tickets to Weaviate with Local Embeddings
This script loads tickets_sample.json and uploads them with sentence-transformers embeddings
"""

import requests
import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ===================== CONFIGURATION =====================
API_BASE_URL = "http://localhost:8000"
SAMPLE_FILE = "tickets_sample.json"

# Initialize local embedding model (downloads on first use, ~80MB)
print("📦 Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded successfully")

# ===================== HELPER FUNCTIONS =====================

def generate_embedding(text: str):
    """Generate embedding vector for text using local model"""
    return embedding_model.encode(text).tolist()

def check_api_health():
    """Check if FastAPI is running and Weaviate is connected"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("weaviate_ready"):
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

def upload_single_ticket_with_embedding(ticket):
    """Upload a single ticket with local embedding to the API"""
    try:
        # Generate embedding from ticket content
        text_to_embed = f"{ticket['title']} {ticket['description']} {ticket['solution']}"
        embedding = generate_embedding(text_to_embed)
        
        # Add embedding to ticket data
        ticket_with_embedding = ticket.copy()
        ticket_with_embedding['_vector'] = embedding
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/tickets",
            json=ticket_with_embedding,
            timeout=30
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def upload_batch_tickets_with_embeddings(tickets):
    """Upload multiple tickets with embeddings in batch"""
    try:
        print(f"📦 Generating embeddings for {len(tickets)} tickets...")
        
        # Generate embeddings for all tickets
        tickets_with_embeddings = []
        for i, ticket in enumerate(tickets, 1):
            text_to_embed = f"{ticket['title']} {ticket['description']} {ticket['solution']}"
            embedding = generate_embedding(text_to_embed)
            
            ticket_with_embedding = ticket.copy()
            ticket_with_embedding['_vector'] = embedding
            tickets_with_embeddings.append(ticket_with_embedding)
            
            print(f"  ✅ Generated embedding {i}/{len(tickets)}")
        
        print(f"\n📤 Uploading {len(tickets_with_embeddings)} tickets with embeddings...")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/tickets/batch",
            json=tickets_with_embeddings,
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

# ===================== MAIN SCRIPT =====================

def main():
    """Main execution flow"""
    
    print("=" * 60)
    print("🎫 SUPPORT TICKETS UPLOADER (Local Embeddings)")
    print("=" * 60)
    print()
    
    # Step 1: Check API health
    print("Step 1: Checking API health...")
    if not check_api_health():
        print("\n❌ Cannot proceed without healthy API connection")
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
    
    # Step 4: Upload tickets with embeddings
    print("Step 4: Uploading tickets with local embeddings...")
    print()
    
    success = upload_batch_tickets_with_embeddings(tickets)
    
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
    print("=" * 60)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 60)
    print()
    print("🎯 Next Steps:")
    print("   1. View all tickets: GET http://localhost:8000/api/v1/tickets")
    print("   2. Search tickets: GET http://localhost:8000/api/v1/tickets/search?query=your+query")
    print("   3. Check Swagger UI: http://localhost:8000/docs")
    print()

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
