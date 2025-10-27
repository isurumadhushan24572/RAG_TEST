# 🗑️ Delete Ticket API - Quick Reference

## ✅ API Endpoint Created

```
DELETE /api/v1/tickets/{ticket_id}
```

---

## 🚀 How to Use

### **Method 1: Using Python Script** (Easiest)

```powershell
python delete_ticket_example.py
```

This script will:
- ✅ List all available tickets
- ✅ Let you choose which ticket to delete
- ✅ Verify ticket exists before deleting
- ✅ Confirm deletion (safety check)
- ✅ Verify deletion was successful

---

### **Method 2: Using Python requests**

```python
import requests

# Delete a specific ticket
ticket_id = "TKT-2024-001"
response = requests.delete(f"http://localhost:8000/api/v1/tickets/{ticket_id}")

if response.status_code == 200:
    result = response.json()
    print(f"✅ Deleted: {result['message']}")
else:
    print(f"❌ Error: {response.json()['detail']}")
```

---

### **Method 3: Using PowerShell**

```powershell
# Delete a ticket by ticket_id
$ticketId = "TKT-2024-001"
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/$ticketId" -Method Delete
```

---

### **Method 4: Using Swagger UI**

1. Open: http://localhost:8000/docs
2. Find: `DELETE /api/v1/tickets/{ticket_id}`
3. Click: **"Try it out"**
4. Enter: Your `ticket_id` (e.g., `TKT-2024-001`)
5. Click: **"Execute"**

---

## 📊 Response Format

### **Success Response (200)**

```json
{
  "success": true,
  "message": "Ticket 'TKT-2024-001' deleted successfully",
  "deleted_ticket": {
    "ticket_id": "TKT-2024-001",
    "uuid": "abc123-def456-...",
    "title": "Payment API Error"
  }
}
```

### **Ticket Not Found (404)**

```json
{
  "detail": "Ticket 'TKT-2024-999' not found in database"
}
```

### **Database Not Connected (503)**

```json
{
  "detail": "Weaviate vector database is not connected. Please check connection."
}
```

---

## 🔍 Complete Example Workflow

```powershell
# 1. Start FastAPI
python app_fastapi.py

# 2. List all tickets to see what's available
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets" | 
    Select-Object -ExpandProperty tickets | 
    Select-Object ticket_id, title

# Output:
# ticket_id      title
# ---------      -----
# TKT-0001       Database timeout issue
# TKT-0002       Login page not loading

# 3. Delete a specific ticket
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-0001" -Method Delete

# 4. Verify it's deleted (should return 404)
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-0001"
# Error: Ticket 'TKT-0001' not found

# 5. Check remaining count
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/collections/SupportTickets/count"
```

---

## 🎯 Quick Commands

```powershell
# Delete ticket TKT-2024-001
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-2024-001" -Method Delete

# Delete ticket and save response
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-2024-001" -Method Delete
Write-Host $result.message

# Delete with error handling
try {
    Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-2024-001" -Method Delete
    Write-Host "✅ Ticket deleted successfully"
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)"
}
```

---

## 🧪 Test the Endpoint

### Create and Delete a Test Ticket

```python
import requests

API_URL = "http://localhost:8000"

# 1. Create a test ticket
test_ticket = {
    "ticket_id": "TEST-DELETE-001",
    "title": "Test ticket for deletion",
    "description": "This ticket will be deleted",
    "category": "Testing",
    "status": "Open",
    "severity": "Low",
    "application": "Test App",
    "affected_users": "None",
    "environment": "Development",
    "solution": "Will be deleted",
    "reasoning": "Testing delete functionality",
    "timestamp": "2025-10-27T12:00:00Z"
}

# Upload test ticket
print("📤 Creating test ticket...")
response = requests.post(f"{API_URL}/api/v1/tickets", json=test_ticket)
print(f"✅ Created: {response.json()['message']}")

# 2. Verify it exists
print("\n🔍 Verifying ticket exists...")
response = requests.get(f"{API_URL}/api/v1/tickets/TEST-DELETE-001")
print(f"✅ Found: {response.json()['ticket']['title']}")

# 3. Delete the ticket
print("\n🗑️  Deleting ticket...")
response = requests.delete(f"{API_URL}/api/v1/tickets/TEST-DELETE-001")
print(f"✅ {response.json()['message']}")

# 4. Verify deletion
print("\n🔍 Verifying deletion...")
try:
    response = requests.get(f"{API_URL}/api/v1/tickets/TEST-DELETE-001")
    print("❌ Error: Ticket still exists!")
except:
    print("✅ Confirmed: Ticket deleted successfully")
```

---

## ⚙️ API Endpoint Details

**Endpoint:** `DELETE /api/v1/tickets/{ticket_id}`

**Parameters:**
- `ticket_id` (path parameter, required): The unique ticket identifier

**Returns:**
- `200 OK`: Ticket deleted successfully
- `404 Not Found`: Ticket doesn't exist
- `503 Service Unavailable`: Database not connected
- `500 Internal Server Error`: Deletion failed

**How it works:**
1. Searches for ticket by `ticket_id` property
2. Gets the internal UUID of the ticket
3. Deletes the ticket using UUID
4. Returns confirmation with deleted ticket details

---

## 🎓 Integration Examples

### Delete Multiple Tickets

```python
import requests

tickets_to_delete = ["TKT-001", "TKT-002", "TKT-003"]

for ticket_id in tickets_to_delete:
    try:
        response = requests.delete(f"http://localhost:8000/api/v1/tickets/{ticket_id}")
        if response.status_code == 200:
            print(f"✅ Deleted: {ticket_id}")
        else:
            print(f"❌ Failed to delete {ticket_id}: {response.json()['detail']}")
    except Exception as e:
        print(f"❌ Error deleting {ticket_id}: {e}")
```

### Delete with Confirmation

```python
import requests

def delete_ticket_with_confirmation(ticket_id):
    # First, get ticket details
    response = requests.get(f"http://localhost:8000/api/v1/tickets/{ticket_id}")
    
    if response.status_code == 404:
        print(f"❌ Ticket {ticket_id} not found")
        return False
    
    ticket = response.json()["ticket"]
    print(f"\nTicket to delete:")
    print(f"  ID: {ticket['ticket_id']}")
    print(f"  Title: {ticket['title']}")
    print(f"  Status: {ticket['status']}")
    
    confirm = input("\nDelete this ticket? (yes/no): ")
    
    if confirm.lower() == 'yes':
        response = requests.delete(f"http://localhost:8000/api/v1/tickets/{ticket_id}")
        print(f"✅ {response.json()['message']}")
        return True
    else:
        print("❌ Deletion cancelled")
        return False

# Usage
delete_ticket_with_confirmation("TKT-2024-001")
```

---

## 📋 All Ticket Endpoints

Now you have all CRUD operations:

| Method | Endpoint | Description |
|--------|----------|-------------|
| **POST** | `/api/v1/tickets` | Create single ticket |
| **POST** | `/api/v1/tickets/batch` | Create multiple tickets |
| **GET** | `/api/v1/tickets` | Get all tickets |
| **GET** | `/api/v1/tickets/{ticket_id}` | Get specific ticket |
| **GET** | `/api/v1/tickets/search?query=...` | Search tickets |
| **DELETE** | `/api/v1/tickets/{ticket_id}` | ✨ Delete ticket |

---

**Ready to delete tickets! 🗑️✨**
