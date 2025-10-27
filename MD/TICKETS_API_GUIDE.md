# 🎫 Support Tickets API - Complete Guide

## 📋 Overview

The Support Tickets API allows you to upload, retrieve, and search support tickets in Weaviate vector database with automatic AI vectorization for similarity search.

---

## 🚀 Quick Start

### 1. Start Weaviate & FastAPI

```powershell
# Start Weaviate Docker container
docker-compose up -d

# Start FastAPI server
python app_fastapi.py
```

**API will be available at:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`

---

## 📊 Collection Schema

**Collection Name:** `SupportTickets`

**Auto-created on startup** with the following schema:

| Field | Type | Description |
|-------|------|-------------|
| `ticket_id` | TEXT | Unique identifier (e.g., TKT-0001) |
| `title` | TEXT | Brief issue summary |
| `description` | TEXT | Detailed problem description |
| `category` | TEXT | Issue category (API Issues, Database, etc.) |
| `status` | TEXT | Ticket status (Open, Resolved) |
| `severity` | TEXT | Severity level (Critical, High, Medium, Low) |
| `application` | TEXT | Affected application/service name |
| `affected_users` | TEXT | Impact scope description |
| `environment` | TEXT | Environment (Production, Staging, etc.) |
| `solution` | TEXT | Resolution steps/fix |
| `reasoning` | TEXT | Root cause analysis |
| `timestamp` | TEXT | Creation timestamp |

**Vectorization:** Automatic using `text2vec-transformers` on all text fields

---

## 🔌 API Endpoints

### 1. **Upload Single Ticket**

**POST** `/api/v1/tickets`

Upload a single support ticket to Weaviate.

#### Request Body (JSON):
```json
{
  "ticket_id": "TKT-0001",
  "title": "Payment API Returning 500 Internal Server Error",
  "description": "Payment processing API is throwing 500 errors for all transactions. Error started approximately 30 minutes ago.",
  "category": "API Issues",
  "status": "Resolved",
  "severity": "Critical",
  "application": "Payment API Service",
  "affected_users": "All users attempting payments",
  "environment": "Production",
  "solution": "1. Checked application logs\n2. Verified database connectivity\n3. Increased connection pool settings\n4. Scaled API pods horizontally",
  "reasoning": "Database connection pool exhaustion during high traffic caused API failures.",
  "timestamp": "2025-10-19 14:30:00"
}
```

#### PowerShell Example:
```powershell
$ticket = @{
    ticket_id = "TKT-0001"
    title = "Payment API Returning 500 Internal Server Error"
    description = "Payment processing API is throwing 500 errors for all transactions."
    category = "API Issues"
    status = "Resolved"
    severity = "Critical"
    application = "Payment API Service"
    affected_users = "All users attempting payments"
    environment = "Production"
    solution = "1. Checked logs\n2. Fixed connection pool"
    reasoning = "Database connection pool exhaustion"
    timestamp = "2025-10-19 14:30:00"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets" `
    -Method Post `
    -ContentType "application/json" `
    -Body $ticket
```

#### Response:
```json
{
  "success": true,
  "message": "Ticket TKT-0001 uploaded successfully",
  "ticket_id": "TKT-0001",
  "data": {
    "uuid": "12345678-1234-1234-1234-123456789abc",
    "ticket": { ... }
  }
}
```

---

### 2. **Batch Upload Tickets**

**POST** `/api/v1/tickets/batch`

Upload multiple tickets at once from `tickets_sample.json`.

#### Request Body (JSON Array):
```json
[
  {
    "ticket_id": "TKT-0001",
    "title": "Payment API Error",
    ...
  },
  {
    "ticket_id": "TKT-0002",
    "title": "Memory Leak",
    ...
  }
]
```

#### PowerShell Example:
```powershell
# Load tickets from file and upload
$tickets = Get-Content "tickets_sample.json" | ConvertFrom-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/batch" `
    -Method Post `
    -ContentType "application/json" `
    -Body ($tickets | ConvertTo-Json -Depth 10)
```

#### Python Example:
```python
import requests
import json

# Load tickets from file
with open('tickets_sample.json', 'r') as f:
    tickets = json.load(f)

# Upload batch
response = requests.post(
    'http://localhost:8000/api/v1/tickets/batch',
    json=tickets
)

print(response.json())
```

#### Response:
```json
{
  "success": true,
  "message": "Batch upload completed",
  "total_tickets": 19,
  "uploaded": 19,
  "failed": 0,
  "failed_tickets": null
}
```

---

### 3. **Get All Tickets**

**GET** `/api/v1/tickets?limit=100&offset=0`

Retrieve all tickets from Weaviate.

#### Parameters:
- `limit` (optional): Max tickets to return (default: 100)
- `offset` (optional): Number to skip for pagination (default: 0)

#### PowerShell Example:
```powershell
# Get first 10 tickets
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets?limit=10&offset=0"

# Get next 10 tickets
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets?limit=10&offset=10"
```

#### Response:
```json
{
  "success": true,
  "total_count": 19,
  "returned_count": 10,
  "limit": 10,
  "offset": 0,
  "tickets": [
    {
      "ticket_id": "TKT-0001",
      "title": "Payment API Error",
      "uuid": "12345678-1234-1234-1234-123456789abc",
      ...
    }
  ]
}
```

---

### 4. **Get Ticket by ID**

**GET** `/api/v1/tickets/{ticket_id}`

Retrieve a specific ticket by its ticket_id.

#### PowerShell Example:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-0001"
```

#### Response:
```json
{
  "success": true,
  "ticket": {
    "ticket_id": "TKT-0001",
    "title": "Payment API Returning 500 Internal Server Error",
    "description": "Payment processing API is throwing 500 errors...",
    "uuid": "12345678-1234-1234-1234-123456789abc",
    ...
  }
}
```

---

### 5. **Search Similar Tickets (Vector Search)**

**GET** `/api/v1/tickets/search?query=api+error&limit=5`

Search for tickets using AI-powered vector similarity.

#### Parameters:
- `query` (required): Natural language search query
- `limit` (optional): Max results (default: 5)

#### PowerShell Example:
```powershell
# Search for API-related issues
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/search?query=payment+api+500+error&limit=3"

# Search for database problems
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/search?query=database+connection+timeout&limit=5"
```

#### Response:
```json
{
  "success": true,
  "query": "payment api 500 error",
  "results_count": 3,
  "results": [
    {
      "ticket_id": "TKT-0001",
      "title": "Payment API Returning 500 Internal Server Error",
      "similarity_score": 0.95,
      "certainty": 0.95,
      "distance": 0.05,
      ...
    },
    {
      "ticket_id": "TKT-0009",
      "title": "API Rate Limiting Issues",
      "similarity_score": 0.78,
      ...
    }
  ]
}
```

**Similarity Score:** 0-1 (higher = more similar)

---

### 6. **Get Collection Count**

**GET** `/api/v1/collections/SupportTickets/count`

Get total number of tickets in collection.

#### PowerShell Example:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/collections/SupportTickets/count"
```

#### Response:
```json
{
  "collection_name": "SupportTickets",
  "document_count": 19,
  "status": "success"
}
```

---

### 7. **List All Collections**

**GET** `/api/v1/collections`

List all Weaviate collections with document counts.

#### PowerShell Example:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/collections"
```

#### Response:
```json
{
  "total_collections": 1,
  "collections": [
    {
      "name": "SupportTickets",
      "document_count": 19
    }
  ],
  "status": "success"
}
```

---

## 📝 Complete Workflow Example

### Step 1: Upload Sample Tickets

```powershell
# Load and upload all sample tickets
$tickets = Get-Content "tickets_sample.json" | ConvertFrom-Json

$response = Invoke-RestMethod `
    -Uri "http://localhost:8000/api/v1/tickets/batch" `
    -Method Post `
    -ContentType "application/json" `
    -Body ($tickets | ConvertTo-Json -Depth 10)

Write-Host "✅ Uploaded: $($response.uploaded) tickets"
```

### Step 2: Verify Upload

```powershell
# Check total count
$count = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/collections/SupportTickets/count"
Write-Host "📊 Total tickets: $($count.document_count)"
```

### Step 3: Search for Similar Issues

```powershell
# Search for API errors
$results = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/search?query=api+timeout+error&limit=3"

foreach ($ticket in $results.results) {
    Write-Host "🎫 $($ticket.ticket_id): $($ticket.title) (Score: $($ticket.similarity_score))"
}
```

### Step 4: Get Specific Ticket

```powershell
# Retrieve ticket details
$ticket = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/TKT-0001"

Write-Host "📋 Ticket: $($ticket.ticket.title)"
Write-Host "✅ Solution: $($ticket.ticket.solution)"
```

---

## 🐍 Python Integration Example

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# 1. Upload single ticket
new_ticket = {
    "ticket_id": "TKT-9999",
    "title": "Test Ticket",
    "description": "This is a test ticket",
    "category": "Other",
    "status": "Open",
    "severity": "Low",
    "application": "Test App",
    "affected_users": "Test users",
    "environment": "Development",
    "solution": "Not resolved yet",
    "reasoning": "Test ticket for API",
    "timestamp": "2025-10-27 10:00:00"
}

response = requests.post(f"{BASE_URL}/api/v1/tickets", json=new_ticket)
print(f"✅ Upload: {response.json()['message']}")

# 2. Search for similar tickets
search_query = "payment api error 500"
response = requests.get(
    f"{BASE_URL}/api/v1/tickets/search",
    params={"query": search_query, "limit": 3}
)

results = response.json()
print(f"\n🔍 Found {results['results_count']} similar tickets:")
for ticket in results['results']:
    print(f"  - {ticket['ticket_id']}: {ticket['title']} (Score: {ticket['similarity_score']:.2f})")

# 3. Get all tickets
response = requests.get(f"{BASE_URL}/api/v1/tickets", params={"limit": 5})
all_tickets = response.json()
print(f"\n📊 Total tickets: {all_tickets['total_count']}")

# 4. Batch upload from file
with open('tickets_sample.json', 'r') as f:
    tickets = json.load(f)

response = requests.post(f"{BASE_URL}/api/v1/tickets/batch", json=tickets)
print(f"\n📦 Batch upload: {response.json()['uploaded']} tickets uploaded")
```

---

## 🔧 Troubleshooting

### Error: "Weaviate vector database is not connected"

**Solution:**
```powershell
# Check if Weaviate is running
docker ps

# If not running, start it
docker-compose up -d

# Wait 30 seconds, then restart FastAPI
python app_fastapi.py
```

### Error: "Collection SupportTickets not found"

**Solution:** Collection is auto-created on startup. Restart FastAPI:
```powershell
# Stop FastAPI (Ctrl+C), then restart
python app_fastapi.py
```

### Empty Search Results

**Cause:** No tickets uploaded yet

**Solution:**
```powershell
# Upload sample tickets first
$tickets = Get-Content "tickets_sample.json" | ConvertFrom-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/batch" `
    -Method Post -ContentType "application/json" -Body ($tickets | ConvertTo-Json -Depth 10)
```

---

## 📈 Performance Tips

1. **Batch Upload:** Use `/api/v1/tickets/batch` for multiple tickets (faster than individual uploads)
2. **Pagination:** Use `limit` and `offset` for large datasets
3. **Search Optimization:** Keep queries concise for better vector matching
4. **Index Size:** Monitor collection size with `/api/v1/collections/SupportTickets/count`

---

## 🎯 Use Cases

### 1. **Similar Incident Detection**
```powershell
# When new ticket arrives, find similar past incidents
$query = "database connection timeout production"
$similar = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/search?query=$query&limit=5"
```

### 2. **Knowledge Base Search**
```powershell
# Search historical solutions
$results = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/tickets/search?query=memory+leak+fix&limit=3"
foreach ($r in $results.results) {
    Write-Host "Solution: $($r.solution)"
}
```

### 3. **Ticket Analytics**
```python
# Get all tickets and analyze
response = requests.get("http://localhost:8000/api/v1/tickets", params={"limit": 1000})
tickets = response.json()['tickets']

# Count by severity
critical = len([t for t in tickets if t['severity'] == 'Critical'])
print(f"Critical tickets: {critical}")
```

---

## 📚 Related Files

- **FastAPI App:** `app_fastapi.py`
- **Sample Data:** `tickets_sample.json`
- **Docker Config:** `docker-compose.yml`
- **Main README:** `README_FASTAPI.md`

---

## 🚀 Next Steps

1. ✅ Upload sample tickets: `tickets_sample.json`
2. ✅ Test search endpoint with various queries
3. ✅ Integrate with your monitoring system
4. ✅ Build custom UI on top of API
5. ✅ Set up automated ticket ingestion

---

**API Status:** ✅ Ready  
**Swagger UI:** http://localhost:8000/docs  
**Collection:** Auto-created on startup  
**Vectorization:** Automatic AI embeddings  

**Happy Ticketing! 🎫**
