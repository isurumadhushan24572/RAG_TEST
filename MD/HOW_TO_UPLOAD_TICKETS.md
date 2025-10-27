# 📤 How to Upload Tickets to Weaviate

This guide shows you **3 different ways** to upload tickets to your Weaviate vector database.

---

## 🎯 Prerequisites

1. **Docker is running** (Weaviate container)
   ```powershell
   docker-compose up -d
   ```

2. **FastAPI is running**
   ```powershell
   python app_fastapi.py
   ```

3. **Packages installed**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 📋 Ticket Format

Every ticket must have these **12 fields**:

```json
{
  "ticket_id": "TICK-2024-001",
  "title": "Short problem description",
  "description": "Detailed problem description",
  "category": "Database/Network/Application/etc",
  "status": "Open/Resolved/In Progress",
  "severity": "Low/Medium/High/Critical",
  "application": "Application name",
  "affected_users": "User count or scope",
  "environment": "Production/Staging/Development",
  "solution": "How it was resolved",
  "reasoning": "Root cause analysis",
  "timestamp": "2024-10-27T10:30:00Z"
}
```

---

## 🚀 Method 1: Using Python Script (Recommended)

### Upload a Single Ticket

```powershell
python upload_single_ticket_example.py
```

This script:
- ✅ Checks API health
- ✅ Uploads one ticket with embedding
- ✅ Shows detailed results

### Upload Multiple Tickets from JSON

```powershell
python upload_tickets_local.py
```

This script:
- ✅ Loads `tickets_sample.json`
- ✅ Generates embeddings for all tickets
- ✅ Batch uploads with progress
- ✅ Shows success/failure count

---

## 🌐 Method 2: Using Swagger UI (Interactive)

1. **Open Swagger UI**
   - Go to: http://localhost:8000/docs

2. **Find the POST /api/v1/tickets endpoint**

3. **Click "Try it out"**

4. **Paste your ticket JSON**
   ```json
   {
     "ticket_id": "TICK-2024-999",
     "title": "Test ticket from Swagger",
     "description": "Testing the upload via Swagger UI",
     "category": "Testing",
     "status": "Open",
     "severity": "Low",
     "application": "Test App",
     "affected_users": "1",
     "environment": "Development",
     "solution": "Test solution",
     "reasoning": "Test reasoning",
     "timestamp": "2024-10-27T12:00:00Z"
   }
   ```

5. **Click "Execute"**

6. **Check the response** - should show `"success": true`

---

## 🔧 Method 3: Using PowerShell/curl

### Using Invoke-WebRequest (PowerShell)

```powershell
$ticket = @{
    ticket_id = "TICK-2024-TEST"
    title = "PowerShell Upload Test"
    description = "Testing upload via PowerShell"
    category = "Testing"
    status = "Open"
    severity = "Low"
    application = "Test"
    affected_users = "1"
    environment = "Development"
    solution = "Test solution"
    reasoning = "Test"
    timestamp = "2024-10-27T12:00:00Z"
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/v1/tickets" `
    -Method POST `
    -ContentType "application/json" `
    -Body $ticket
```

### Using curl

```bash
curl -X POST "http://localhost:8000/api/v1/tickets" \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TICK-2024-TEST",
    "title": "Curl Upload Test",
    "description": "Testing upload via curl",
    "category": "Testing",
    "status": "Open",
    "severity": "Low",
    "application": "Test",
    "affected_users": "1",
    "environment": "Development",
    "solution": "Test solution",
    "reasoning": "Test",
    "timestamp": "2024-10-27T12:00:00Z"
  }'
```

---

## 🎯 Quick Test Flow

### 1. Start Everything
```powershell
# Terminal 1: Start Docker
docker-compose up -d

# Terminal 2: Start FastAPI (wait for embedding model to load)
python app_fastapi.py
```

### 2. Upload a Test Ticket
```powershell
# Terminal 3: Upload
python upload_single_ticket_example.py
```

### 3. Verify Upload
```powershell
# Check ticket count
curl http://localhost:8000/api/v1/collections/SupportTickets/count

# View all tickets
curl http://localhost:8000/api/v1/tickets

# Search for similar tickets
curl "http://localhost:8000/api/v1/tickets/search?query=database+problem&limit=3"
```

---

## 📊 Expected Output (Single Upload)

```
🎫 UPLOADING SINGLE TICKET
============================================================

📝 Ticket ID: TICK-2024-001
📋 Title: Database Connection Timeout
⚠️ Severity: High

📤 Sending ticket to API...
✅ SUCCESS! Ticket uploaded with embedding

📊 Response:
{
  "success": true,
  "message": "Ticket TICK-2024-001 uploaded successfully with embedding",
  "ticket_id": "TICK-2024-001",
  "data": {
    "uuid": "abc123-def456-...",
    "ticket": { ... }
  }
}

============================================================
✅ UPLOAD COMPLETE!
============================================================
```

---

## ❌ Common Issues

### 1. "Cannot connect to API"
**Solution:** Start FastAPI first
```powershell
python app_fastapi.py
```

### 2. "Weaviate is not connected"
**Solution:** Start Docker container
```powershell
docker-compose up -d
```

### 3. "Embedding model not initialized"
**Solution:** Wait for model to load (first time ~80MB download)
```
📦 Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...
✅ Embedding model loaded successfully
```

### 4. Missing fields error
**Solution:** Make sure your ticket has all 12 required fields

---

## 🔍 How It Works (Behind the Scenes)

When you upload a ticket:

1. **API receives ticket** → FastAPI validates the 12 fields
2. **Generate embedding** → sentence-transformers creates a 384-dimensional vector from title + description + solution
3. **Store in Weaviate** → Ticket data + embedding vector stored together
4. **Return success** → You get UUID and confirmation

Now you can:
- **Search semantically** → Find similar tickets by meaning, not keywords
- **Get recommendations** → "Show me tickets like this one"
- **Build RAG systems** → Use ticket knowledge for AI responses

---

## 📚 Next Steps

- **Batch upload**: Use `upload_tickets_local.py` for multiple tickets
- **Search**: Try semantic search at `/api/v1/tickets/search`
- **Explore**: Check Swagger UI at http://localhost:8000/docs
- **Build**: Use this as foundation for RAG chatbot!

---

## 🎓 Example Tickets in JSON Format

Create a file `my_tickets.json`:

```json
[
  {
    "ticket_id": "TICK-001",
    "title": "Login page not loading",
    "description": "Users cannot access the login page, getting 404 error",
    "category": "Web Application",
    "status": "Resolved",
    "severity": "Critical",
    "application": "User Portal",
    "affected_users": "All users",
    "environment": "Production",
    "solution": "Fixed routing configuration in nginx",
    "reasoning": "Incorrect nginx configuration after deployment",
    "timestamp": "2024-10-27T09:00:00Z"
  },
  {
    "ticket_id": "TICK-002",
    "title": "Slow report generation",
    "description": "Monthly reports taking over 10 minutes to generate",
    "category": "Performance",
    "status": "In Progress",
    "severity": "Medium",
    "application": "Analytics Dashboard",
    "affected_users": "Finance team (5 users)",
    "environment": "Production",
    "solution": "Added database indexes and optimized queries",
    "reasoning": "Missing indexes on large tables causing full table scans",
    "timestamp": "2024-10-27T11:30:00Z"
  }
]
```

Then upload with:
```powershell
python upload_tickets_local.py
```

---

**Happy uploading! 🚀**
