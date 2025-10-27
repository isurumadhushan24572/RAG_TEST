# 🚀 Quick Start Guide - Weaviate + FastAPI

## ⚡ Fast Setup (3 Steps)

### Step 1: Start Weaviate Database
```powershell
docker-compose up -d
```
Wait 10 seconds for Weaviate to initialize.

### Step 2: Install Dependencies
```powershell
pip install -r requirements_fastapi.txt
```

### Step 3: Run FastAPI Application
```powershell
python app_fastapi.py
```

---

## ✅ Verify Everything Works

### Check Weaviate is Running:
```powershell
# In browser or PowerShell
Invoke-RestMethod -Uri "http://localhost:8080/v1/.well-known/ready" -Method Get
```
Should return: `{"status":"ok"}`

### Check FastAPI is Running:
```powershell
# In browser or PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```
Should return: `{"status":"healthy","weaviate_status":"connected",...}`

---

## 📡 Test the Main Endpoint

### Get Document Count:
```powershell
# Replace "Documents" with your collection name
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/collections/Documents/count" -Method Get
```

### List All Collections:
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/collections" -Method Get
```

---

## 🌐 Interactive API Documentation

Open in browser:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🛑 Stop Everything

```powershell
# Stop FastAPI: Press CTRL+C in terminal

# Stop Weaviate:
docker-compose down
```

---

## 📖 Full Documentation

See `README_FASTAPI.md` for complete setup guide and troubleshooting.

---

**Services Running:**
- 🐳 Weaviate Vector DB: http://localhost:8080
- 🚀 FastAPI Application: http://localhost:8000
