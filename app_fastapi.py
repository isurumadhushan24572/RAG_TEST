"""
FastAPI Application with Weaviate Vector Database Integration
This application provides REST API endpoints to interact with Weaviate vector database.
"""

# ===================== IMPORTS =====================
# Import required libraries for FastAPI and Weaviate integration
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pydantic import BaseModel
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ===================== ENVIRONMENT SETUP =====================
# Load environment variables from .env file
load_dotenv()

# ===================== WEAVIATE CLIENT CONFIGURATION =====================
# Global variable to store Weaviate client instance
weaviate_client = None

# Global embedding model for local vectorization
embedding_model = None

# Collection name for storing tickets
TICKETS_COLLECTION_NAME = "SupportTickets"

# ===================== PYDANTIC MODELS =====================
class TicketModel(BaseModel):
    """Pydantic model for ticket data validation"""
    ticket_id: str
    title: str
    description: str
    category: str
    status: str
    severity: str
    application: str
    affected_users: str
    environment: str
    solution: str
    reasoning: str
    timestamp: str

class TicketResponse(BaseModel):
    """Response model for ticket operations"""
    success: bool
    message: str
    ticket_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

# ===================== LIFESPAN CONTEXT MANAGER =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events using modern async context manager.
    """
    global weaviate_client, embedding_model
    
    # STARTUP: Initialize Weaviate connection and embedding model
    try:
        # Load local embedding model (downloads on first use, ~80MB)
        print("📦 Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Connect to local Weaviate instance running on Docker
        weaviate_client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),  # Docker container host (default: localhost)
            port=int(os.getenv("WEAVIATE_PORT", "8080")),  # Weaviate default port (default: 8080)
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),  # gRPC port for v4 client (default: 50051)
        )
        
        # Verify connection is ready
        if weaviate_client.is_ready():
            print("✅ Successfully connected to Weaviate vector database")
            print(f"📊 Weaviate is ready: {weaviate_client.is_ready()}")
            
            # ========== SMART COLLECTION HANDLING ==========
            # Check if SupportTickets collection already exists
            # This prevents errors when restarting the application
            # Collections persist in Weaviate even after app shutdown
            try:
                # Check if collection exists in Weaviate
                collection_exists = weaviate_client.collections.exists(TICKETS_COLLECTION_NAME)
                
                if collection_exists:
                    # Collection already exists - skip creation and reuse existing data
                    print(f"✅ Collection '{TICKETS_COLLECTION_NAME}' already exists (using existing collection)")
                    
                    # Get collection info to verify schema and show current data
                    try:
                        collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
                        count_result = collection.aggregate.over_all(total_count=True)
                        ticket_count = count_result.total_count
                        print(f"📊 Existing collection has {ticket_count} ticket(s)")
                    except Exception as count_error:
                        print(f"⚠️ Could not get ticket count: {str(count_error)}")
                    
                else:
                    # Collection doesn't exist - create new one
                    print(f"📝 Collection '{TICKETS_COLLECTION_NAME}' not found - creating new collection...")
                    
                    # Create collection with proper schema for tickets
                    # Using vectorizer_config=None for manual/local embeddings
                    weaviate_client.collections.create(
                        name=TICKETS_COLLECTION_NAME,
                        description="Support ticket incidents with AI-generated solutions",
                        vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # No automatic vectorization
                        properties=[
                            Property(name="ticket_id", data_type=DataType.TEXT, description="Unique ticket identifier"),
                            Property(name="title", data_type=DataType.TEXT, description="Ticket title/summary"),
                            Property(name="description", data_type=DataType.TEXT, description="Detailed problem description"),
                            Property(name="category", data_type=DataType.TEXT, description="Issue category"),
                            Property(name="status", data_type=DataType.TEXT, description="Ticket status (Open/Resolved)"),
                            Property(name="severity", data_type=DataType.TEXT, description="Severity level"),
                            Property(name="application", data_type=DataType.TEXT, description="Affected application/service"),
                            Property(name="affected_users", data_type=DataType.TEXT, description="Impact scope"),
                            Property(name="environment", data_type=DataType.TEXT, description="Environment (Production/Staging/etc)"),
                            Property(name="solution", data_type=DataType.TEXT, description="Resolution steps"),
                            Property(name="reasoning", data_type=DataType.TEXT, description="Root cause analysis"),
                            Property(name="timestamp", data_type=DataType.TEXT, description="Ticket creation timestamp"),
                        ]
                    )
                    print(f"✅ Collection '{TICKETS_COLLECTION_NAME}' created successfully with empty data")
                    print(f"ℹ️  Using local embeddings (sentence-transformers) for vectorization")
                    
            except Exception as e:
                print(f"⚠️ Error handling collection: {str(e)}")
                # Don't fail startup - collection might still be usable
                
        else:
            print("⚠️ Weaviate client connected but not ready")
            
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {str(e)}")
        print("⚠️ Make sure Weaviate Docker container is running")
        print("💡 Run: docker-compose up -d")
        weaviate_client = None
    
    # Yield control to the application
    yield
    
    # SHUTDOWN: Close Weaviate connection
    if weaviate_client is not None:
        try:
            weaviate_client.close()
            print("✅ Weaviate connection closed successfully")
            print("💾 Collection data is persisted in Weaviate (will be available on next startup)")
        except Exception as e:
            print(f"⚠️ Error closing Weaviate connection: {str(e)}")

# ===================== FASTAPI APP INITIALIZATION =====================
# Create FastAPI application instance with lifespan handler
app = FastAPI(
    title="Weaviate Vector DB API",
    description="REST API for Weaviate Vector Database Operations",
    version="1.0.0",
    lifespan=lifespan  # Use modern lifespan context manager
)

# ===================== HELPER FUNCTIONS =====================
def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for text using local sentence-transformers model.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    global embedding_model
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding model not initialized"
        )
    return embedding_model.encode(text).tolist()

# ===================== ROOT ENDPOINT =====================
@app.get("/", tags=["Health Check"])
async def root():
    """
    Root endpoint to check if API is running.
    
    Returns:
        Dict: Welcome message and API status
    """
    return {
        "message": "Weaviate Vector DB API is running",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "document_count": "/api/v1/collections/{collection_name}/count",
            "collections": "/api/v1/collections"
        }
    }

# ===================== HEALTH CHECK ENDPOINT =====================
@app.get("/health", tags=["Health Check"])
async def health_check():
    """
    Health check endpoint to verify Weaviate connection status.
    
    Returns:
        Dict: Health status of API and Weaviate database
    
    Raises:
        HTTPException: If Weaviate connection fails
    """
    global weaviate_client
    
    try:
        # Check if Weaviate client exists
        if weaviate_client is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "weaviate_status": "not_connected",
                    "weaviate_ready": False,
                    "message": "Weaviate client not initialized"
                }
            )
        
        # Check if Weaviate is ready
        is_ready = weaviate_client.is_ready()
        
        if is_ready:
            return {
                "status": "healthy",
                "weaviate_status": "connected",
                "weaviate_ready": True,
                "message": "API and Weaviate are running successfully"
            }
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "weaviate_status": "not_ready",
                    "weaviate_ready": False,
                    "message": "Weaviate is not ready"
                }
            )
    except Exception as e:
        # Return error if connection fails
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Weaviate connection failed: {str(e)}"
        )

# ===================== GET DOCUMENT COUNT ENDPOINT =====================
@app.get("/api/v1/collections/{collection_name}/count", tags=["Collections"])
async def get_document_count(collection_name: str):
    """
    Get the total number of documents in a specific Weaviate collection.
    
    Args:
        collection_name (str): Name of the collection to query
    
    Returns:
        Dict: Collection name and document count
    
    Raises:
        HTTPException: If collection doesn't exist or query fails
    """
    global weaviate_client
    
    try:
        # Check if Weaviate client is initialized
        if weaviate_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Weaviate client not initialized. Check if Weaviate is running."
            )
        
        # Check if the collection exists in Weaviate schema
        if not weaviate_client.collections.exists(collection_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' does not exist in Weaviate"
            )
        
        # Get the collection reference
        collection = weaviate_client.collections.get(collection_name)
        
        # Perform aggregation query to count total objects in collection
        # This uses Weaviate's aggregate API to get total count
        result = collection.aggregate.over_all(total_count=True)
        
        # Extract total count from result
        total_count = result.total_count
        
        # Return the count in JSON format
        return {
            "collection_name": collection_name,
            "document_count": total_count,
            "status": "success",
            "message": f"Successfully retrieved document count from collection '{collection_name}'"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 404 for collection not found)
        raise
    except Exception as e:
        # Handle any other errors during query
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving document count: {str(e)}"
        )

# ===================== LIST ALL COLLECTIONS ENDPOINT =====================
@app.get("/api/v1/collections", tags=["Collections"])
async def list_collections():
    """
    List all available collections in Weaviate vector database.
    
    Returns:
        Dict: List of all collection names with their counts
    
    Raises:
        HTTPException: If query fails
    """
    global weaviate_client
    
    try:
        # Check if Weaviate client is initialized
        if weaviate_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Weaviate client not initialized. Check if Weaviate is running."
            )
        
        # Get all collections from Weaviate schema
        collections = weaviate_client.collections.list_all()
        
        # Create list to store collection information
        collection_list = []
        
        # Iterate through each collection and get its count
        for collection_name in collections:
            try:
                # Get collection reference
                collection = weaviate_client.collections.get(collection_name)
                
                # Get document count for this collection
                result = collection.aggregate.over_all(total_count=True)
                count = result.total_count
                
                # Add to list
                collection_list.append({
                    "name": collection_name,
                    "document_count": count
                })
            except Exception as e:
                # If count fails for a collection, add it with error
                collection_list.append({
                    "name": collection_name,
                    "document_count": 0,
                    "error": str(e)
                })
        
        # Return all collections with their counts
        return {
            "total_collections": len(collection_list),
            "collections": collection_list,
            "status": "success"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any errors during collection listing
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )

# ===================== TICKET UPLOAD ENDPOINT =====================
@app.post("/api/v1/tickets", tags=["Tickets"], response_model=TicketResponse)
async def upload_ticket(ticket: TicketModel):
    """
    Upload a single ticket to Weaviate SupportTickets collection with local embeddings.
    
    Args:
        ticket: TicketModel with all required fields
        
    Returns:
        TicketResponse: Success status and ticket details
        
    Raises:
        HTTPException: If Weaviate is not connected or upload fails
    """
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Get the SupportTickets collection
        tickets_collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
        
        # Prepare ticket data for Weaviate
        ticket_data = {
            "ticket_id": ticket.ticket_id,
            "title": ticket.title,
            "description": ticket.description,
            "category": ticket.category,
            "status": ticket.status,
            "severity": ticket.severity,
            "application": ticket.application,
            "affected_users": ticket.affected_users,
            "environment": ticket.environment,
            "solution": ticket.solution,
            "reasoning": ticket.reasoning,
            "timestamp": ticket.timestamp
        }
        
        # Generate embedding from ticket content (title + description + solution)
        text_to_embed = f"{ticket.title} {ticket.description} {ticket.solution}"
        embedding = generate_embedding(text_to_embed)
        
        # Insert ticket into Weaviate with local embedding
        uuid = tickets_collection.data.insert(
            properties=ticket_data,
            vector=embedding  # Provide embedding manually
        )
        
        return TicketResponse(
            success=True,
            message=f"Ticket {ticket.ticket_id} uploaded successfully with embedding",
            ticket_id=ticket.ticket_id,
            data={"uuid": str(uuid), "ticket": ticket_data}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading ticket: {str(e)}"
        )

# ===================== BATCH TICKET UPLOAD ENDPOINT =====================
@app.post("/api/v1/tickets/batch", tags=["Tickets"])
async def upload_tickets_batch(tickets: List[TicketModel]):
    """
    Upload multiple tickets to Weaviate SupportTickets collection in batch with local embeddings.
    
    Args:
        tickets: List of TicketModel objects
        
    Returns:
        Dict: Success status, count, and details
        
    Raises:
        HTTPException: If Weaviate is not connected or batch upload fails
    """
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Get the SupportTickets collection
        tickets_collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
        
        # Prepare batch data
        uploaded_count = 0
        failed_tickets = []
        
        # Use Weaviate batch insert for efficiency
        with tickets_collection.batch.dynamic() as batch:
            for ticket in tickets:
                try:
                    ticket_data = {
                        "ticket_id": ticket.ticket_id,
                        "title": ticket.title,
                        "description": ticket.description,
                        "category": ticket.category,
                        "status": ticket.status,
                        "severity": ticket.severity,
                        "application": ticket.application,
                        "affected_users": ticket.affected_users,
                        "environment": ticket.environment,
                        "solution": ticket.solution,
                        "reasoning": ticket.reasoning,
                        "timestamp": ticket.timestamp
                    }
                    
                    # Generate embedding from ticket content
                    text_to_embed = f"{ticket.title} {ticket.description} {ticket.solution}"
                    embedding = generate_embedding(text_to_embed)
                    
                    # Add to batch with embedding
                    batch.add_object(
                        properties=ticket_data,
                        vector=embedding
                    )
                    uploaded_count += 1
                    
                except Exception as e:
                    failed_tickets.append({
                        "ticket_id": ticket.ticket_id,
                        "error": str(e)
                    })
        
        return {
            "success": True,
            "message": f"Batch upload completed with local embeddings",
            "total_tickets": len(tickets),
            "uploaded": uploaded_count,
            "failed": len(failed_tickets),
            "failed_tickets": failed_tickets if failed_tickets else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch upload: {str(e)}"
        )

# ===================== GET ALL TICKETS ENDPOINT =====================
@app.get("/api/v1/tickets", tags=["Tickets"])
async def get_all_tickets(limit: int = 100, offset: int = 0):
    """
    Retrieve all tickets from Weaviate SupportTickets collection.
    
    Args:
        limit: Maximum number of tickets to return (default 100)
        offset: Number of tickets to skip (default 0)
        
    Returns:
        Dict: List of tickets and metadata
        
    Raises:
        HTTPException: If Weaviate is not connected or query fails
    """
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Get the SupportTickets collection
        tickets_collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
        
        # Query all tickets with limit and offset
        response = tickets_collection.query.fetch_objects(
            limit=limit,
            offset=offset
        )
        
        # Get total count
        total_result = tickets_collection.aggregate.over_all(total_count=True)
        total_count = total_result.total_count
        
        # Extract ticket data
        tickets_list = []
        for obj in response.objects:
            ticket_data = obj.properties
            ticket_data["uuid"] = str(obj.uuid)
            tickets_list.append(ticket_data)
        
        return {
            "success": True,
            "total_count": total_count,
            "returned_count": len(tickets_list),
            "limit": limit,
            "offset": offset,
            "tickets": tickets_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving tickets: {str(e)}"
        )

# ===================== GET TICKET BY ID ENDPOINT =====================
@app.get("/api/v1/tickets/{ticket_id}", tags=["Tickets"])
async def get_ticket_by_id(ticket_id: str):
    """
    Retrieve a specific ticket by its ticket_id.
    
    Args:
        ticket_id: Unique ticket identifier (e.g., TKT-0001)
        
    Returns:
        Dict: Ticket data
        
    Raises:
        HTTPException: If ticket not found or query fails
    """
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Get the SupportTickets collection
        tickets_collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
        
        # Query for specific ticket_id
        response = tickets_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("ticket_id").equal(ticket_id),
            limit=1
        )
        
        if len(response.objects) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket {ticket_id} not found"
            )
        
        # Extract ticket data
        obj = response.objects[0]
        ticket_data = obj.properties
        ticket_data["uuid"] = str(obj.uuid)
        
        return {
            "success": True,
            "ticket": ticket_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ticket: {str(e)}"
        )

# ===================== DELETE TICKET BY ID ENDPOINT =====================
@app.delete("/api/v1/tickets/{ticket_id}", tags=["Tickets"])
async def delete_ticket_by_id(ticket_id: str):
    """
    Delete a specific ticket from Weaviate by its ticket_id.
    
    Args:
        ticket_id: Unique ticket identifier (e.g., TKT-0001)
        
    Returns:
        Dict: Success status and deletion details
        
    Raises:
        HTTPException: If ticket not found or deletion fails
    """
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Get the SupportTickets collection
        tickets_collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
        
        # First, find the ticket to get its UUID
        response = tickets_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("ticket_id").equal(ticket_id),
            limit=1
        )
        
        # Check if ticket exists
        if len(response.objects) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket '{ticket_id}' not found in database"
            )
        
        # Get the UUID of the ticket to delete
        ticket_uuid = response.objects[0].uuid
        ticket_data = response.objects[0].properties
        
        # Delete the ticket by UUID
        tickets_collection.data.delete_by_id(ticket_uuid)
        
        return {
            "success": True,
            "message": f"Ticket '{ticket_id}' deleted successfully",
            "deleted_ticket": {
                "ticket_id": ticket_id,
                "uuid": str(ticket_uuid),
                "title": ticket_data.get("title", "N/A")
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting ticket: {str(e)}"
        )

# ===================== SEARCH TICKETS ENDPOINT =====================
@app.get("/api/v1/tickets/search", tags=["Tickets"])
async def search_tickets(query: str, limit: int = 5):
    """
    Search for similar tickets using vector similarity search with local embeddings.
    
    Args:
        query: Search query (natural language description)
        limit: Maximum number of results (default 5)
        
    Returns:
        Dict: List of similar tickets with similarity scores
        
    Raises:
        HTTPException: If search fails
    """
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Get the SupportTickets collection
        tickets_collection = weaviate_client.collections.get(TICKETS_COLLECTION_NAME)
        
        # Generate embedding from search query using local model
        query_embedding = generate_embedding(query)
        
        # Perform vector similarity search using generated embedding
        response = tickets_collection.query.near_vector(
            near_vector=query_embedding,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(distance=True, certainty=True)
        )
        
        # Extract results
        results = []
        for obj in response.objects:
            result_data = obj.properties
            result_data["uuid"] = str(obj.uuid)
            result_data["distance"] = obj.metadata.distance
            result_data["certainty"] = obj.metadata.certainty
            result_data["similarity_score"] = obj.metadata.certainty  # 0-1 score
            results.append(result_data)
        
        return {
            "success": True,
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching tickets: {str(e)}"
        )

# ===================== DELETE COLLECTION ENDPOINT =====================
@app.delete("/api/v1/collections/{collection_name}", tags=["Collections"])
async def delete_collection(collection_name: str):
    """
    Delete a collection from Weaviate.
    ⚠️ WARNING: This permanently deletes all data in the collection!
    
    Args:
        collection_name: Name of collection to delete
        
    Returns:
        Dict: Success status and deleted document count
        
    Raises:
        HTTPException: If collection doesn't exist or deletion fails
    """
    global weaviate_client
    
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(collection_name):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' does not exist"
            )
        
        # Get count before deletion (for reporting)
        try:
            collection = weaviate_client.collections.get(collection_name)
            count_result = collection.aggregate.over_all(total_count=True)
            deleted_count = count_result.total_count
        except Exception:
            deleted_count = 0  # If we can't get count, set to 0
        
        # Delete the collection
        weaviate_client.collections.delete(collection_name)
        
        return {
            "success": True,
            "message": f"Collection '{collection_name}' deleted successfully",
            "deleted_documents": deleted_count,
            "note": "Restart the application to recreate an empty collection"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection: {str(e)}"
        )

# ===================== GET DEFAULT COLLECTION COUNT ENDPOINT =====================
@app.get("/api/v1/count", tags=["Collections"])
async def get_default_collection_count():
    """
    Get document count from the default 'Documents' collection.
    This is a convenience endpoint that doesn't require specifying collection name.
    
    Returns:
        Dict: Default collection name and document count
    
    Raises:
        HTTPException: If default collection doesn't exist or query fails
    """
    # Call the main count endpoint with SupportTickets collection
    return await get_document_count(TICKETS_COLLECTION_NAME)

# ===================== APPLICATION ENTRY POINT =====================
if __name__ == "__main__":
    """
    Entry point for running the FastAPI application.
    This runs when executing: python app_fastapi.py
    """
    import uvicorn
    
    # Get port from environment or use default
    api_port = int(os.getenv("API_PORT", "8000"))  # Default to 8000 if not set
    
    # Run the FastAPI application with Uvicorn server
    # Use import string format for reload to work properly
    uvicorn.run(
        "app_fastapi:app",      # Import string format (required for reload)
        host="0.0.0.0",         # Listen on all interfaces
        port=api_port,          # Application port (Weaviate runs on 8080)
        log_level="info",       # Set logging level
        reload=True             # Enable auto-reload for development
    )

