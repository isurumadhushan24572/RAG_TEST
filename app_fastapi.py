# ===================== IMPORTS =====================
# Import required libraries for FastAPI and Weaviate integration
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager # For lifespan context manager
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import os
from typing import Optional, Dict, Any, List # For type hints
from dotenv import load_dotenv
from pydantic import BaseModel # For data validation
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq


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

class CollectionPropertyModel(BaseModel):
    """Model for defining a property in a collection"""
    name: str
    data_type: str  # TEXT, NUMBER, BOOLEAN, DATE, etc.
    description: Optional[str] = None

class CreateCollectionModel(BaseModel):
    """Model for creating a new collection"""
    name: str
    description: Optional[str] = None
    properties: List[CollectionPropertyModel]
    use_vectorizer: bool = False  # Whether to use automatic vectorization

class TicketSubmissionModel(BaseModel):
    """Model for submitting a new ticket to get AI-generated solution"""
    title: str
    description: str
    category: str
    severity: str = "Medium"
    application: str = ""
    affected_users: str = ""
    environment: str = "Production"
    collection_name: Optional[str] = None  # Collection to search for similar tickets

class AITicketResponse(BaseModel):
    """Response model for AI-generated ticket solution"""
    success: bool
    ticket_id: str
    status: str  # "Resolved" or "Open"
    reasoning: str  # Root cause analysis
    solution: str  # Resolution steps
    similar_tickets: List[Dict[str, Any]]  # Similar tickets found
    message: str

class TicketSubmissionModel(BaseModel):
    """Model for submitting a new ticket to get AI-generated solution"""
    title: str
    description: str
    category: str
    severity: str = "Medium"
    application: str = ""
    affected_users: str = ""
    environment: str = "Production"
    collection_name: Optional[str] = None  # Collection to search for similar tickets

class AITicketResponse(BaseModel):
    """Response model for AI-generated ticket solution"""
    success: bool
    ticket_id: str
    status: str  # "Resolved" or "Open"
    reasoning: str  # Root cause analysis
    solution: str  # Resolution steps
    similar_tickets: List[Dict[str, Any]]  # Similar tickets found
    message: str

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
       
        print("📦 Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Connect to local Weaviate instance running on Docker
        weaviate_client = weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST"), # Docker container host (default: localhost)
            port=int(os.getenv("WEAVIATE_PORT")),  # Weaviate default port (default: 8080)
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT")),  # gRPC port for v4 client (default: 50051)
        )
        
        # Verify connection is ready
        if weaviate_client.is_ready():
            print("✅ Successfully connected to Weaviate vector database")

            # ========== SMART COLLECTION HANDLING ==========
            # Check if SupportTickets collection already exists
            # Collections persist in Weaviate even after app shutdown
            try:
                # Check if collection exists in Weaviate
                collection_exists = weaviate_client.collections.exists(TICKETS_COLLECTION_NAME)
                
                if collection_exists:
                    # Collection already exists - skip creation and reuse existing data
                    print(f"✅ Collection '{TICKETS_COLLECTION_NAME}' already exists (using existing collection)")
                    
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

    global embedding_model
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding model not initialized"
        )
    return embedding_model.encode(text).tolist()

def find_similar_tickets_in_weaviate(collection_name: str, query_text: str, k: int = 3, similarity_threshold: float = 0.85) -> List[Dict]:
    """
    Find similar tickets in Weaviate using vector similarity search.
    
    Args:
        collection_name: Name of the collection to search
        query_text: Query text to search for
        k: Number of results to return
        similarity_threshold: Minimum similarity score (0-1)
        
    Returns:
        List of similar tickets with metadata and similarity scores
    """
    global weaviate_client
    
    if weaviate_client is None:
        return []
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(collection_name):
            return []
        
        # Get collection
        collection = weaviate_client.collections.get(collection_name)
        
        # Generate embedding for query
        query_embedding = generate_embedding(query_text)
        
        # Perform vector search
        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=k,
            return_metadata=wvc.query.MetadataQuery(distance=True, certainty=True)
        )
        
        # Extract results with similarity filtering
        similar_tickets = []
        for obj in response.objects:
            # Weaviate certainty is already 0-1 (cosine similarity)
            similarity_score = obj.metadata.certainty
            
            # Only include tickets above threshold
            if similarity_score >= similarity_threshold:
                ticket_data = {
                    "ticket_id": obj.properties.get("ticket_id", "N/A"),
                    "title": obj.properties.get("title", ""),
                    "description": obj.properties.get("description", ""),
                    "solution": obj.properties.get("solution", ""),
                    "reasoning": obj.properties.get("reasoning", ""),
                    "category": obj.properties.get("category", ""),
                    "severity": obj.properties.get("severity", ""),
                    "similarity_score": float(similarity_score)
                }
                similar_tickets.append(ticket_data)
        
        return similar_tickets
        
    except Exception as e:
        print(f"Error finding similar tickets: {e}")
        return []

def generate_solution_with_groq(ticket_data: Dict, similar_tickets: List[Dict]) -> tuple:
    """
    Generate AI solution using Groq's open-source LLM based on ticket data and similar tickets.
    
    Args:
        ticket_data: Dictionary containing ticket information
        similar_tickets: List of similar tickets from vector DB
        
    Returns:
        Tuple of (reasoning, solution)
    """
    try:
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            return "Unable to generate root cause analysis. GROQ_API_KEY not found in environment variables."
  
        # Initialize Groq LLM (open-source model) with explicit API key
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Open-source Llama model via Groq
            temperature=0.1,
            api_key=groq_api_key  # Explicitly pass the API key
        )
        
        # Prepare context from similar tickets
        context = ""
        if similar_tickets:
            context = "### Similar Past Cloud Application Issues (85%+ match confidence):\n\n"
            for i, ticket in enumerate(similar_tickets, 1):
                similarity_percent = ticket['similarity_score'] * 100
                context += f"**Past Incident {i} (Match: {similarity_percent:.1f}%):**\n"
                context += f"Title: {ticket['title']}\n"
                context += f"Description: {ticket['description']}\n"
                context += f"Solution: {ticket['solution']}\n"
                context += f"Reasoning: {ticket['reasoning']}\n\n"
        else:
            context = "No similar past incidents found in the knowledge base (minimum 85% similarity required).\n\n"
        
        # Create prompt
        prompt = f"""You are an expert cloud application support engineer. A support team member has received a new incident ticket from application monitoring or end-users.

### Current Incident:
**Title:** {ticket_data['title']}
**Description:** {ticket_data['description']}
**Category:** {ticket_data['category']}
**Severity:** {ticket_data['severity']}
**Application:** {ticket_data['application']}
**Environment:** {ticket_data['environment']}
**Affected Users:** {ticket_data['affected_users']}

{context}

Based on the current incident and similar past incidents (if any), please provide:

1. **Root Cause Analysis:** Explain the likely root cause of this issue based on the symptoms. Consider cloud infrastructure, application code, database, API dependencies, or configuration issues.

2. **Resolution Steps:** Provide clear, actionable steps to resolve this incident. Include:
   - Immediate actions to mitigate impact
   - Diagnostic commands/queries to verify the issue
   - Fix implementation steps
   - Verification steps to confirm resolution
   - Preventive measures to avoid recurrence

Be specific to cloud applications, microservices, APIs, databases, and modern DevOps practices. Reference similar past incidents when applicable.

Format your response as:
ROOT CAUSE: <your analysis here>
RESOLUTION: <your step-by-step solution here>
"""
        
        # Get response from Groq LLM
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Parse reasoning and solution
        reasoning = ""
        solution = ""
        
        if "ROOT CAUSE:" in response_text and "RESOLUTION:" in response_text:
            parts = response_text.split("RESOLUTION:")
            reasoning = parts[0].replace("ROOT CAUSE:", "").strip()
            solution = parts[1].strip()
        else:
            # Fallback if format is not followed
            lines = response_text.split("\n", 1)
            reasoning = lines[0] if len(lines) > 0 else "Analysis based on incident description."
            solution = lines[1] if len(lines) > 1 else response_text
        
        return reasoning, solution
    
    except Exception as e:
        error_msg = str(e)
        
        # Provide specific error messages
        if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
            return ("Unable to generate root cause analysis. API authentication failed. Please check your GROQ_API_KEY in .env file.", 
                    "Unable to generate resolution steps. API authentication error - verify your Groq API key is correct.")
        elif "rate limit" in error_msg.lower():
            return ("Unable to generate root cause analysis. Rate limit exceeded.", 
                    "Unable to generate resolution steps. Try again in a moment or switch to llama-3.1-8b-instant model for higher rate limits.")
        else:
            return (f"Unable to generate root cause analysis. Error: {error_msg}", 
                    f"Unable to generate resolution steps. Please investigate manually. Error: {error_msg}")


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

# # ===================== CREATE COLLECTION ENDPOINT =====================
# @app.post("/api/v1/collections", tags=["Collections"])
# async def create_collection(collection_model: CreateCollectionModel):
#     """
#     Create a new collection in Weaviate with custom schema.
    
#     Args:
#         collection_model: CreateCollectionModel with collection name, description, and properties
        
#     Returns:
#         Dict: Success status and collection details
        
#     Raises:
#         HTTPException: If collection creation fails or already exists
#     """
#     global weaviate_client
    
#     try:
#         # Check if Weaviate client is initialized
#         if weaviate_client is None:
#             raise HTTPException(
#                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#                 detail="Weaviate client not initialized. Check if Weaviate is running."
#             )
        
#         # Check if collection already exists
#         if weaviate_client.collections.exists(collection_model.name):
#             raise HTTPException(
#                 status_code=status.HTTP_409_CONFLICT,
#                 detail=f"Collection '{collection_model.name}' already exists"
#             )
        
#         # Map string data types to Weaviate DataType enums
#         data_type_mapping = {
#             "TEXT": DataType.TEXT,
#             "NUMBER": DataType.NUMBER,
#             "INT": DataType.INT,
#             "BOOLEAN": DataType.BOOL,
#             "DATE": DataType.DATE,
#             "UUID": DataType.UUID,
#             "TEXT_ARRAY": DataType.TEXT_ARRAY,
#             "NUMBER_ARRAY": DataType.NUMBER_ARRAY,
#             "INT_ARRAY": DataType.INT_ARRAY,
#             "BOOLEAN_ARRAY": DataType.BOOL_ARRAY,
#             "DATE_ARRAY": DataType.DATE_ARRAY,
#             "UUID_ARRAY": DataType.UUID_ARRAY,
#         }
        
#         # Build properties list
#         properties = []
#         for prop in collection_model.properties:
#             # Validate data type
#             data_type_upper = prop.data_type.upper()
#             if data_type_upper not in data_type_mapping:
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"Invalid data type '{prop.data_type}'. Supported types: {', '.join(data_type_mapping.keys())}"
#                 )
            
#             # Create property
#             properties.append(
#                 Property(
#                     name=prop.name,
#                     data_type=data_type_mapping[data_type_upper],
#                     description=prop.description or f"Property: {prop.name}"
#                 )
#             )
        
#         # Create collection with or without vectorizer
#         vectorizer_config = wvc.config.Configure.Vectorizer.none() if not collection_model.use_vectorizer else None
        
#         weaviate_client.collections.create(
#             name=collection_model.name,
#             description=collection_model.description or f"Custom collection: {collection_model.name}",
#             vectorizer_config=vectorizer_config,
#             properties=properties
#         )
        
#         return {
#             "success": True,
#             "message": f"Collection '{collection_model.name}' created successfully",
#             "collection": {
#                 "name": collection_model.name,
#                 "description": collection_model.description,
#                 "properties_count": len(properties),
#                 "vectorizer": "manual/local" if not collection_model.use_vectorizer else "automatic",
#                 "properties": [
#                     {
#                         "name": prop.name,
#                         "data_type": prop.data_type,
#                         "description": prop.description
#                     } for prop in collection_model.properties
#                 ]
#             }
#         }
        
#     except HTTPException:
#         # Re-raise HTTP exceptions
#         raise
#     except Exception as e:
#         # Handle any other errors during collection creation
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error creating collection: {str(e)}"
#         )

# ===================== CREATE COLLECTION ENDPOINT =====================
@app.post("/api/v1/collections", tags=["Collections"])
async def create_collection(collection_name: str):
    """
    Create a new collection with default ticket schema by just providing a name.
    This is a simplified endpoint that creates a collection with predefined ticket properties.
    
    Args:
        collection_name: Name of the collection to create (query parameter)
        
    Returns:
        Dict: Success status and collection details
        
    Raises:
        HTTPException: If collection creation fails or already exists
    """
    global weaviate_client
    
    try:
        # Check if Weaviate client is initialized
        if weaviate_client is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Weaviate client not initialized. Check if Weaviate is running."
            )
        
        # Check if collection already exists
        if weaviate_client.collections.exists(collection_name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection '{collection_name}' already exists"
            )
        
        # Create collection with default ticket schema
        # Using the same schema as SupportTickets for consistency
        weaviate_client.collections.create(
            name=collection_name,
            description=f"Support ticket collection: {collection_name}",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),  # Manual/local embeddings
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
        
        return {
            "success": True,
            "message": f"Collection '{collection_name}' created successfully with default ticket schema",
            "collection": {
                "name": collection_name,
                "description": f"Support ticket collection: {collection_name}",
                "schema_type": "default_ticket_schema",
                "vectorizer": "manual/local (sentence-transformers)",
                "properties_count": 12,
                "properties": [
                    "ticket_id", "title", "description", "category", "status", 
                    "severity", "application", "affected_users", "environment", 
                    "solution", "reasoning", "timestamp"
                ]
            }
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any other errors during collection creation
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating collection: {str(e)}"
        )

# ===================== TICKET UPLOAD ENDPOINT =====================
@app.post("/api/v1/tickets", tags=["Tickets"], response_model=TicketResponse)
async def upload_ticket(ticket: TicketModel, collection_name: Optional[str] = None):
    """
    Upload a single ticket to Weaviate collection with local embeddings.
    
    Args:
        ticket: TicketModel with all required fields
        collection_name: Name of the collection to upload to (default: SupportTickets)
        
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
    
    # Use default collection name if not provided
    target_collection = collection_name or TICKETS_COLLECTION_NAME
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist"
            )
        
        # Get the target collection
        tickets_collection = weaviate_client.collections.get(target_collection)
        
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
            message=f"Ticket {ticket.ticket_id} uploaded successfully to collection '{target_collection}' with embedding",
            ticket_id=ticket.ticket_id,
            data={"uuid": str(uuid), "ticket": ticket_data, "collection": target_collection}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading ticket: {str(e)}"
        )

# ===================== BATCH TICKET UPLOAD ENDPOINT =====================
@app.post("/api/v1/tickets/batch", tags=["Tickets"])
async def upload_tickets_batch(tickets: List[TicketModel], collection_name: Optional[str] = None):
    """
    Upload multiple tickets to Weaviate collection in batch with local embeddings.
    
    Args:
        tickets: List of TicketModel objects
        collection_name: Name of the collection to upload to (default: SupportTickets)
        
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
    
    # Use default collection name if not provided
    target_collection = collection_name or TICKETS_COLLECTION_NAME
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist"
            )
        
        # Get the target collection
        tickets_collection = weaviate_client.collections.get(target_collection)
        
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
            "message": f"Batch upload completed to collection '{target_collection}' with local embeddings",
            "collection": target_collection,
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

# ===================== SUBMIT TICKET WITH AI SOLUTION ENDPOINT =====================
@app.post("/api/v1/tickets/submit-user-input", tags=["Tickets"], response_model=AITicketResponse)
async def submit_ticket_with_ai_solution(ticket: TicketSubmissionModel):
    """
    This endpoint uses RAG (Retrieval-Augmented Generation) to find similar tickets and generate solutions.
    
    Workflow:
    1. Search vector DB for similar past incidents (85% similarity threshold)
    2. Generate AI solution using Groq's Llama model based on similar tickets
    3. Return reasoning, solution, and similar tickets found (WITHOUT saving to DB)
    
    Args:
        ticket: TicketSubmissionModel with incident details
        
    Returns:
        AITicketResponse: Generated ticket ID, status, AI reasoning, solution, and similar tickets
        
    Raises:
        HTTPException: If Weaviate is not connected or processing fails
    """
    global weaviate_client
    
    # Check if Weaviate client is connected
    if weaviate_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Weaviate vector database is not connected. Please check connection."
        )
    
    try:
        # Use default collection name if not provided
        target_collection = ticket.collection_name or TICKETS_COLLECTION_NAME
        
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist. Create it first."
            )
        
        # Step 1: Search for similar tickets in vector DB
        query_text = f"{ticket.title} {ticket.description} {ticket.application}"
        similar_tickets = find_similar_tickets_in_weaviate(
            collection_name=target_collection,
            query_text=query_text,
            k=3,
            similarity_threshold=0.85
        )
        
        # Step 2: Prepare ticket data for AI
        ticket_data = {
            "title": ticket.title,
            "description": ticket.description,
            "category": ticket.category,
            "severity": ticket.severity,
            "application": ticket.application,
            "affected_users": ticket.affected_users,
            "environment": ticket.environment
        }
        
        # Step 3: Generate AI solution using Groq's open-source LLM
        reasoning, solution = generate_solution_with_groq(ticket_data, similar_tickets)
        
        # Step 4: Determine ticket status
        has_error = reasoning.startswith("Unable to generate") or solution.startswith("Unable to generate")
        has_similar_tickets = len(similar_tickets) > 0
        
        if has_error:
            status_value = "Open"
            message = "AI generation failed. Manual review recommended."
        elif not has_similar_tickets:
            status_value = "Open"
            message = "No similar incidents found (85% threshold). Expert validation recommended."
        else:
            status_value = "Resolved"
            message = f"AI-generated solution based on {len(similar_tickets)} similar incident(s)."
        
        # Step 5: Generate temporary ticket ID (not saved to DB)
        collection = weaviate_client.collections.get(target_collection)
        count_result = collection.aggregate.over_all(total_count=True)
        ticket_count = count_result.total_count
        ticket_id = f"TKT-PREVIEW-{ticket_count + 1:04d}"
        
        # Step 6: Format similar tickets for response
        similar_tickets_response = [
            {
                "ticket_id": st["ticket_id"],
                "title": st["title"],
                "similarity_score": st["similarity_score"],
                "similarity_percent": f"{st['similarity_score'] * 100:.1f}%",
                "category": st["category"],
                "severity": st["severity"]
            }
            for st in similar_tickets
        ]
        
        # Return response (ticket NOT saved to database)
        return AITicketResponse(
            success=True,
            ticket_id=ticket_id,
            status=status_value,
            reasoning=reasoning,
            solution=solution,
            similar_tickets=similar_tickets_response,
            message=f"{message} Note: Ticket not saved to database."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing ticket submission: {str(e)}"
        )

# ===================== GET ALL TICKETS ENDPOINT =====================
@app.get("/api/v1/tickets", tags=["Tickets"])
async def get_all_tickets(limit: int = 100, offset: int = 0, collection_name: Optional[str] = None):
    """
    Retrieve all tickets from Weaviate collection.
    
    Args:
        limit: Maximum number of tickets to return (default 100)
        offset: Number of tickets to skip (default 0)
        collection_name: Name of the collection to query (default: SupportTickets)
        
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
    
    # Use default collection name if not provided
    target_collection = collection_name or TICKETS_COLLECTION_NAME
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist"
            )
        
        # Get the target collection
        tickets_collection = weaviate_client.collections.get(target_collection)
        
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
            "collection": target_collection,
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

# ===================== SEARCH TICKETS ENDPOINT =====================
@app.get("/api/v1/tickets/search", tags=["Tickets"])
async def search_tickets(query: str, limit: int = 3, collection_name: Optional[str] = None):
    """
    Search for similar tickets using vector similarity search with local embeddings.
    
    Args:
        query: Search query (natural language description)
        limit: Maximum number of results (default 5)
        collection_name: Name of the collection to search in (default: SupportTickets)
        
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
    
    # Use default collection name if not provided
    target_collection = collection_name or TICKETS_COLLECTION_NAME
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist"
            )
        
        # Get the target collection
        tickets_collection = weaviate_client.collections.get(target_collection)
        
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
            "collection": target_collection,
            "query": query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching tickets: {str(e)}"
        )

# ===================== GET TICKET BY ID ENDPOINT =====================
@app.get("/api/v1/tickets/{ticket_id}", tags=["Tickets"])
async def get_ticket_by_id(ticket_id: str, collection_name: Optional[str] = None):
    """
    Retrieve a specific ticket by its ticket_id.
    
    Args:
        ticket_id: Unique ticket identifier (e.g., TKT-0001)
        collection_name: Name of the collection to search in (default: SupportTickets)
        
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
    
    # Use default collection name if not provided
    target_collection = collection_name or TICKETS_COLLECTION_NAME
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist"
            )
        
        # Get the target collection
        tickets_collection = weaviate_client.collections.get(target_collection)
        
        # Query for specific ticket_id
        response = tickets_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("ticket_id").equal(ticket_id),
            limit=1
        )
        
        if len(response.objects) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket {ticket_id} not found in collection '{target_collection}'"
            )
        
        # Extract ticket data
        obj = response.objects[0]
        ticket_data = obj.properties
        ticket_data["uuid"] = str(obj.uuid)
        
        return {
            "success": True,
            "collection": target_collection,
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
async def delete_ticket_by_id(ticket_id: str, collection_name: Optional[str] = None):
    """
    Delete a specific ticket from Weaviate by its ticket_id.
    
    Args:
        ticket_id: Unique ticket identifier (e.g., TKT-0001)
        collection_name: Name of the collection to delete from (default: SupportTickets)
        
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
    
    # Use default collection name if not provided
    target_collection = collection_name or TICKETS_COLLECTION_NAME
    
    try:
        # Check if collection exists
        if not weaviate_client.collections.exists(target_collection):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{target_collection}' does not exist"
            )
        
        # Get the target collection
        tickets_collection = weaviate_client.collections.get(target_collection)
        
        # First, find the ticket to get its UUID
        response = tickets_collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("ticket_id").equal(ticket_id),
            limit=1
        )
        
        # Check if ticket exists
        if len(response.objects) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket '{ticket_id}' not found in collection '{target_collection}'"
            )
        
        # Get the UUID of the ticket to delete
        ticket_uuid = response.objects[0].uuid
        ticket_data = response.objects[0].properties
        
        # Delete the ticket by UUID
        tickets_collection.data.delete_by_id(ticket_uuid)
        
        return {
            "success": True,
            "message": f"Ticket '{ticket_id}' deleted successfully from collection '{target_collection}'",
            "collection": target_collection,
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

