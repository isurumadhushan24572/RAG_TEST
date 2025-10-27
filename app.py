import torch
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

# """
# CLOUD APPLICATION SUPPORT MANAGEMENT SYSTEM WITH RAG

# ARCHITECTURE:
# 1. USER TICKETS (tickets.json) - All user-submitted incidents (Open & Resolved)
#    - Stored permanently in JSON
#    - NOT added to vector database
#    - Used for incident tracking and reporting

# 2. KNOWLEDGE BASE (tickets_sample.json) - Pre-loaded sample incidents
#    - Loaded into FAISS vector database (READ-ONLY)
#    - Used ONLY for similarity matching with RAG
#    - Contains resolved technical incidents for reference
#    - New user tickets DO NOT update this knowledge base

# WORKFLOW:
# - User submits new incident → Saved to tickets.json
# - System searches vector DB (sample tickets) for similar issues
# - AI generates solution based on similar past incidents
# - New ticket is NOT added to vector DB (KB remains static)
# """


# ===================== TICKET DATA STRUCTURE =====================
# Ticket class to manage service tickets

class Ticket:
    """Represents a service ticket with issue and solution."""
    def __init__(self, ticket_id: str, title: str, description: str, 
                 category: str, status: str = "Open", solution: str = "", 
                 reasoning: str = "", timestamp: str = None, severity: str = "Medium",
                 application: str = "", affected_users: str = "", environment: str = "Production"):
        self.ticket_id = ticket_id
        self.title = title
        self.description = description
        self.category = category
        self.status = status
        self.solution = solution
        self.reasoning = reasoning
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.severity = severity
        self.application = application
        self.affected_users = affected_users
        self.environment = environment

    # Convert ticket to dictionary.
    def to_dict(self):
        
        return {
            "ticket_id": self.ticket_id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "solution": self.solution,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "severity": self.severity,
            "application": self.application,
            "affected_users": self.affected_users,
            "environment": self.environment
        }
    
    # Create ticket from dictionary.
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


# ===================== TICKET STORAGE MANAGER =====================
# Functions to save and load tickets

class TicketManager:

    # Manages ticket storage and retrieval.
    
    def __init__(self, storage_file="tickets.json"):
        self.storage_file = storage_file
        self.tickets = self.load_tickets()
    
    def load_tickets(self) -> List[Ticket]:
        """Load tickets from JSON file."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return [Ticket.from_dict(t) for t in data]
            except Exception as e:
                st.error(f"Error loading tickets: {e}")
                return []
        return []
    
    def save_tickets(self):
        """Save tickets to JSON file."""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump([t.to_dict() for t in self.tickets], f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving tickets: {e}")
    
    def add_ticket(self, ticket: Ticket):
        """Add a new ticket."""
        self.tickets.append(ticket)
        self.save_tickets()
    
    def update_ticket(self, ticket_id: str, solution: str, reasoning: str, status: str = "Resolved"):
        """Update a ticket with solution."""
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                ticket.solution = solution
                ticket.reasoning = reasoning
                ticket.status = status
                self.save_tickets()
                return True
        return False
    
    def get_tickets_by_severity(self, severity: str) -> List[Ticket]:
        """Get tickets filtered by severity."""
        return [t for t in self.tickets if t.severity == severity]
    
    def get_tickets_by_application(self, application: str) -> List[Ticket]:
        """Get tickets filtered by application."""
        return [t for t in self.tickets if t.application == application]
    
    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get a specific ticket."""
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None
    
    def get_all_tickets(self) -> List[Ticket]:
        """Get all tickets."""
        return self.tickets
    
    def get_resolved_tickets(self) -> List[Ticket]:
        """Get only resolved tickets with solutions."""
        return [t for t in self.tickets if t.status == "Resolved" and t.solution]
    
    def generate_ticket_id(self) -> str:
        """Generate a unique ticket ID."""
        return f"TKT-{len(self.tickets) + 1:04d}"


# ===================== RAG SYSTEM FOR TICKETS =====================
# Functions to create vector store and find similar tickets

def create_ticket_embeddings(tickets: List[Ticket]):
    """Create vector embeddings from historical tickets."""
    if not tickets:
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )
    
    # Create text documents from tickets (title + description + solution)
    documents = []
    metadatas = []
    
    for ticket in tickets:
        text = f"Title: {ticket.title}\nDescription: {ticket.description}\nSolution: {ticket.solution}\nReasoning: {ticket.reasoning}"
        documents.append(text)
        metadatas.append({
            "ticket_id": ticket.ticket_id,
            "category": ticket.category,
            "title": ticket.title,
            "severity": ticket.severity
        })
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas)
    return vector_store


def find_similar_tickets(vector_store, query: str, k: int = 3, similarity_threshold: float = 0.85) -> List[Dict]:
    """Find similar tickets using RAG with cosine similarity threshold.
    
    Args:
        vector_store: FAISS vector store containing ticket embeddings
        query: Search query string
        k: Number of candidates to retrieve
        similarity_threshold: Minimum cosine similarity (0-1). 
    
    Returns:
        List of similar tickets with similarity >= threshold
    """
    if not vector_store:
        return []
    
    try:
        # Search for similar tickets (FAISS returns L2 distance, need to convert)
        results = vector_store.similarity_search_with_score(query, k=k)
        similar_tickets = []
        
        for doc, score in results:
            # FAISS returns L2 distance - convert to cosine similarity
            # For normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
            # Approximate conversion for similarity score
            cosine_similarity = 1 - (score ** 2 / 2)

            
            # Only include tickets above threshold
            if cosine_similarity >= similarity_threshold:
                similar_tickets.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(cosine_similarity)
                })
        
        return similar_tickets
    except Exception as e:
        st.error(f"Error finding similar tickets: {e}")
        return []


# ===================== AI SOLUTION GENERATOR =====================
# Function to generate solution using LLM and similar tickets

def generate_solution_with_rag(current_ticket: Ticket, similar_tickets: List[Dict]) -> tuple:
    """Generate solution and reasoning using Groq (open-source models) and RAG."""
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  
            temperature=0.0
        )
        
        # Prepare context from similar tickets
        context = ""
        if similar_tickets:
            context = "### Similar Past Cloud Application Issues (85% match confidence):\n\n"
            for i, ticket in enumerate(similar_tickets, 1):
                similarity_percent = ticket['similarity_score'] * 100
                context += f"**Past Incident {i} (Match: {similarity_percent:.1f}%):**\n{ticket['content']}\n\n"
        else:
            context = "No similar past incidents found in the knowledge base (minimum 85% similarity required).\n\n"
        
        # Create prompt specific to cloud application support
        prompt = f"""You are an expert cloud application support engineer. A support team member has received a new incident ticket from application monitoring or end-users.
        
### Current Incident:
**Title:** {current_ticket.title}
**Description:** {current_ticket.description}
**Category:** {current_ticket.category}
**Severity:** {current_ticket.severity}
**Application:** {current_ticket.application}
**Environment:** {current_ticket.environment}
**Affected Users:** {current_ticket.affected_users}

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
        
        # Get response from LLM
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
        st.error(f"Error generating solution: {error_msg}")
        
        # Provide specific error messages based on error type
        if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
            return ("Unable to generate root cause analysis. API authentication failed. Please check your GROQ_API_KEY in .env file.", 
                    "Unable to generate resolution steps. API authentication error - verify your Groq API key is correct.")
        elif "rate limit" in error_msg.lower():
            return ("Unable to generate root cause analysis. Rate limit exceeded.", 
                    "Unable to generate resolution steps. Try again in a moment or switch to llama-3.1-8b-instant model for higher rate limits.")
        else:
            return (f"Unable to generate root cause analysis. Error: {error_msg}", 
                    f"Unable to generate resolution steps. Please investigate manually. Error: {error_msg}")


# ===================== STREAMLIT APP =====================
# Main application UI

def main():
    """Main Streamlit application for Service Management System."""
    
    load_dotenv()
    st.set_page_config(page_title="Cloud Application Support System", page_icon="☁️", layout="wide")
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #0078D4;'>
        ☁️ Cloud Application Support Management System
        </h1>
        <p style='text-align: center; color: #555;'>
        Monitor and resolve cloud application incidents with AI-powered root cause analysis and solutions
        </p>
    """, unsafe_allow_html=True)
    
    # Initialize ticket manager
    if "ticket_manager" not in st.session_state:
        st.session_state.ticket_manager = TicketManager()
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    ticket_manager = st.session_state.ticket_manager
    
    # Initialize vector store from SAMPLE TICKETS ONLY (read-only knowledge base)
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        with st.spinner("Loading incident knowledge base from sample tickets..."):
            # Load ONLY from tickets_sample.json for vector store (read-only KB)
            if os.path.exists("tickets_sample.json"):
                try:
                    with open("tickets_sample.json", 'r', encoding='utf-8') as f:
                        sample_data = json.load(f)
                        sample_tickets = [Ticket.from_dict(t) for t in sample_data if t.get("status") == "Resolved"]
                        
                    if sample_tickets:
                        st.session_state.vector_store = create_ticket_embeddings(sample_tickets)
                        st.sidebar.success(f"✅ Knowledge Base: {len(sample_tickets)} sample incidents loaded")
                    else:
                        st.sidebar.warning("⚠️ No resolved sample tickets found")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading sample tickets: {e}")
            else:
                st.sidebar.warning("⚠️ tickets_sample.json not found - Knowledge base empty")
    
    # Sidebar navigation
    st.sidebar.title("🎯 Support Dashboard")
    page = st.sidebar.radio("Go to:", ["Report New Incident", "All Incidents", "Incident Details", "Support Metrics"])
    
    # ==================== PAGE: REPORT NEW INCIDENT ====================
    if page == "Report New Incident":
        st.header("� Report New Application Incident")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            ticket_title = st.text_input("Incident Summary", placeholder="Brief description of the issue")
            ticket_description = st.text_area("Detailed Description", 
                                             placeholder="Include error messages, affected functionality, time started, steps to reproduce...", 
                                             height=150)
            
            col_app, col_env = st.columns(2)
            with col_app:
                application = st.text_input("Application/Service", placeholder="e.g., Payment API, User Portal")
            with col_env:
                environment = st.selectbox("Environment", ["Production", "Staging", "Development", "UAT"])
        
        with col2:
            ticket_category = st.selectbox("Category", 
                                          ["API Issues", "Database", "Authentication", "Performance", 
                                           "Deployment", "Integration", "UI/Frontend", "Backend Service", "Other"])
            
            severity = st.selectbox("Severity", 
                                   ["Critical", "High", "Medium", "Low"],
                                   help="Critical: Service down | High: Major feature broken | Medium: Moderate impact | Low: Minor issue")
            
            affected_users = st.text_input("Affected Users", placeholder="All users / Specific tenant / Region")
        
        if st.button("Submit Incident & Get AI Analysis", type="primary"):
            if ticket_title and ticket_description:
                with st.spinner("Analyzing incident and searching knowledge base..."):
                    # Create new ticket
                    ticket_id = ticket_manager.generate_ticket_id()
                    new_ticket = Ticket(
                        ticket_id=ticket_id,
                        title=ticket_title,
                        description=ticket_description,
                        category=ticket_category,
                        severity=severity,
                        application=application,
                        affected_users=affected_users,
                        environment=environment
                    )
                    
                    # Find similar tickets using RAG
                    similar_tickets = []
                    if st.session_state.vector_store:
                        query = f"{ticket_title} {ticket_description} {application}"
                        similar_tickets = find_similar_tickets(st.session_state.vector_store, query, k=3)
                    
                    # Generate solution using AI
                    reasoning, solution = generate_solution_with_rag(new_ticket, similar_tickets)
                    
                    # Update ticket with solution
                    new_ticket.solution = solution
                    new_ticket.reasoning = reasoning
                    
                    # Determine status based on knowledge base match and AI success
                    has_error = reasoning.startswith("Unable to generate") or solution.startswith("Unable to generate")
                    has_similar_tickets = len(similar_tickets) > 0
                    
                    if has_error:
                        # AI failed due to error (API key, rate limit, etc.) - mark as Open
                        new_ticket.status = "Open"
                        status_message = "⚠️ AI generation failed. Incident marked as Open for manual review."
                    elif not has_similar_tickets:
                        # No similar tickets found in KB - new type of issue, mark as Open for manual verification
                        new_ticket.status = "Open"
                        status_message = "⚠️ No similar incidents in knowledge base. Incident marked as Open for expert review and validation."
                    else:
                        # Similar tickets found and AI generated solution successfully - mark as Resolved
                        new_ticket.status = "Resolved"
                        status_message = "✅ AI-generated solution based on similar past incidents. Incident marked as Resolved."
                    
                    # Save ticket to JSON only (DO NOT add to vector store)
                    ticket_manager.add_ticket(new_ticket)
                    
                    # NOTE: Vector store is READ-ONLY and only contains pre-loaded sample tickets
                    # New tickets are NOT added to vector store - they're only saved to tickets.json
                    
                    # Show status message with severity badge
                    severity_colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                    if new_ticket.status == "Resolved":
                        st.success(f"✅ Incident {ticket_id} logged successfully! {severity_colors[severity]} Severity: {severity}")
                        st.info(status_message)
                        if has_similar_tickets:
                            st.caption(f"📊 Matched with {len(similar_tickets)} similar past incident(s)")
                    else:
                        st.warning(f"⚠️ Incident {ticket_id} logged as Open. {severity_colors[severity]} Severity: {severity}")
                        st.info(status_message)
                        if not has_similar_tickets and not has_error:
                            st.caption("💡 This appears to be a new type of issue. Please review and validate the AI-generated solution before marking as resolved.")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("🔍 AI Analysis Results")
                    
                    # Similar incidents
                    if similar_tickets:
                        with st.expander(f"📚 {len(similar_tickets)} Similar Past Incident(s) Found (85% match)", expanded=True):
                            for i, ticket in enumerate(similar_tickets, 1):
                                similarity_percent = ticket['similarity_score'] * 100
                                match_color = "🟢" if similarity_percent >= 90 else "🟡" if similarity_percent >= 80 else "🟠"
                                st.markdown(f"{match_color} **{i}. {ticket['metadata']['title']}** (ID: {ticket['metadata']['ticket_id']})")
                                st.caption(f"Match Confidence: {similarity_percent:.1f}%")
                                st.text(ticket['content'][:250] + "...")
                                st.markdown("---")
                    else:
                        st.info("ℹ️ No similar incidents found with 85% similarity. This appears to be a new type of issue.")
                    
                    # Root Cause
                    st.subheader("🔬 Root Cause Analysis")
                    if reasoning.startswith("Unable to generate"):
                        st.error(reasoning)
                        st.info("💡 **Tip**: This might be a new type of issue not in the knowledge base. Please add manual resolution to improve future recommendations.")
                    else:
                        st.info(reasoning)
                    
                    # Resolution
                    st.subheader("✅ Resolution Steps")
                    if solution.startswith("Unable to generate"):
                        st.error(solution)
                        st.warning("⚠️ **Action Required**: Please investigate manually and add resolution in 'Incident Details' page to help build the knowledge base.")
                    else:
                        st.success(solution)
            else:
                st.warning("⚠️ Please fill in incident summary and detailed description.")
    
    # ==================== PAGE: VIEW ALL INCIDENTS ====================
    elif page == "All Incidents":
        st.header("📊 All Application Incidents")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Open", "Resolved"])
        with col2:
            category_filter = st.selectbox("Filter by Category", 
                                          ["All", "API Issues", "Database", "Authentication", "Performance", 
                                           "Deployment", "Integration", "UI/Frontend", "Backend Service", "Other"])
        with col3:
            severity_filter = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium", "Low"])
        with col4:
            env_filter = st.selectbox("Filter by Environment", ["All", "Production", "Staging", "Development", "UAT"])
        
        # Get tickets
        all_tickets = ticket_manager.get_all_tickets()
        
        # Apply filters
        filtered_tickets = all_tickets
        if status_filter != "All":
            filtered_tickets = [t for t in filtered_tickets if t.status == status_filter]
        if category_filter != "All":
            filtered_tickets = [t for t in filtered_tickets if t.category == category_filter]
        if severity_filter != "All":
            filtered_tickets = [t for t in filtered_tickets if t.severity == severity_filter]
        if env_filter != "All":
            filtered_tickets = [t for t in filtered_tickets if t.environment == env_filter]
        
        # Statistics
        st.markdown("### 📈 Incident Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Incidents", len(all_tickets))
        with col2:
            open_tickets = len([t for t in all_tickets if t.status == "Open"])
            st.metric("Open", open_tickets, delta=None, delta_color="inverse")
        with col3:
            resolved_count = len([t for t in all_tickets if t.status == "Resolved"])
            st.metric("Resolved", resolved_count)
        with col4:
            critical_count = len([t for t in all_tickets if t.severity == "Critical"])
            st.metric("🔴 Critical", critical_count, delta=None, delta_color="inverse")
        with col5:
            resolution_rate = (resolved_count / len(all_tickets) * 100) if all_tickets else 0
            st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        
        st.markdown("---")
        
        # Display tickets
        if filtered_tickets:
            # Severity icons
            severity_icons = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
            
            for ticket in reversed(filtered_tickets):  # Show newest first
                status_color = "🟢" if ticket.status == "Resolved" else "🔴"
                severity_icon = severity_icons.get(ticket.severity, "⚪")
                
                with st.expander(f"{status_color} {severity_icon} [{ticket.ticket_id}] {ticket.title} - {ticket.application}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Description:** {ticket.description}")
                        if ticket.solution:
                            st.markdown(f"**Resolution:** {ticket.solution[:200]}...")
                    with col2:
                        st.markdown(f"**Category:** {ticket.category}")
                        st.markdown(f"**Severity:** {severity_icon} {ticket.severity}")
                        st.markdown(f"**Status:** {ticket.status}")
                        st.markdown(f"**Environment:** {ticket.environment}")
                        st.markdown(f"**Application:** {ticket.application}")
                        st.markdown(f"**Reported:** {ticket.timestamp}")
        else:
            st.info("No incidents found matching the filters.")
    
    # ==================== PAGE: INCIDENT DETAILS ====================
    elif page == "Incident Details":
        st.header("🔍 Incident Details")
        
        all_tickets = ticket_manager.get_all_tickets()
        
        if all_tickets:
            ticket_ids = [t.ticket_id for t in all_tickets]
            selected_id = st.selectbox("Select Incident ID", ticket_ids)
            
            ticket = ticket_manager.get_ticket(selected_id)
            
            if ticket:
                # Display ticket info
                severity_icons = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                severity_icon = severity_icons.get(ticket.severity, "⚪")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"Incident: {ticket.ticket_id}")
                    st.markdown(f"### {ticket.title}")
                    st.markdown(f"**Application/Service:** {ticket.application}")
                    st.markdown(f"**Description:**")
                    st.write(ticket.description)
                
                with col2:
                    status_badge = "🟢 Resolved" if ticket.status == "Resolved" else "🔴 Open"
                    st.markdown(f"**Status:** {status_badge}")
                    st.markdown(f"**Severity:** {severity_icon} {ticket.severity}")
                    st.markdown(f"**Category:** {ticket.category}")
                    st.markdown(f"**Environment:** {ticket.environment}")
                    st.markdown(f"**Affected Users:** {ticket.affected_users}")
                    st.markdown(f"**Reported:** {ticket.timestamp}")
                
                st.markdown("---")
                
                if ticket.reasoning:
                    st.subheader("� Root Cause Analysis")
                    st.info(ticket.reasoning)
                
                if ticket.solution:
                    st.subheader("✅ Resolution Steps")
                    st.success(ticket.solution)
                else:
                    st.warning("⚠️ No resolution available yet for this incident.")
                    
                    # Allow manual solution input
                    st.markdown("---")
                    st.subheader("Add Resolution Manually")
                    manual_reasoning = st.text_area("Root Cause Analysis", height=100)
                    manual_solution = st.text_area("Resolution Steps", height=150)
                    
                    if st.button("Save Resolution"):
                        if manual_reasoning and manual_solution:
                            ticket_manager.update_ticket(ticket.ticket_id, 
                                                        manual_solution, 
                                                        manual_reasoning, 
                                                        "Resolved")
                            st.success("✅ Resolution saved successfully!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Please provide both root cause analysis and resolution steps.")
        else:
            st.info("ℹ️ No incidents available. Report a new incident first!")
    
    # ==================== PAGE: SUPPORT METRICS ====================
    elif page == "Support Metrics":
        st.header("📊 Support Team Metrics & Analytics")
        
        all_tickets = ticket_manager.get_all_tickets()
        
        if all_tickets:
            # Overall metrics
            st.markdown("### 🎯 Overall Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            total = len(all_tickets)
            open_count = len([t for t in all_tickets if t.status == "Open"])
            resolved = len([t for t in all_tickets if t.status == "Resolved"])
            
            with col1:
                st.metric("Total Incidents", total)
            with col2:
                st.metric("Open Incidents", open_count, delta=None, delta_color="inverse")
            with col3:
                st.metric("Resolved", resolved)
            with col4:
                resolution_rate = (resolved / total * 100) if total > 0 else 0
                st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
            
            st.markdown("---")
            
            # Severity breakdown
            st.markdown("### 🎚️ Incidents by Severity")
            col1, col2, col3, col4 = st.columns(4)
            
            critical = len([t for t in all_tickets if t.severity == "Critical"])
            high = len([t for t in all_tickets if t.severity == "High"])
            medium = len([t for t in all_tickets if t.severity == "Medium"])
            low = len([t for t in all_tickets if t.severity == "Low"])
            
            with col1:
                st.metric("🔴 Critical", critical)
            with col2:
                st.metric("🟠 High", high)
            with col3:
                st.metric("🟡 Medium", medium)
            with col4:
                st.metric("🟢 Low", low)
            
            st.markdown("---")
            
            # Category breakdown
            st.markdown("### 📂 Incidents by Category")
            categories = {}
            for ticket in all_tickets:
                categories[ticket.category] = categories.get(ticket.category, 0) + 1
            
            cols = st.columns(4)
            for idx, (cat, count) in enumerate(sorted(categories.items(), key=lambda x: x[1], reverse=True)):
                with cols[idx % 4]:
                    st.metric(cat, count)
            
            st.markdown("---")
            
            # Environment breakdown
            st.markdown("### 🌐 Incidents by Environment")
            col1, col2, col3, col4 = st.columns(4)
            
            prod = len([t for t in all_tickets if t.environment == "Production"])
            staging = len([t for t in all_tickets if t.environment == "Staging"])
            dev = len([t for t in all_tickets if t.environment == "Development"])
            uat = len([t for t in all_tickets if t.environment == "UAT"])
            
            with col1:
                st.metric("Production", prod, delta=None, delta_color="inverse")
            with col2:
                st.metric("Staging", staging)
            with col3:
                st.metric("Development", dev)
            with col4:
                st.metric("UAT", uat)
            
            st.markdown("---")
            
            # Recent activity
            st.markdown("### 🕐 Recent Incidents")
            recent = sorted(all_tickets, key=lambda x: x.timestamp, reverse=True)[:5]
            
            for ticket in recent:
                severity_icons = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                severity_icon = severity_icons.get(ticket.severity, "⚪")
                status_icon = "🟢" if ticket.status == "Resolved" else "🔴"
                
                st.markdown(f"{status_icon} {severity_icon} **{ticket.ticket_id}** - {ticket.title} ({ticket.application}) - {ticket.timestamp}")
            
            st.markdown("---")
            
            # Knowledge base stats
            st.markdown("### 📚 Knowledge Base (Read-Only)")
            st.info("💡 Vector DB contains only pre-loaded sample tickets for similarity matching")
            
            # Count sample tickets in vector store
            sample_kb_count = 0
            if os.path.exists("tickets_sample.json"):
                try:
                    with open("tickets_sample.json", 'r', encoding='utf-8') as f:
                        sample_data = json.load(f)
                        sample_kb_count = len([t for t in sample_data if t.get("status") == "Resolved"])
                except:
                    pass
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sample Incidents in Vector DB", sample_kb_count)
            with col2:
                st.metric("User Reported Incidents", len(ticket_manager.get_all_tickets()))
            
        else:
            st.info("ℹ️ No incident data available yet. Start by reporting incidents.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**User Incidents (tickets.json):** {len(ticket_manager.get_all_tickets())}")
    
    # Show vector store status
    if st.session_state.vector_store:
        sample_count = 0
        if os.path.exists("tickets_sample.json"):
            try:
                with open("tickets_sample.json", 'r', encoding='utf-8') as f:
                    sample_data = json.load(f)
                    sample_count = len([t for t in sample_data if t.get("status") == "Resolved"])
            except:
                pass
        st.sidebar.markdown(f"**Knowledge Base (Read-Only):** {sample_count} sample incidents")
    else:
        st.sidebar.markdown("**Knowledge Base:** Not loaded")
    
    # Quick stats in sidebar
    all_incidents = ticket_manager.get_all_tickets()
    if all_incidents:
        critical_count = len([t for t in all_incidents if t.severity == "Critical"])
        open_count = len([t for t in all_incidents if t.status == "Open"])
        
        if critical_count > 0:
            st.sidebar.error(f"🔴 {critical_count} Critical incident(s)")
        if open_count > 0:
            st.sidebar.warning(f"⚠️ {open_count} Open incident(s)")


# ===================== RUN APP =====================
if __name__ == "__main__":
    main()
