"""
Embedding Database utilities for RAG (Retrieval Augmented Generation).
Supports storing and retrieving CV/JD embeddings using ChromaDB.
"""

import os
from typing import List, Dict, Optional
import hashlib
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class CVEmbeddingDB:
    """
    Manage embeddings for CVs and Job Descriptions using ChromaDB.
    Enables RAG by retrieving similar CVs/JDs for context.
    """
    
    def __init__(self, db_path: str = "./chroma_db"):
        """
        Initialize embedding database.
        
        Args:
            db_path: Path to store ChromaDB data
        """
        self.db_path = db_path
        self.client = None
        self.collection = None
        
        if not CHROMADB_AVAILABLE:
            print("Warning: ChromaDB not installed. Install with: pip install chromadb")
            return
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="cv_jd_collection",
                metadata={"description": "Collection of CVs and Job Descriptions"}
            )
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
    
    def _generate_id(self, text: str) -> str:
        """
        Generate unique ID for a document.
        
        Args:
            text: Document text
            
        Returns:
            Unique hash ID
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Split text into chunks for better embedding.
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        if LANGCHAIN_AVAILABLE:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
        else:
            # Simple chunking fallback
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size // 5):  # Rough word estimate
                chunk = ' '.join(words[i:i + chunk_size // 5])
                chunks.append(chunk)
        
        return chunks if chunks else [text]
    
    def add_cv(self, cv_text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add CV to the database.
        
        Args:
            cv_text: CV text content
            metadata: Additional metadata (e.g., candidate name, date)
            
        Returns:
            True if successful
        """
        if not self.collection:
            return False
        
        try:
            doc_id = self._generate_id(cv_text)
            
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                'type': 'cv',
                'added_date': datetime.now().isoformat(),
                'length': len(cv_text)
            })
            
            # Add to collection
            self.collection.add(
                documents=[cv_text],
                metadatas=[meta],
                ids=[doc_id]
            )
            
            return True
        except Exception as e:
            print(f"Error adding CV: {e}")
            return False
    
    def add_jd(self, jd_text: str, metadata: Optional[Dict] = None) -> bool:
        """
        Add Job Description to the database.
        
        Args:
            jd_text: JD text content
            metadata: Additional metadata (e.g., job title, company)
            
        Returns:
            True if successful
        """
        if not self.collection:
            return False
        
        try:
            doc_id = self._generate_id(jd_text)
            
            # Prepare metadata
            meta = metadata or {}
            meta.update({
                'type': 'jd',
                'added_date': datetime.now().isoformat(),
                'length': len(jd_text)
            })
            
            # Add to collection
            self.collection.add(
                documents=[jd_text],
                metadatas=[meta],
                ids=[doc_id]
            )
            
            return True
        except Exception as e:
            print(f"Error adding JD: {e}")
            return False
    
    def search_similar_cvs(self, query_text: str, n_results: int = 3) -> Dict:
        """
        Search for similar CVs in the database.
        
        Args:
            query_text: Query text (CV or JD)
            n_results: Number of results to return
            
        Returns:
            Dictionary with similar CVs and their metadata
        """
        if not self.collection:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"type": "cv"}
            )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            print(f"Error searching CVs: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def search_similar_jds(self, query_text: str, n_results: int = 3) -> Dict:
        """
        Search for similar Job Descriptions in the database.
        
        Args:
            query_text: Query text (CV or JD)
            n_results: Number of results to return
            
        Returns:
            Dictionary with similar JDs and their metadata
        """
        if not self.collection:
            return {'documents': [], 'metadatas': [], 'distances': []}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"type": "jd"}
            )
            
            return {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else []
            }
        except Exception as e:
            print(f"Error searching JDs: {e}")
            return {'documents': [], 'metadatas': [], 'distances': []}
    
    def get_rag_context(self, cv_text: str, jd_text: str, n_results: int = 2) -> str:
        """
        Get relevant context from database for RAG.
        
        Args:
            cv_text: Current CV text
            jd_text: Current JD text
            n_results: Number of similar documents to retrieve
            
        Returns:
            Context string for LLM
        """
        if not self.collection:
            return ""
        
        context_parts = []
        
        # Find similar CVs based on JD
        similar_cvs = self.search_similar_cvs(jd_text, n_results)
        if similar_cvs['documents']:
            context_parts.append("Similar CVs that matched related job descriptions:")
            for i, (doc, meta) in enumerate(zip(similar_cvs['documents'], similar_cvs['metadatas'])):
                context_parts.append(f"\nExample CV {i+1}: {doc[:300]}...")
        
        # Find similar JDs based on CV
        similar_jds = self.search_similar_jds(cv_text, n_results)
        if similar_jds['documents']:
            context_parts.append("\n\nSimilar job descriptions:")
            for i, (doc, meta) in enumerate(zip(similar_jds['documents'], similar_jds['metadatas'])):
                context_parts.append(f"\nExample JD {i+1}: {doc[:300]}...")
        
        return "\n".join(context_parts)
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection stats
        """
        if not self.collection:
            return {'total_documents': 0, 'cvs': 0, 'jds': 0}
        
        try:
            all_items = self.collection.get()
            metadatas = all_items['metadatas']
            
            cvs = sum(1 for m in metadatas if m.get('type') == 'cv')
            jds = sum(1 for m in metadatas if m.get('type') == 'jd')
            
            return {
                'total_documents': len(metadatas),
                'cvs': cvs,
                'jds': jds
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total_documents': 0, 'cvs': 0, 'jds': 0}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        if not self.collection:
            return False
        
        try:
            # Delete and recreate collection
            self.client.delete_collection("cv_jd_collection")
            self.collection = self.client.get_or_create_collection(
                name="cv_jd_collection",
                metadata={"description": "Collection of CVs and Job Descriptions"}
            )
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
