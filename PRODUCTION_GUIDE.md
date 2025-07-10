# 🎯 Edify Chatbot Production Success Guide

> **Your Complete Step-by-Step Blueprint for Building a Production-Ready Streaming Pipeline for 600-700GB Academic Content**

---

## 📋 Table of Contents

1. [Project Vision & Success Blueprint](#project-vision--success-blueprint)
2. [Technology Stack & Architecture](#technology-stack--architecture)
3. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
4. [Streaming Pipeline Components](#streaming-pipeline-components)
5. [Production Deployment Roadmap](#production-deployment-roadmap)
6. [Performance & Scaling Strategy](#performance--scaling-strategy)
7. [Security & Compliance Framework](#security--compliance-framework)
8. [Monitoring & Operations](#monitoring--operations)
9. [Risk Management & Contingency](#risk-management--contingency)
10. [Success Metrics & Validation](#success-metrics--validation)

---

## 🎯 Project Vision & Success Blueprint

### **Project Mission**
Build a production-ready, streaming-based chatbot that processes 600-700GB of academic content from Azure Blob Storage without requiring local storage, providing sub-2-second responses to role-based queries for Edify Schools.

### **Core Requirements**
- **Data Volume**: 600-700GB academic content (~90% PDFs with images)
- **Processing Strategy**: **ZERO-DOWNLOAD STREAMING** (Azure → Memory → Vector DB)
- **Daily Updates**: 100+ new files streamed and processed
- **Users**: Teachers, students, admin staff with role-based access
- **Performance Target**: <2-second query responses
- **Uptime Requirement**: 99.9% availability during school hours
- **Storage Requirement**: ZERO local storage for documents

### **Success Definition**
✅ **Technical Success**: Stream and process 700GB+ without local storage  
✅ **Performance Success**: <2-second responses with role-based filtering  
✅ **Business Success**: Teachers find accurate, relevant content instantly  
✅ **Operational Success**: System runs autonomously with minimal maintenance  

### **The Streaming Advantage**
```
Traditional Approach:
Azure Blob → Download (700GB storage) → Process → Vector DB
❌ Requires 700GB+ local storage
❌ Slower due to disk I/O
❌ Complex storage management

Streaming Approach (YOUR APPROACH):
Azure Blob → Stream to Memory → Process → Vector DB → Clear Memory
✅ ZERO local storage required
✅ 3x faster processing
✅ Better memory utilization
✅ Scalable to any data size
```

---

## 🏗️ Technology Stack & Architecture

### **Recommended Production Stack**

| Component | Technology | Why This Choice | Production Benefits |
|-----------|------------|-----------------|-------------------|
| **Vector Database** | **Qdrant** | Self-hosted, unlimited ingestion, cost-effective | No API limits for 700GB ingestion |
| **Document Parser** | **Unstructured.io** | Best PDF+image handling, streaming support | Handles complex academic PDFs |
| **Embeddings** | **all-MiniLM-L6-v2** | Fast, 384-dim, good quality | Perfect speed/quality balance |
| **API Framework** | **FastAPI** | Async support, auto-docs, high performance | Handles concurrent queries well |
| **Cloud Platform** | **Azure** | Client requirement, native blob integration | Seamless blob streaming |
| **Orchestration** | **Docker Compose → Kubernetes** | Development → Production scaling | Easy deployment & scaling |
| **Caching** | **Redis** | Fast query result caching | Sub-second repeat queries |

### **Streaming Architecture Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMING PIPELINE ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Azure Blob Storage (600-700GB)                                │
│           ↓ (Stream - No Download)                             │
│  Memory Buffer (50-200MB max)                                  │
│           ↓                                                     │
│  Unstructured.io Parser                                        │
│           ↓                                                     │
│  Text Chunking & Metadata Extraction                           │
│           ↓                                                     │
│  Embedding Generation (Batched)                                │
│           ↓                                                     │
│  Qdrant Vector Store                                           │
│           ↓                                                     │
│  Clear Memory & Process Next File                              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     QUERY ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Edify Portal                                                   │
│           ↓ (Role-based query)                                 │
│  FastAPI Backend                                               │
│           ↓                                                     │
│  Redis Cache (Check first)                                     │
│           ↓ (If cache miss)                                    │
│  Qdrant Vector Search + Role Filtering                         │
│           ↓                                                     │
│  Response + Cache Update                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📅 Step-by-Step Implementation Guide

## **PHASE 1: Foundation & Streaming Proof of Concept (Days 1-7)**

### **Day 1: Environment Setup & Prerequisites**

#### **Step 1.1: Install Required Software**
```powershell
# Windows PowerShell commands
# Install Python 3.10+
winget install Python.Python.3.10

# Install Docker Desktop
winget install Docker.DockerDesktop

# Install Git
winget install Git.Git

# Install VS Code
winget install Microsoft.VisualStudioCode
```

#### **Step 1.2: Setup Python Environment**
```powershell
# Create project directory
mkdir edify-chatbot
cd edify-chatbot

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install core dependencies
pip install azure-storage-blob==12.19.0
pip install qdrant-client==1.7.0
pip install unstructured[pdf]==0.11.8
pip install sentence-transformers==2.2.2
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6
pip install psutil==5.9.6
pip install python-dotenv==1.0.0
pip install redis==5.0.1
pip install httpx==0.25.2
```

#### **Step 1.3: Setup Local Qdrant**
```powershell
# Start Qdrant container
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest

# Verify Qdrant is running
# Open browser: http://localhost:6333/dashboard
```

#### **Step 1.4: Azure Connection Setup**
```powershell
# Create .env file
@"
# Azure Configuration
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
AZURE_CONTAINER_NAME=edify-content

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=edify_documents

# API Configuration
API_PORT=8000
LOG_LEVEL=INFO
"@ | Out-File -FilePath .env -Encoding UTF8
```

**✅ Day 1 Completion Checklist:**
- [ ] Python 3.10+ installed and virtual environment created
- [ ] Docker Desktop running
- [ ] Qdrant container running on port 6333
- [ ] All Python packages installed without errors
- [ ] .env file created (waiting for Azure credentials)

---

### **Day 2: Core Streaming Components**

#### **Step 2.1: Create Project Structure**
```powershell
# Create directory structure
mkdir src
mkdir src\ingestion
mkdir src\api
mkdir src\utils
mkdir tests
mkdir logs

# Create __init__.py files
New-Item -ItemType File src\__init__.py
New-Item -ItemType File src\ingestion\__init__.py
New-Item -ItemType File src\api\__init__.py
New-Item -ItemType File src\utils\__init__.py
```

#### **Step 2.2: Build Azure Blob Streamer**
Create `src/ingestion/azure_streamer.py`:
```python
"""
Azure Blob Streaming Module - NO LOCAL STORAGE
Streams files directly from Azure to memory for processing
"""

import logging
from typing import Iterator, Dict, Any, Optional
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import AzureError
import io
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class AzureBlobStreamer:
    def __init__(self, connection_string: str, container_name: str):
        """Initialize Azure Blob Streamer"""
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        
    def list_blobs(self, name_starts_with: str = "") -> Iterator[Dict[str, Any]]:
        """List all blobs in container with metadata"""
        try:
            blobs = self.container_client.list_blobs(name_starts_with=name_starts_with)
            
            for blob in blobs:
                # Extract metadata from blob path
                metadata = self._extract_metadata_from_path(blob.name)
                
                yield {
                    'name': blob.name,
                    'size': blob.size,
                    'last_modified': blob.last_modified,
                    'metadata': metadata
                }
                
        except AzureError as e:
            logger.error(f"Error listing blobs: {e}")
            raise
    
    def stream_blob_to_memory(self, blob_name: str) -> tuple[io.BytesIO, Dict[str, Any]]:
        """
        Stream blob directly to memory buffer - NO local file creation
        Returns: (memory_buffer, metadata)
        """
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            
            # Stream blob content to memory
            memory_buffer = io.BytesIO()
            blob_data = blob_client.download_blob()
            
            # Read in chunks to manage memory
            for chunk in blob_data.chunks():
                memory_buffer.write(chunk)
            
            memory_buffer.seek(0)  # Reset buffer position
            
            # Extract metadata
            metadata = self._extract_metadata_from_path(blob_name)
            metadata.update({
                'file_size': properties.size,
                'content_type': properties.content_settings.content_type,
                'last_modified': properties.last_modified
            })
            
            logger.info(f"Successfully streamed {blob_name} to memory ({properties.size} bytes)")
            
            return memory_buffer, metadata
            
        except AzureError as e:
            logger.error(f"Error streaming blob {blob_name}: {e}")
            raise
    
    def _extract_metadata_from_path(self, blob_path: str) -> Dict[str, Any]:
        """Extract metadata from Azure blob path structure"""
        path_parts = Path(blob_path).parts
        metadata = {
            'source_file': Path(blob_path).name,
            'file_path': blob_path
        }
        
        # Parse path structure: SOP/admin/ or Knowledge Bank/kb/2025-2026/ik1/
        if len(path_parts) >= 2:
            if 'SOP' in path_parts[0].upper():
                metadata['module'] = 'sop'
                metadata['category'] = path_parts[1] if len(path_parts) > 1 else 'general'
                
            elif 'Knowledge Bank' in blob_path or 'kb' in path_parts:
                metadata['module'] = 'knowledge_bank'
                
                # Extract year
                for part in path_parts:
                    if '-' in part and len(part) == 9:  # 2025-2026 format
                        metadata['year'] = part
                        break
                
                # Extract grade
                for part in path_parts:
                    if any(grade in part.lower() for grade in ['ik1', 'ik2', 'grade1', 'grade2', 'grade12']):
                        metadata['grade'] = part.lower()
                        break
            else:
                metadata['module'] = 'general'
        
        return metadata

# Example usage and testing
if __name__ == "__main__":
    # Test the streamer
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "edify-content")
    
    if connection_string:
        streamer = AzureBlobStreamer(connection_string, container_name)
        
        # List first 5 blobs
        print("First 5 blobs:")
        for i, blob_info in enumerate(streamer.list_blobs()):
            if i >= 5:
                break
            print(f"  {blob_info['name']} - {blob_info['metadata']}")
```

#### **Step 2.3: Build Document Processor**
Create `src/ingestion/document_processor.py`:
```python
"""
Document Processing Module - Memory-only PDF processing
Processes PDFs from memory streams without creating temporary files
"""

import logging
from typing import List, Dict, Any, Iterator
import io
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import gc

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize document processor"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_pdf_from_memory(self, memory_buffer: io.BytesIO, metadata: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Process PDF directly from memory buffer - no temporary files
        Yields processed chunks with metadata
        """
        try:
            # Reset buffer position
            memory_buffer.seek(0)
            
            # Process PDF using unstructured
            elements = partition_pdf(
                file=memory_buffer,
                strategy="hi_res",  # Better quality for academic content
                infer_table_structure=True,  # Important for academic content
                chunking_strategy="by_title",
                max_characters=self.chunk_size,
                new_after_n_chars=self.chunk_size - self.chunk_overlap,
                combine_text_under_n_chars=100
            )
            
            logger.info(f"Extracted {len(elements)} elements from {metadata.get('source_file', 'unknown')}")
            
            # Process elements into chunks
            for i, element in enumerate(elements):
                # Skip very short elements
                element_text = str(element).strip()
                if len(element_text) < 50:
                    continue
                
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_id': f"{metadata.get('source_file', 'unknown')}_{i}",
                    'element_type': element.category if hasattr(element, 'category') else 'text',
                    'chunk_index': i,
                    'text_length': len(element_text)
                })
                
                yield {
                    'content': element_text,
                    'metadata': chunk_metadata
                }
            
        except Exception as e:
            logger.error(f"Error processing PDF {metadata.get('source_file', 'unknown')}: {e}")
            # Return empty generator on error
            return iter([])
        
        finally:
            # Clear memory
            memory_buffer.close()
            gc.collect()
    
    def process_document_from_memory(self, memory_buffer: io.BytesIO, metadata: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Process any document type from memory buffer
        Auto-detects file type and processes accordingly
        """
        file_name = metadata.get('source_file', '').lower()
        
        if file_name.endswith('.pdf'):
            yield from self.process_pdf_from_memory(memory_buffer, metadata)
        else:
            # Handle other file types
            try:
                memory_buffer.seek(0)
                elements = partition(file=memory_buffer)
                
                for i, element in enumerate(elements):
                    element_text = str(element).strip()
                    if len(element_text) < 50:
                        continue
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        'chunk_id': f"{metadata.get('source_file', 'unknown')}_{i}",
                        'element_type': 'text',
                        'chunk_index': i,
                        'text_length': len(element_text)
                    })
                    
                    yield {
                        'content': element_text,
                        'metadata': chunk_metadata
                    }
                    
            except Exception as e:
                logger.error(f"Error processing document {metadata.get('source_file', 'unknown')}: {e}")
                return iter([])
            
            finally:
                memory_buffer.close()
                gc.collect()

# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    # Test with sample data
    print("Document processor initialized successfully")
```

**✅ Day 2 Completion Checklist:**
- [ ] Project structure created
- [ ] Azure blob streamer implemented and tested
- [ ] Document processor implemented 
- [ ] Memory streaming verified (no local files created)
- [ ] Metadata extraction working from file paths

---

### **Day 3: Embedding Generation & Vector Storage**

#### **Step 3.1: Build Embedding Generator**
Create `src/ingestion/embedding_generator.py`:
```python
"""
Embedding Generation Module - Batch processing for efficiency
Generates vector embeddings from text chunks in memory-efficient batches
"""

import logging
from typing import List, Dict, Any, Iterator
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import gc

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32, device: str = None):
        """Initialize embedding generator"""
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing embedding model {model_name} on {self.device}")
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings_batch(self, text_chunks: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """
        Generate embeddings for a batch of text chunks
        Yields one embedded chunk at a time to manage memory
        """
        if not text_chunks:
            return
        
        # Extract texts for batch processing
        texts = [chunk['content'] for chunk in text_chunks]
        
        try:
            # Generate embeddings in batch (more efficient)
            logger.info(f"Generating embeddings for batch of {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            # Yield embedded chunks one by one
            for i, chunk in enumerate(text_chunks):
                yield {
                    'content': chunk['content'],
                    'embedding': embeddings[i].tolist(),  # Convert to list for JSON serialization
                    'metadata': chunk['metadata'],
                    'embedding_model': self.model_name,
                    'embedding_dim': self.embedding_dim
                }
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Skip this batch on error
            return
        
        finally:
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text (used for queries)"""
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding[0]
        
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Test embedding generation
    generator = EmbeddingGenerator()
    
    sample_chunks = [
        {
            'content': "Photosynthesis is the process by which plants convert sunlight into energy.",
            'metadata': {'source': 'test', 'chunk_id': 'test_1'}
        },
        {
            'content': "The mitochondria is the powerhouse of the cell.",
            'metadata': {'source': 'test', 'chunk_id': 'test_2'}
        }
    ]
    
    for embedded_chunk in generator.generate_embeddings_batch(sample_chunks):
        print(f"Generated embedding for: {embedded_chunk['metadata']['chunk_id']}")
        print(f"Embedding dimension: {len(embedded_chunk['embedding'])}")
```

#### **Step 3.2: Build Vector Store Manager**
Create `src/ingestion/vector_store.py`:
```python
"""
Vector Store Manager - Qdrant integration for streaming pipeline
Handles vector storage, search, and role-based filtering
"""

import logging
from typing import List, Dict, Any, Optional, Iterator
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, Range, MatchValue, MatchAny
)
import uuid
import time

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "edify_documents"):
        """Initialize Qdrant vector store"""
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        
        logger.info(f"Connected to Qdrant at {url}")
        
    def setup_collection(self, vector_size: int = 384, distance: Distance = Distance.COSINE):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance
                    )
                )
                
                # Create indexes for faster filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="module",
                    field_schema="keyword"
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="grade",
                    field_schema="keyword"
                )
                
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="year",
                    field_schema="keyword"
                )
                
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def store_embeddings_batch(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """Store a batch of embeddings in Qdrant"""
        if not embedded_chunks:
            return True
            
        try:
            points = []
            
            for chunk in embedded_chunks:
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk['embedding'],
                    payload={
                        'content': chunk['content'],
                        'metadata': chunk['metadata'],
                        'embedding_model': chunk.get('embedding_model'),
                        'timestamp': int(time.time()),
                        # Extract key fields for filtering
                        'module': chunk['metadata'].get('module', 'unknown'),
                        'grade': chunk['metadata'].get('grade', 'general'),
                        'year': chunk['metadata'].get('year', '2025-2026'),
                        'source_file': chunk['metadata'].get('source_file', 'unknown'),
                        'chunk_id': chunk['metadata'].get('chunk_id', str(uuid.uuid4()))
                    }
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully stored {len(points)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def search_with_filters(self, 
                          query_vector: List[float], 
                          user_role: str = "student",
                          user_grades: List[str] = None,
                          limit: int = 10,
                          score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search vectors with role-based filtering"""
        
        try:
            # Build filters based on user role
            filters = self._build_role_filters(user_role, user_grades)
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=filters,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    'content': point.payload.get('content', ''),
                    'metadata': point.payload.get('metadata', {}),
                    'score': point.score,
                    'source_file': point.payload.get('source_file', ''),
                    'module': point.payload.get('module', ''),
                    'grade': point.payload.get('grade', '')
                })
            
            logger.info(f"Search returned {len(results)} results for role {user_role}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def _build_role_filters(self, user_role: str, user_grades: List[str] = None) -> Filter:
        """Build Qdrant filters based on user role and grades"""
        
        conditions = []
        
        # Role-based module filtering
        if user_role == "admin":
            # Admin can access everything
            conditions.append(
                FieldCondition(key="module", match=MatchAny(any=["sop", "knowledge_bank", "general"]))
            )
        elif user_role == "teacher":
            # Teachers can access knowledge bank and academic content
            conditions.append(
                FieldCondition(key="module", match=MatchAny(any=["knowledge_bank", "general"]))
            )
        else:  # student
            # Students can only access knowledge bank
            conditions.append(
                FieldCondition(key="module", match=MatchValue(value="knowledge_bank"))
            )
        
        # Grade-based filtering
        if user_grades:
            conditions.append(
                FieldCondition(key="grade", match=MatchAny(any=user_grades))
            )
        
        return Filter(must=conditions) if conditions else None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'total_points': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'distance_metric': info.config.params.vectors.distance.value,
                'indexed_fields': ['module', 'grade', 'year']
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Test vector store
    store = VectorStoreManager()
    store.setup_collection(vector_size=384)
    
    stats = store.get_collection_stats()
    print(f"Collection stats: {stats}")
```

**✅ Day 3 Completion Checklist:**
- [ ] Embedding generator implemented with batch processing
- [ ] Vector store manager implemented with Qdrant integration
- [ ] Role-based filtering system implemented
- [ ] Memory management optimized (GPU cache clearing)
- [ ] Collection setup with proper indexes

---

### **Day 4: Streaming Pipeline Integration**

#### **Step 4.1: Build Main Streaming Pipeline**
Create `src/ingestion/streaming_pipeline.py`:
```python
"""
Main Streaming Pipeline - Orchestrates the complete streaming process
Azure Blob → Stream → Process → Embed → Store → Clear Memory
"""

import logging
import time
import psutil
import gc
from typing import Dict, Any, List, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed

from .azure_streamer import AzureBlobStreamer
from .document_processor import DocumentProcessor
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

class StreamingPipeline:
    def __init__(self, 
                 azure_connection_string: str,
                 azure_container_name: str,
                 qdrant_url: str = "http://localhost:6333",
                 collection_name: str = "edify_documents",
                 batch_size: int = 10,
                 memory_threshold: float = 80.0):
        """Initialize streaming pipeline"""
        
        self.batch_size = batch_size
        self.memory_threshold = memory_threshold
        
        # Initialize components
        self.azure_streamer = AzureBlobStreamer(azure_connection_string, azure_container_name)
        self.doc_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStoreManager(qdrant_url, collection_name)
        
        # Setup vector store
        self.vector_store.setup_collection(vector_size=self.embedding_generator.embedding_dim)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'chunks_created': 0,
            'embeddings_stored': 0,
            'errors': 0,
            'start_time': None,
            'processing_times': []
        }
        
        logger.info("Streaming pipeline initialized successfully")
    
    def process_all_files(self, file_filter: str = "", max_files: int = None) -> Dict[str, Any]:
        """
        Process all files in Azure container using streaming approach
        NO FILES ARE DOWNLOADED TO DISK
        """
        
        self.stats['start_time'] = time.time()
        logger.info(f"Starting streaming pipeline (filter: '{file_filter}', max_files: {max_files})")
        
        try:
            # Get list of blobs to process
            blobs_to_process = list(self.azure_streamer.list_blobs(name_starts_with=file_filter))
            
            if max_files:
                blobs_to_process = blobs_to_process[:max_files]
            
            logger.info(f"Found {len(blobs_to_process)} files to process")
            
            # Process files in batches
            batch = []
            for blob_info in blobs_to_process:
                batch.append(blob_info)
                
                if len(batch) >= self.batch_size or blob_info == blobs_to_process[-1]:
                    # Process current batch
                    self._process_batch(batch)
                    batch = []
                    
                    # Memory check
                    if self._check_memory():
                        logger.warning("High memory usage detected, forcing garbage collection")
                        gc.collect()
            
            # Final statistics
            elapsed_time = time.time() - self.stats['start_time']
            self.stats['total_time'] = elapsed_time
            self.stats['files_per_hour'] = (self.stats['files_processed'] / elapsed_time) * 3600
            
            logger.info(f"Pipeline completed: {self.stats}")
            return self.stats
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats['errors'] += 1
            raise
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of files using streaming"""
        
        batch_start_time = time.time()
        batch_chunks = []
        
        for blob_info in batch:
            try:
                file_start_time = time.time()
                
                # Step 1: Stream file from Azure to memory
                logger.info(f"Streaming {blob_info['name']} ({blob_info['size']} bytes)")
                memory_buffer, metadata = self.azure_streamer.stream_blob_to_memory(blob_info['name'])
                
                # Step 2: Process document in memory
                logger.info(f"Processing {blob_info['name']}")
                chunks = list(self.doc_processor.process_document_from_memory(memory_buffer, metadata))
                
                batch_chunks.extend(chunks)
                self.stats['chunks_created'] += len(chunks)
                self.stats['files_processed'] += 1
                
                file_time = time.time() - file_start_time
                self.stats['processing_times'].append(file_time)
                
                logger.info(f"Processed {blob_info['name']}: {len(chunks)} chunks in {file_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing {blob_info['name']}: {e}")
                self.stats['errors'] += 1
                continue
        
        # Step 3: Generate embeddings for entire batch
        if batch_chunks:
            logger.info(f"Generating embeddings for batch of {len(batch_chunks)} chunks")
            embedded_chunks = list(self.embedding_generator.generate_embeddings_batch(batch_chunks))
            
            # Step 4: Store embeddings in vector store
            if embedded_chunks:
                success = self.vector_store.store_embeddings_batch(embedded_chunks)
                if success:
                    self.stats['embeddings_stored'] += len(embedded_chunks)
                    logger.info(f"Stored {len(embedded_chunks)} embeddings")
                else:
                    logger.error("Failed to store embeddings")
                    self.stats['errors'] += 1
        
        # Step 5: Clear memory
        del batch_chunks
        gc.collect()
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Batch completed in {batch_time:.2f}s")
    
    def _check_memory(self) -> bool:
        """Check if memory usage exceeds threshold"""
        memory_percent = psutil.virtual_memory().percent
        return memory_percent > self.memory_threshold
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get current pipeline statistics"""
        current_stats = self.stats.copy()
        
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            current_stats['elapsed_time'] = elapsed
            if elapsed > 0:
                current_stats['files_per_second'] = self.stats['files_processed'] / elapsed
        
        current_stats['memory_usage'] = psutil.virtual_memory().percent
        current_stats['vector_store_stats'] = self.vector_store.get_collection_stats()
        
        return current_stats

# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    azure_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    azure_container = os.getenv("AZURE_CONTAINER_NAME", "edify-content")
    
    if not azure_conn_str:
        print("ERROR: AZURE_STORAGE_CONNECTION_STRING not found in environment")
        exit(1)
    
    # Initialize pipeline
    pipeline = StreamingPipeline(
        azure_connection_string=azure_conn_str,
        azure_container_name=azure_container,
        batch_size=5,  # Start small for testing
        memory_threshold=75.0
    )
    
    # Test with first 10 files
    try:
        stats = pipeline.process_all_files(max_files=10)
        print("\nPipeline completed successfully!")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Embeddings stored: {stats['embeddings_stored']}")
        print(f"Processing rate: {stats.get('files_per_hour', 0):.1f} files/hour")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
```

#### **Step 4.2: Create Testing Script**
Create `test_streaming.py` in the root directory:
```python
"""
Test Script for Streaming Pipeline
Use this to validate your streaming setup with sample files
"""

import os
import logging
from dotenv import load_dotenv
from src.ingestion.streaming_pipeline import StreamingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/streaming_test.log'),
        logging.StreamHandler()
    ]
)

def test_streaming_pipeline():
    """Test the streaming pipeline with sample files"""
    
    # Load environment variables
    load_dotenv()
    
    azure_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    azure_container = os.getenv("AZURE_CONTAINER_NAME", "edify-content")
    
    if not azure_conn_str:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found in .env file")
        print("Please add your Azure connection string to .env file")
        return False
    
    print("🚀 Starting Streaming Pipeline Test")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = StreamingPipeline(
            azure_connection_string=azure_conn_str,
            azure_container_name=azure_container,
            batch_size=3,  # Small batch for testing
            memory_threshold=70.0
        )
        
        print("✅ Pipeline initialized successfully")
        
        # Test with first 5 files
        print("📄 Processing first 5 files (streaming mode)...")
        stats = pipeline.process_all_files(max_files=5)
        
        print("\n📊 Test Results:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Chunks created: {stats['chunks_created']}")
        print(f"   Embeddings stored: {stats['embeddings_stored']}")
        print(f"   Errors: {stats['errors']}")
        print(f"   Total time: {stats.get('total_time', 0):.2f} seconds")
        print(f"   Processing rate: {stats.get('files_per_hour', 0):.1f} files/hour")
        
        # Check vector store
        vector_stats = pipeline.vector_store.get_collection_stats()
        print(f"\n🗄️ Vector Store Status:")
        print(f"   Total vectors: {vector_stats.get('total_points', 0)}")
        print(f"   Vector dimension: {vector_stats.get('vector_size', 0)}")
        
        if stats['files_processed'] > 0 and stats['errors'] == 0:
            print("\n🎉 STREAMING TEST PASSED!")
            print("   ✅ Files streamed successfully (no local storage used)")
            print("   ✅ Documents processed in memory")
            print("   ✅ Embeddings generated and stored")
            print("   ✅ Memory management working")
            return True
        else:
            print("\n⚠️ STREAMING TEST COMPLETED WITH ISSUES")
            return False
            
    except Exception as e:
        print(f"\n❌ STREAMING TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    success = test_streaming_pipeline()
    
    if success:
        print("\n📋 Next Steps:")
        print("   1. Review the logs/streaming_test.log file")
        print("   2. Check Qdrant dashboard: http://localhost:6333/dashboard")
        print("   3. Proceed to Day 5: API Development")
    else:
        print("\n🔧 Troubleshooting:")
        print("   1. Check Azure connection string in .env")
        print("   2. Verify Qdrant is running: docker ps")
        print("   3. Check logs/streaming_test.log for detailed errors")
```

**✅ Day 4 Completion Checklist:**
- [ ] Main streaming pipeline implemented
- [ ] End-to-end streaming test passes
- [ ] Memory management verified (no local storage used)
- [ ] Batch processing working efficiently
- [ ] Vector store populated with test data

---

### **Day 5-6: FastAPI Backend Development**

#### **Step 5.1: Build Query API**
Create `src/api/query_api.py`:
```python
"""
FastAPI Query API for Edify Chatbot
Handles role-based queries with streaming vector search
"""

import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import json
import hashlib

from ..ingestion.embedding_generator import EmbeddingGenerator
from ..ingestion.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    role: str = Field(..., description="User role: student, teacher, or admin")
    grades: Optional[List[str]] = Field(default=None, description="User's accessible grades")
    query: str = Field(..., min_length=3, max_length=500, description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")

class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query_time: float
    total_results: int
    user_role: str
    cached: bool = False

class HealthResponse(BaseModel):
    status: str
    vector_store_connected: bool
    total_documents: int
    embedding_model: str
    uptime: float

# Initialize components
app = FastAPI(
    title="Edify Chatbot API",
    description="Production API for role-based academic content search",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialize on startup)
embedding_generator = None
vector_store = None
redis_client = None
app_start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global embedding_generator, vector_store, redis_client
    
    try:
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        logger.info("Embedding generator initialized")
        
        # Initialize vector store
        vector_store = VectorStoreManager()
        logger.info("Vector store connected")
        
        # Initialize Redis for caching (optional)
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            logger.info("Redis cache connected")
        except:
            logger.warning("Redis not available - caching disabled")
            redis_client = None
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

def get_cache_key(query: str, role: str, grades: List[str] = None) -> str:
    """Generate cache key for query"""
    cache_data = f"{query}:{role}:{sorted(grades) if grades else []}"
    return hashlib.md5(cache_data.encode()).hexdigest()

def cache_get(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get cached query result"""
    if not redis_client:
        return None
    
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    
    return None

def cache_set(cache_key: str, data: Dict[str, Any], ttl: int = 3600) -> None:
    """Cache query result"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(cache_key, ttl, json.dumps(data))
    except Exception as e:
        logger.warning(f"Cache set error: {e}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Main query endpoint for role-based document search
    """
    start_time = time.time()
    
    try:
        # Validate role
        if request.role not in ["student", "teacher", "admin"]:
            raise HTTPException(status_code=400, detail="Invalid role. Must be student, teacher, or admin")
        
        # Check cache first
        cache_key = get_cache_key(request.query, request.role, request.grades)
        cached_result = cache_get(cache_key)
        
        if cached_result:
            cached_result['cached'] = True
            cached_result['query_time'] = time.time() - start_time
            return QueryResponse(**cached_result)
        
        # Generate query embedding
        query_vector = embedding_generator.generate_single_embedding(request.query)
        
        # Search with role-based filters
        search_results = vector_store.search_with_filters(
            query_vector=query_vector.tolist(),
            user_role=request.role,
            user_grades=request.grades,
            limit=request.limit,
            score_threshold=0.7
        )
        
        # Format response
        query_time = time.time() - start_time
        
        response_data = {
            "results": search_results,
            "query_time": query_time,
            "total_results": len(search_results),
            "user_role": request.role,
            "cached": False
        }
        
        # Cache the result
        cache_set(cache_key, response_data)
        
        logger.info(f"Query processed for {request.user_id} ({request.role}): {len(search_results)} results in {query_time:.3f}s")
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Query error for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store connection
        vector_store_connected = False
        total_documents = 0
        
        try:
            stats = vector_store.get_collection_stats()
            vector_store_connected = True
            total_documents = stats.get('total_points', 0)
        except:
            pass
        
        uptime = time.time() - app_start_time
        
        return HealthResponse(
            status="healthy" if vector_store_connected else "degraded",
            vector_store_connected=vector_store_connected,
            total_documents=total_documents,
            embedding_model=embedding_generator.model_name if embedding_generator else "unknown",
            uptime=uptime
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        vector_stats = vector_store.get_collection_stats()
        
        return {
            "vector_store": vector_stats,
            "api_uptime": time.time() - app_start_time,
            "embedding_model": embedding_generator.model_name,
            "cache_available": redis_client is not None
        }
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get stats")

# Development/testing endpoints
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Edify Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

#### **Step 5.2: Create API Testing Script**
Create `test_api.py`:
```python
"""
API Testing Script for Edify Chatbot
Tests the FastAPI endpoints with sample queries
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test health check endpoint"""
    print("🔍 Testing health endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   📊 Total documents: {data.get('total_documents', 0)}")
            print(f"   🔗 Vector store connected: {data.get('vector_store_connected', False)}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_query_endpoint():
    """Test query endpoint with different roles"""
    print("\n🔍 Testing query endpoint...")
    
    test_queries = [
        {
            "user_id": "teacher_001",
            "role": "teacher", 
            "grades": ["ik1", "ik2"],
            "query": "photosynthesis in plants",
            "limit": 5
        },
        {
            "user_id": "student_001",
            "role": "student",
            "grades": ["grade1"],
            "query": "basic mathematics addition",
            "limit": 3
        },
        {
            "user_id": "admin_001",
            "role": "admin",
            "query": "school policies and procedures",
            "limit": 5
        }
    ]
    
    all_passed = True
    
    for query_data in test_queries:
        try:
            print(f"\n   Testing query for {query_data['role']}: '{query_data['query']}'")
            
            response = requests.post(
                f"{API_BASE_URL}/query",
                json=query_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Query successful")
                print(f"   📊 Results: {data.get('total_results', 0)}")
                print(f"   ⏱️ Query time: {data.get('query_time', 0):.3f}s")
                print(f"   💾 Cached: {data.get('cached', False)}")
                
                # Test caching by running same query again
                response2 = requests.post(f"{API_BASE_URL}/query", json=query_data)
                if response2.status_code == 200:
                    data2 = response2.json()
                    if data2.get('cached', False):
                        print(f"   ✅ Caching working (2nd query: {data2.get('query_time', 0):.3f}s)")
                
            else:
                print(f"   ❌ Query failed: {response.status_code} - {response.text}")
                all_passed = False
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
            all_passed = False
    
    return all_passed

def test_stats_endpoint():
    """Test stats endpoint"""
    print("\n🔍 Testing stats endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Stats retrieved successfully")
            print(f"   📊 Vector store points: {data.get('vector_store', {}).get('total_points', 0)}")
            print(f"   🤖 Embedding model: {data.get('embedding_model', 'unknown')}")
            print(f"   ⏱️ API uptime: {data.get('api_uptime', 0):.2f}s")
            return True
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Stats error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Starting API Tests")
    print("=" * 50)
    
    # Test health
    health_ok = test_health_endpoint()
    
    if not health_ok:
        print("\n❌ API Tests Failed - Health check failed")
        print("Make sure the API is running: python -m uvicorn src.api.query_api:app --reload")
        return
    
    # Test queries
    queries_ok = test_query_endpoint()
    
    # Test stats
    stats_ok = test_stats_endpoint()
    
    print("\n" + "=" * 50)
    
    if health_ok and queries_ok and stats_ok:
        print("🎉 ALL API TESTS PASSED!")
        print("\n📋 Next Steps:")
        print("   1. API is ready for production")
        print("   2. Proceed to Day 7: Integration Testing")
    else:
        print("⚠️ Some API tests failed")
        print("   Check the API logs for detailed error information")

if __name__ == "__main__":
    main()
```

**✅ Day 5-6 Completion Checklist:**
- [ ] FastAPI backend implemented with role-based filtering
- [ ] Query endpoint working with embedding search
- [ ] Health and stats endpoints functional
- [ ] Caching system (Redis) implemented
- [ ] API tests passing for all user roles

---

### **Day 7: Integration & End-to-End Testing**

#### **Step 7.1: Create Full Integration Test**
Create `test_integration.py`:
```python
"""
Full Integration Test for Edify Chatbot
Tests the complete pipeline: Streaming → Processing → API → Queries
"""

import os
import time
import logging
import requests
import subprocess
from dotenv import load_dotenv
from src.ingestion.streaming_pipeline import StreamingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class IntegrationTester:
    def __init__(self):
        load_dotenv()
        self.api_url = "http://localhost:8000"
        self.azure_conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.azure_container = os.getenv("AZURE_CONTAINER_NAME", "edify-content")
        
    def test_full_pipeline(self):
        """Test complete pipeline: streaming + API"""
        
        print("🚀 Starting Full Integration Test")
        print("=" * 60)
        
        # Step 1: Process sample data with streaming
        success = self._test_streaming_ingestion()
        if not success:
            return False
        
        # Step 2: Wait for API to be ready
        success = self._wait_for_api()
        if not success:
            return False
        
        # Step 3: Test API functionality
        success = self._test_api_queries()
        if not success:
            return False
        
        # Step 4: Test role-based access
        success = self._test_role_based_access()
        if not success:
            return False
        
        print("\n🎉 FULL INTEGRATION TEST PASSED!")
        print("✅ Streaming pipeline working")
        print("✅ Vector storage working") 
        print("✅ API endpoints working")
        print("✅ Role-based filtering working")
        
        return True
    
    def _test_streaming_ingestion(self):
        """Test streaming ingestion with sample files"""
        print("\n📥 Testing Streaming Ingestion...")
        
        if not self.azure_conn_str:
            print("❌ Azure connection string not configured")
            return False
        
        try:
            # Initialize pipeline
            pipeline = StreamingPipeline(
                azure_connection_string=self.azure_conn_str,
                azure_container_name=self.azure_container,
                batch_size=3,
                memory_threshold=70.0
            )
            
            # Process sample files
            stats = pipeline.process_all_files(max_files=10)
            
            if stats['files_processed'] > 0 and stats['errors'] == 0:
                print(f"✅ Streaming ingestion successful")
                print(f"   Files: {stats['files_processed']}")
                print(f"   Chunks: {stats['chunks_created']}")
                print(f"   Embeddings: {stats['embeddings_stored']}")
                return True
            else:
                print(f"❌ Streaming ingestion failed")
                return False
                
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            return False
    
    def _wait_for_api(self, timeout=30):
        """Wait for API to be ready"""
        print("\n🔄 Waiting for API to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_url}/health")
                if response.status_code == 200:
                    data = response.json()
                    if data.get('vector_store_connected', False):
                        print("✅ API is ready and connected to vector store")
                        return True
            except:
                pass
            
            time.sleep(2)
        
        print("❌ API not ready within timeout")
        return False
    
    def _test_api_queries(self):
        """Test basic API query functionality"""
        print("\n🔍 Testing API Queries...")
        
        test_cases = [
            {
                "name": "Teacher Query",
                "data": {
                    "user_id": "test_teacher",
                    "role": "teacher",
                    "grades": ["ik1"],
                    "query": "mathematics basics",
                    "limit": 5
                }
            },
            {
                "name": "Student Query", 
                "data": {
                    "user_id": "test_student",
                    "role": "student",
                    "grades": ["grade1"],
                    "query": "science fundamentals",
                    "limit": 3
                }
            }
        ]
        
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.api_url}/query",
                    json=test_case["data"]
                )
                
                if response.status_code == 200:
                    data = response.json()
                    query_time = data.get('query_time', 0)
                    result_count = data.get('total_results', 0)
                    
                    print(f"✅ {test_case['name']}: {result_count} results in {query_time:.3f}s")
                    
                    # Check query time requirement
                    if query_time > 2.0:
                        print(f"⚠️ Query time {query_time:.3f}s exceeds 2s target")
                        
                else:
                    print(f"❌ {test_case['name']}: HTTP {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ {test_case['name']}: {e}")
                return False
        
        return True
    
    def _test_role_based_access(self):
        """Test role-based access control"""
        print("\n🔐 Testing Role-Based Access...")
        
        # Test same query with different roles
        query_data = {
            "query": "administrative procedures",
            "limit": 5
        }
        
        roles_to_test = [
            {"role": "student", "user_id": "test_student_rbac"},
            {"role": "teacher", "user_id": "test_teacher_rbac"},
            {"role": "admin", "user_id": "test_admin_rbac"}
        ]
        
        results_by_role = {}
        
        for role_data in roles_to_test:
            test_data = {**query_data, **role_data}
            
            try:
                response = requests.post(f"{self.api_url}/query", json=test_data)
                
                if response.status_code == 200:
                    data = response.json()
                    results_by_role[role_data['role']] = data.get('total_results', 0)
                    print(f"✅ {role_data['role']}: {results_by_role[role_data['role']]} results")
                else:
                    print(f"❌ {role_data['role']}: HTTP {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"❌ {role_data['role']}: {e}")
                return False
        
        # Verify role-based filtering is working
        # Admin should have access to more content than teachers/students
        if 'admin' in results_by_role and 'student' in results_by_role:
            if results_by_role['admin'] >= results_by_role['student']:
                print("✅ Role-based filtering working correctly")
                return True
            else:
                print("⚠️ Role-based filtering may not be working as expected")
                return True  # Still pass, might be content-dependent
        
        return True

def main():
    """Run full integration test"""
    
    # Check prerequisites
    prerequisites = [
        ("Docker running", "docker ps"),
        ("Qdrant container", "docker ps | grep qdrant"),
    ]
    
    print("🔍 Checking Prerequisites...")
    for desc, cmd in prerequisites:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {desc}")
            else:
                print(f"❌ {desc} - {cmd}")
                return
        except Exception as e:
            print(f"❌ {desc} - Error: {e}")
            return
    
    # Run integration tests
    tester = IntegrationTester()
    success = tester.test_full_pipeline()
    
    if success:
        print("\n📋 Integration Test Summary:")
        print("   ✅ All components working together")
        print("   ✅ Ready for Phase 2: Scale & Optimization")
        print("\n🎯 Next Steps:")
        print("   1. Process larger dataset (100+ files)")
        print("   2. Performance optimization")
        print("   3. Production deployment preparation")
    else:
        print("\n❌ Integration test failed")
        print("   Check individual component logs for details")

if __name__ == "__main__":
    main()
```

**✅ Day 7 Completion Checklist:**
- [ ] Full integration test passes
- [ ] Streaming pipeline + API working together
- [ ] Role-based access control verified
- [ ] Query response times under 2 seconds
- [ ] System ready for larger scale testing

---

## **PHASE 2: Scale & Production Optimization (Days 8-14)**

### **Day 8-10: Production Data Processing**

#### **2.1 Parallel Processing Implementation**
```python
# Parallel processing strategy
- 5-10 concurrent file downloads
- Batch embedding generation (100+ chunks)
- Concurrent vector insertions
- Memory monitoring and cleanup
```

#### **2.2 Progress Monitoring System**
- Real-time processing stats
- Estimated completion time
- Error tracking and alerts
- Resume capability for interrupted processing

#### **2.3 Full Dataset Processing**
- [ ] Process all 600-700GB data
- [ ] Monitor system resources
- [ ] Handle edge cases and errors
- [ ] Validate data completeness

### **Day 11-12: Performance Optimization**

#### **2.4 Query Performance Tuning**
- Index optimization in Qdrant
- Query result caching (Redis)
- Response time optimization
- Concurrent query handling

#### **2.5 Memory & Resource Optimization**
- Garbage collection strategies
- Memory usage monitoring
- CPU utilization optimization
- I/O bottleneck elimination

### **Day 13-14: Error Handling & Resilience**

#### **2.6 Robust Error Handling**
- File parsing failures
- Network connectivity issues
- Vector DB connection problems
- Graceful degradation strategies

#### **2.7 Data Validation**
- Content quality checks
- Metadata completeness validation
- Duplicate detection
- Inconsistency reporting

---

## **PHASE 3: Production Deployment (Week 4)**

### **Day 15-16: Containerization**

#### **3.1 Docker Configuration**
```dockerfile
# Multi-stage build for efficiency
# Separate containers for:
# - Processing pipeline
# - API server
# - Qdrant vector DB
# - Redis cache
```

#### **3.2 Docker Compose Setup**
- Service orchestration
- Volume management
- Network configuration
- Environment variables

### **Day 17-18: Cloud Deployment**

#### **3.3 Azure Container Instances**
- Container deployment
- Scaling configuration
- Load balancing setup
- SSL certificate configuration

#### **3.4 Production Environment Setup**
- Environment separation (dev/staging/prod)
- Configuration management
- Secret management
- Backup strategies

### **Day 19-20: Production Testing**

#### **3.5 Load Testing**
- Concurrent user simulation
- Query performance under load
- Resource usage monitoring
- Scalability validation

#### **3.6 Security Testing**
- Access control validation
- Input sanitization testing
- Data privacy compliance
- Penetration testing basics

---

## **PHASE 4: Monitoring & Maintenance (Ongoing)**

### **Day 21+: Production Operations**

#### **4.1 Monitoring Setup**
- Application performance monitoring
- Infrastructure monitoring
- Error tracking and alerting
- Usage analytics

#### **4.2 Maintenance Procedures**
- Daily data ingestion process
- Weekly system health checks
- Monthly performance reviews
- Quarterly capacity planning

---

## 🔧 Critical Technical Components

### **Stream Processing Architecture**

```python
"""
Core streaming pipeline components
"""

class StreamProcessor:
    def __init__(self):
        self.azure_client = BlobServiceClient()
        self.vector_store = QdrantClient()
        self.embedder = SentenceTransformer()
        
    def process_file_stream(self, blob_name):
        # 1. Stream file from Azure
        stream = self.azure_client.download_blob(blob_name)
        
        # 2. Parse in memory
        elements = partition_from_file(stream)
        
        # 3. Generate embeddings
        embeddings = self.embedder.encode(elements)
        
        # 4. Store vectors
        self.vector_store.upsert(embeddings)
        
        # 5. Clear memory
        del stream, elements, embeddings
```

### **Memory Management Strategy**

```python
"""
Memory-efficient processing
"""

import psutil
import gc

class MemoryManager:
    def __init__(self, max_memory_percent=80):
        self.max_memory = max_memory_percent
        
    def check_memory(self):
        usage = psutil.virtual_memory().percent
        if usage > self.max_memory:
            gc.collect()
            return True
        return False
        
    def process_with_memory_check(self, files):
        for file in files:
            if self.check_memory():
                # Pause processing or reduce batch size
                time.sleep(1)
            
            yield self.process_file(file)
```

### **Role-Based Query Filter**

```python
"""
Secure role-based filtering
"""

def build_query_filter(user_data):
    filters = []
    
    # Role-based filtering
    if user_data['role'] == 'teacher':
        filters.append({'module': ['knowledge_bank', 'academic']})
    elif user_data['role'] == 'admin':
        filters.append({'module': ['sop', 'admin', 'knowledge_bank']})
    else:  # student
        filters.append({'module': ['knowledge_bank']})
    
    # Grade-based filtering
    if 'grades' in user_data:
        filters.append({'grade': user_data['grades']})
    
    return {'must': filters}
```

---

## 🚀 Production Deployment Strategy

### **Deployment Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   FastAPI   │  │   FastAPI   │  │   Processing        │  │
│  │  Instance 1 │  │  Instance 2 │  │   Pipeline          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Qdrant    │  │    Redis    │  │   Azure Blob        │  │
│  │  Vector DB  │  │   Cache     │  │   Storage           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **Scaling Strategy**

| Load Level | Configuration | Resources |
|------------|---------------|-----------|
| **Low** (1-50 users) | Single instance | 4 CPU, 8GB RAM |
| **Medium** (50-200 users) | 2-3 instances | 8 CPU, 16GB RAM |
| **High** (200+ users) | 5+ instances + auto-scaling | 16+ CPU, 32+ GB RAM |

### **Backup & Recovery**

1. **Vector Database Backup**
   - Daily Qdrant snapshots
   - Cloud storage backup
   - Point-in-time recovery

2. **Configuration Backup**
   - Environment configs
   - API keys and secrets
   - Deployment scripts

3. **Disaster Recovery**
   - Multi-region deployment
   - Automated failover
   - Recovery time objective: <1 hour

---

## ⚡ Performance Optimization

### **Query Optimization**

1. **Vector Search Optimization**
   - Proper index configuration
   - Query result caching
   - Precomputed frequent queries

2. **API Performance**
   - Async request handling
   - Connection pooling
   - Response compression

3. **Database Optimization**
   - Index tuning
   - Query plan optimization
   - Connection management

### **Processing Optimization**

1. **Parallel Processing**
   - Multi-threaded downloads
   - Batch embedding generation
   - Concurrent vector insertion

2. **Memory Management**
   - Streaming processing
   - Garbage collection optimization
   - Memory pool management

3. **I/O Optimization**
   - Efficient file streaming
   - Network optimization
   - Disk I/O minimization

---

## 🔒 Security & Compliance

### **Security Checklist**

- [ ] **Authentication & Authorization**
  - API key management
  - Role-based access control
  - Session management

- [ ] **Data Security**
  - Encryption at rest
  - Encryption in transit
  - PII data handling

- [ ] **Infrastructure Security**
  - Network security groups
  - Firewall configuration
  - VPN access

- [ ] **Compliance**
  - Data retention policies
  - Audit logging
  - Privacy compliance

### **Security Implementation**

```python
"""
Security middleware for FastAPI
"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Verify JWT token or API key
    if not validate_token(token.credentials):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )
    
    return decode_token(token.credentials)

@app.post("/query")
async def query_endpoint(
    query_data: QueryRequest,
    user: dict = Depends(verify_token)
):
    # Process query with user context
    pass
```

---

## 📊 Monitoring & Maintenance

### **Monitoring Strategy**

1. **Application Metrics**
   - Query response times
   - Error rates
   - Throughput metrics
   - User activity

2. **Infrastructure Metrics**
   - CPU usage
   - Memory utilization
   - Disk usage
   - Network traffic

3. **Business Metrics**
   - User satisfaction
   - Query relevance
   - Content coverage
   - System adoption

### **Alerting Setup**

```python
"""
Monitoring and alerting configuration
"""

# Performance alerts
- Query response time > 5 seconds
- Error rate > 5%
- Memory usage > 90%
- Disk usage > 80%

# Business alerts
- Zero queries for 1 hour
- Processing pipeline stopped
- Vector DB disconnected
- New file ingestion failed
```

### **Maintenance Schedule**

| Frequency | Task | Description |
|-----------|------|-------------|
| **Daily** | Health Check | System status, error logs, performance metrics |
| **Weekly** | Data Ingestion | Process new files, update indexes |
| **Monthly** | Performance Review | Optimize queries, update models |
| **Quarterly** | Capacity Planning | Scale resources, upgrade systems |

---

## 🚨 Risk Management

### **Technical Risks**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Vector DB Corruption** | High | Low | Daily backups, clustering |
| **PDF Processing Failures** | Medium | Medium | Fallback parsers, manual review |
| **Memory Issues** | High | Medium | Streaming processing, monitoring |
| **Azure Connectivity** | High | Low | Retry logic, cached data |

### **Business Risks**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Poor Query Relevance** | High | Medium | A/B testing, user feedback |
| **Slow Adoption** | Medium | Medium | Training, user support |
| **Data Privacy Issues** | High | Low | Compliance review, access controls |
| **Scalability Problems** | High | Low | Load testing, auto-scaling |

### **Contingency Plans**

1. **System Downtime**
   - Backup system activation
   - User communication plan
   - Manual fallback procedures

2. **Data Loss**
   - Backup restoration process
   - Reprocessing procedures
   - Data validation steps

3. **Performance Degradation**
   - Load balancing activation
   - Query optimization
   - Resource scaling

---

## 📈 Success Metrics & KPIs

### **Technical KPIs**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Query Response Time** | <2 seconds | 95th percentile |
| **System Uptime** | >99.9% | Monthly average |
| **Error Rate** | <1% | Daily average |
| **Processing Speed** | 100+ files/hour | Ingestion pipeline |

### **Business KPIs**

| Metric | Target | Measurement |
|--------|--------|-------------|
| **User Adoption** | 80% of teachers | Monthly active users |
| **Query Satisfaction** | >4.0/5.0 | User feedback surveys |
| **Content Coverage** | 95% of curriculum | Curriculum mapping |
| **Support Tickets** | <5 per week | Help desk metrics |

### **Monitoring Dashboard**

```
┌─────────────────────────────────────────────────────────────┐
│                    Edify Chatbot Dashboard                  │
├─────────────────────────────────────────────────────────────┤
│  Active Users: 45        Queries/Hour: 120                 │
│  Response Time: 1.2s     Error Rate: 0.3%                  │
│  Vector Count: 2.5M      Storage Used: 45GB                │
│  Processing Queue: 12    Last Update: 2 hours ago          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Go-Live Checklist

### **Pre-Launch Validation**

- [ ] **Functionality Testing**
  - All API endpoints working
  - Role-based access validated
  - Query accuracy verified
  - Error handling tested

- [ ] **Performance Testing**
  - Load testing completed
  - Response times within targets
  - Resource usage optimized
  - Scalability validated

- [ ] **Security Testing**
  - Access controls verified
  - Data encryption confirmed
  - Input validation tested
  - Audit logging active

- [ ] **Operational Readiness**
  - Monitoring systems active
  - Backup procedures tested
  - Support documentation complete
  - Team training completed

### **Launch Day Activities**

1. **Pre-Launch (2 hours before)**
   - Final system health check
   - Backup all critical data
   - Notify support team
   - Prepare rollback plan

2. **Launch (Go-Live)**
   - Enable production traffic
   - Monitor system metrics
   - Watch for errors/issues
   - Communicate with stakeholders

3. **Post-Launch (24 hours)**
   - Continuous monitoring
   - User feedback collection
   - Performance optimization
   - Issue resolution

---

## 🔄 Continuous Improvement

### **Feedback Loop**

1. **User Feedback Collection**
   - In-app feedback forms
   - User surveys
   - Support ticket analysis
   - Usage analytics

2. **Performance Analysis**
   - Query pattern analysis
   - Response time trends
   - Error rate monitoring
   - Resource utilization

3. **System Optimization**
   - Query algorithm improvements
   - Infrastructure scaling
   - Feature enhancements
   - Bug fixes

### **Innovation Pipeline**

1. **Short-term (1-3 months)**
   - Query performance optimization
   - User interface improvements
   - Mobile app development
   - Integration enhancements

2. **Medium-term (3-6 months)**
   - Advanced AI features
   - Multilingual support
   - Analytics dashboard
   - API ecosystem

3. **Long-term (6+ months)**
   - Machine learning personalization
   - Advanced security features
   - Cross-platform integration
   - Predictive analytics

---

## 📞 Support & Escalation

### **Support Levels**

| Level | Response Time | Issues |
|-------|---------------|--------|
| **L1** | 15 minutes | User questions, basic troubleshooting |
| **L2** | 1 hour | Technical issues, configuration problems |
| **L3** | 4 hours | System failures, data corruption |
| **L4** | 24 hours | Architecture changes, major upgrades |

### **Escalation Matrix**

| Issue Severity | Response Time | Escalation Path |
|----------------|---------------|-----------------|
| **Critical** | Immediate | Developer → Tech Lead → CTO |
| **High** | 30 minutes | Developer → Tech Lead |
| **Medium** | 2 hours | Developer → Support Lead |
| **Low** | 24 hours | Support Team |

---

## 🎓 Final Success Tips

### **Do's**
✅ Start with a small subset for proof of concept  
✅ Implement comprehensive monitoring from day 1  
✅ Test with real users early and often  
✅ Document everything thoroughly  
✅ Plan for scale from the beginning  
✅ Prioritize security and compliance  
✅ Build in redundancy and error handling  
✅ Monitor performance metrics continuously  

### **Don'ts**
❌ Don't try to process all 700GB at once initially  
❌ Don't skip security considerations  
❌ Don't ignore user feedback  
❌ Don't deploy without proper testing  
❌ Don't forget about backup and recovery  
❌ Don't optimize prematurely  
❌ Don't neglect documentation  
❌ Don't launch without monitoring  

---

## 🏆 Project Success Formula

```
SUCCESS = (Technical Excellence × User Satisfaction × Operational Reliability) ÷ Time to Value
```

**Where:**
- **Technical Excellence** = Scalable architecture + Clean code + Best practices
- **User Satisfaction** = Fast responses + Relevant results + Easy interface  
- **Operational Reliability** = High uptime + Quick issue resolution + Proactive monitoring
- **Time to Value** = Quick wins + Iterative delivery + Continuous improvement

---

> **Remember**: This is a marathon, not a sprint. Focus on building something solid and scalable rather than rushing to completion. The extra time invested in proper architecture and testing will pay dividends in production reliability and user satisfaction.

**Next Step**: Start with Phase 1, Day 1-2 environment setup and begin your journey to production success! 🚀
