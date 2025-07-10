# Edify Azure Ingestion & Chatbot Pipeline

This project provides a modular pipeline for downloading, parsing, and embedding academic content from Azure Blob Storage, supporting a role-based chatbot for Edify Schools.

# 🚀 Edify School Chatbot Project

![Edify Logo](https://dummyimage.com/200x60/ededed/333333&text=Edify+Chatbot) <!-- Replace with actual logo if available -->

---

## 📚 Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Data Flow](#data-flow)
4. [Access Control](#access-control)
5. [Project Timeline](#project-timeline)
6. [Tech Stack](#tech-stack)
7. [Setup & Next Steps](#setup--next-steps)
8. [Contacts](#contacts)

---

## 🎯 Overview
A large-scale, role-based chatbot system for Edify Schools. Teachers, students, and admin staff can securely query academic content stored on Azure Blob Storage.

---

## 🏗️ System Architecture

```
Azure Blob Storage
      │
      ▼
Streamer (stream.py)
      │
      ▼
Document Parser & Chunker (parse.py)
      │
      ▼
Embedding Generator (embed.py)
      │
      ▼
Vector DB (Qdrant)
      │
      ▼
Chatbot API
      │
      ▼
Edify Portal
```


- **Source:** Azure Blob Storage
- **Structure:**
  ```
  SOP/
    ├── admin/
    ├── academic/
  Knowledge Bank/
    └── kb/2025-2026/
        ├── ik1/
        ├── grade1/
        └── grade12/
  ```
- ~600-700 GB of data (~90% PDFs with images), ~100 new files uploaded daily.

---

## 🔐 Access Control
Edify’s portal handles all user authentication and role mapping.

- On each query, the portal sends a JSON payload to our API:

```json
{
  "user_id": "raj_teacher",
  "role": "teacher",
  "grades": ["ik1", "ik2"],
  "query": "Explain photosynthesis"
}
```

- Our backend filters results in the vector database based on this metadata.

**Sample Metadata Structure:**
```json
{
  "content": "Light travels faster than sound...",
  "metadata": {
    "module": "knowledge_bank",
    "grade": "ik1",
    "year": "2025-2026",
    "source_file": "abc.pdf"
  }
}
```

---

## 🗓️ Project Timeline

| Task                                 | Est. Time |
| ------------------------------------ | --------- |
| Collect Azure credentials            | 0.5 day   |
| Setup Azure Blob downloader          | 1 day     |
| Parse & chunk with Unstructured.io   | 2 days    |
| Auto metadata tagging logic          | 1 day     |
| Generate embeddings                  | 2 days    |
| Store in Vector DB (Qdrant/Pinecone) | 1 day     |
| Test with small sample batch         | 0.5 day   |
| Documentation & logs                 | 0.5 day   |
| **Total**                            | **~8 days (~1.5 weeks)** |

---

## 🛠️ Tech Stack
- Python 3.10+
- Azure Blob SDK
- Unstructured.io for document parsing
- SentenceTransformers / OpenAI for embeddings
- Qdrant / Pinecone as vector DB
- FastAPI / Flask (for chatbot endpoint)
- VS Code / Cursor for development

---

## ⚙️ Setup & Next Steps
- Await Azure connection string or SAS token from Edify.
- Confirm sample files for initial tests.
- Setup Python environment, VS Code, pip packages.
- Initialize GitHub repo & push first pipeline modules.

---

## 👥 Contacts
| Role         | Name/Contact         |
|--------------|---------------------|
| Developer    | Akshay (i95) |
| Client Side  | Sampath (Edify)     |

---

# Edify Chatbot Production-Ready Pipeline

This README is a condensed, actionable guide for building and operating a streaming, role-based academic chatbot at scale. For full details, see `PRODUCTION_GUIDE.md`.

---

## 📋 Table of Contents
1. [Project Vision](#project-vision)
2. [Production Architecture](#production-architecture)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Key Components](#key-components)
5. [Deployment & Operations](#deployment--operations)
6. [KPIs & Success](#kpis--success)
7. [Support](#support)

---

## 🎯 Project Vision
- **Goal:** Stream and process 600-700GB of academic content from Azure Blob Storage with ZERO local storage, supporting sub-2s, role-based queries for Edify Schools.
- **Users:** Teachers, students, admin staff (role-based access)
- **Performance:** <2s query response, 99.9% uptime
- **Approach:** Streaming pipeline (Azure → Memory → Vector DB)

---

## 🏗️ Production Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   STREAMING PIPELINE ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────┤
│  Azure Blob Storage (600-700GB)                                │
│           ↓ (Stream - No Download)                             │
│  Memory Buffer (50-200MB max)                                  │
│           ↓                                                   │
│  Unstructured.io Parser                                        │
│           ↓                                                   │
│  Text Chunking & Metadata Extraction                           │
│           ↓                                                   │
│  Embedding Generation (Batched)                                │
│           ↓                                                   │
│  Qdrant Vector Store                                           │
│           ↓                                                   │
│  Clear Memory & Process Next File                              │
├─────────────────────────────────────────────────────────────────┤
│                     QUERY ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│  Edify Portal → FastAPI Backend → Redis Cache → Qdrant Vector  │
│  Search + Role Filtering → Response + Cache Update             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚦 Step-by-Step Implementation

### 1. Environment & Prerequisites
- Python 3.10+, Docker Desktop, Git, VS Code
- Install dependencies: `azure-storage-blob`, `qdrant-client`, `unstructured[pdf]`, `sentence-transformers`, `fastapi`, `uvicorn[standard]`, `redis`, `python-dotenv`, `psutil`, `httpx`
- Start Qdrant: `docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest`
- Create `.env` with Azure and Qdrant config

### 2. Project Structure
```
project/
├── src/
│   ├── ingestion/
│   │   ├── azure_streamer.py
│   │   ├── document_processor.py
│   │   ├── embedding_generator.py
│   │   └── vector_store.py
│   ├── api/
│   │   └── query_api.py
│   └── utils/
├── test_streaming.py
├── test_api.py
├── test_integration.py
├── logs/
├── .env
├── requirements.txt
└── README.md
```

### 3. Streaming Pipeline
- **AzureBlobStreamer:** Streams files from Azure to memory, extracts metadata
- **DocumentProcessor:** Parses PDFs in memory, chunks text, attaches metadata
- **EmbeddingGenerator:** Generates vector embeddings in batches
- **VectorStoreManager:** Stores/searches vectors in Qdrant, role-based filtering
- **StreamingPipeline:** Orchestrates end-to-end streaming, embedding, and storage

### 4. FastAPI Backend
- `/query`: Role-based search endpoint (with Redis caching)
- `/health`: Health check
- `/stats`: System stats

---

## 🧩 Key Components
- **Streaming (No Local Storage):** All processing in memory, no temp files
- **Role-Based Filtering:** Secure, metadata-driven access for students, teachers, admins
- **Batch Processing:** Efficient embedding and vector storage
- **Caching:** Redis for sub-second repeat queries
- **Monitoring:** Logs, stats endpoints, and alerting

---

## 🚀 Deployment & Operations
- **Containerization:** Use Docker Compose for API, Qdrant, Redis
- **Cloud:** Deploy on Azure Container Instances or Kubernetes
- **Scaling:** Multi-instance FastAPI, auto-scaling, resource monitoring
- **Backup:** Daily Qdrant snapshots, config backup, disaster recovery plan
- **Security:** API key/JWT auth, role-based access, encryption, compliance
- **Maintenance:** Daily health checks, weekly ingestion, monthly reviews

---

## 📈 KPIs & Success
| Metric                | Target         |
|-----------------------|---------------|
| Query Response Time   | <2 seconds    |
| System Uptime         | >99.9%        |
| Error Rate            | <1%           |
| Processing Speed      | 100+ files/hr |
| User Adoption         | 80% teachers  |
| Query Satisfaction    | >4.0/5.0      |

---

## 🆘 Support & Escalation
| Level | Response Time | Issues                        |
|-------|---------------|-------------------------------|
| L1    | 15 min        | User questions, troubleshooting|
| L2    | 1 hour        | Technical/config issues        |
| L3    | 4 hours       | System failures, data issues   |
| L4    | 24 hours      | Major upgrades, architecture   |

---

> For full production details, troubleshooting, and code samples, see `PRODUCTION_GUIDE.md`.



