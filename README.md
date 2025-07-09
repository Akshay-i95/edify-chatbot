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

```mermaid
graph TD
    A[Azure Blob Storage] --> B[Downloader]
    B --> C[Document Parser & Chunker]
    C --> D[Metadata Tagger]
    D --> E[Embedding Generator]
    E --> F[Vector DB (Qdrant/Pinecone)]
    F --> G[Chatbot API]
    G --> H[Edify Portal]
```

---

## 🔄 Data Flow

```
[Azure Blob]
    ↓
[Download]
    ↓
[Parse & Chunk]
    ↓
[Attach Metadata]
    ↓
[Generate Embeddings]
    ↓
[Store in Vector DB]
    ↓
[Query with Role-Based Filters]
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

> **Note:**
> This README will evolve with API endpoints, role-based examples, and test cases in the next phase.



