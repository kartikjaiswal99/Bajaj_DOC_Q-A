# Workflow and System Diagrams

This document contains detailed workflow diagrams and visual representations of the system architecture to help understand the complete document processing and question-answering pipeline.

## Complete System Workflow

```mermaid
graph TB
    %% Client Layer
    A[Client Application] --> B[HTTP POST Request]
    B --> C[FastAPI /hackrx/run Endpoint]
    
    %% Request Processing
    C --> D{Validate Input}
    D -->|Invalid| E[Return 422 Error]
    D -->|Valid| F[Extract Document URL & Questions]
    
    %% Document Processing Pipeline
    F --> G[Document Processor]
    G --> H{Check LRU Cache}
    H -->|Cache Hit| I[Use Cached Chunks]
    H -->|Cache Miss| J[Download Document]
    
    %% Document Download & Processing
    J --> K{Document Accessible?}
    K -->|No| L[Return 400 Error]
    K -->|Yes| M[Detect File Format]
    
    %% Format-Specific Processing
    M --> N{File Type?}
    N -->|PDF| O[PyMuPDF Text Extraction]
    N -->|DOCX| P[python-docx Extraction]
    N -->|EML| Q[Email Parser Extraction]
    N -->|Other| L
    
    %% Text Processing
    O --> R[Raw Text Content]
    P --> R
    Q --> R
    
    %% Size-Based Processing Strategy
    R --> S{Document Size Check}
    S -->|>1MB| T[Large Document Handler]
    S -->|≤1MB| U[Standard Processing]
    
    %% Large Document Processing
    T --> V[Split into 500KB Sections]
    V --> W[Process Each Section]
    W --> X[Structure-Aware Chunking]
    
    %% Standard Document Processing
    U --> X
    
    %% Chunking and Metadata
    X --> Y[Add Metadata to Chunks]
    Y --> Z[Store in Cache]
    Z --> I
    
    %% Vector Processing Pipeline
    I --> AA[Estimate Token Count]
    AA --> BB[Create Embedding Batches]
    BB --> CC{Batch Size OK?}
    CC -->|Too Large| DD[Split into Sub-batches]
    CC -->|OK| EE[Generate OpenAI Embeddings]
    DD --> EE
    
    %% Vector Store Creation
    EE --> FF[Create FAISS Vector Store]
    FF --> GG[Build Search Index]
    
    %% Question Processing
    GG --> HH[Initialize Parallel Processing]
    HH --> II{Determine Processing Mode}
    II -->|Large Dataset| JJ[Fast Mode - Skip Reranking]
    II -->|Small Dataset| KK[Thorough Mode - With Reranking]
    
    %% Parallel Question Processing
    JJ --> LL[Process Questions in Parallel]
    KK --> LL
    
    %% Individual Question Processing
    LL --> MM[For Each Question:]
    MM --> NN[Vector Similarity Search]
    NN --> OO[Retrieve Top-K Documents]
    OO --> PP{Reranking Mode?}
    PP -->|Fast| QQ[Use Top 6 Directly]
    PP -->|Thorough| RR[LLM-Based Reranking]
    
    %% Context Assembly
    QQ --> SS[Assemble Context]
    RR --> SS
    SS --> TT[Create Prompt with Context]
    
    %% Answer Generation
    TT --> UU[Send to OpenAI GPT-4o-mini]
    UU --> VV[Receive Generated Answer]
    VV --> WW[Post-process Answer]
    WW --> XX[Optimize Length & Format]
    
    %% Response Assembly
    XX --> YY[Collect All Answers]
    YY --> ZZ[Format JSON Response]
    ZZ --> AAA[Return to Client]
    
    %% Error Handling
    L --> BBB[Error Handler]
    E --> BBB
    BBB --> CCC[Log Error]
    CCC --> DDD[Return Error Response]
    
    %% Styling
    classDef errorClass fill:#ffcccc,stroke:#ff6666,stroke-width:2px
    classDef processClass fill:#cceeff,stroke:#0066cc,stroke-width:2px
    classDef cacheClass fill:#ccffcc,stroke:#00cc00,stroke-width:2px
    classDef aiClass fill:#ffffcc,stroke:#ffcc00,stroke-width:2px
    
    class E,L,BBB,CCC,DDD errorClass
    class G,X,Y,AA,BB,FF,GG,HH,LL,NN,SS,TT processClass
    class H,I,Z cacheClass
    class EE,UU,VV,RR aiClass
```

## Document Processing Detailed Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant DocProcessor
    participant Cache
    participant FileDownloader
    participant TextExtractor
    participant Chunker
    participant VectorStore
    participant OpenAI
    
    Client->>FastAPI: POST /hackrx/run
    activate FastAPI
    
    FastAPI->>FastAPI: Validate Input Schema
    FastAPI->>DocProcessor: process_document(url)
    activate DocProcessor
    
    DocProcessor->>Cache: check_cache(url)
    activate Cache
    
    alt Cache Hit
        Cache-->>DocProcessor: return cached_chunks
    else Cache Miss
        Cache-->>DocProcessor: cache_miss
        
        DocProcessor->>FileDownloader: download(url)
        activate FileDownloader
        FileDownloader->>FileDownloader: HTTP GET request
        FileDownloader->>FileDownloader: save to temp file
        FileDownloader-->>DocProcessor: temp_file_path
        deactivate FileDownloader
        
        DocProcessor->>TextExtractor: extract_text(file_path)
        activate TextExtractor
        
        alt PDF File
            TextExtractor->>TextExtractor: PyMuPDF processing
        else DOCX File
            TextExtractor->>TextExtractor: python-docx processing
        else EML File
            TextExtractor->>TextExtractor: email parsing
        end
        
        TextExtractor-->>DocProcessor: raw_text
        deactivate TextExtractor
        
        alt Large Document (>1MB)
            DocProcessor->>DocProcessor: split_into_sections(500KB)
            loop For each section
                DocProcessor->>Chunker: structure_aware_chunking(section)
                activate Chunker
                Chunker->>Chunker: detect_headers
                Chunker->>Chunker: split_by_structure
                Chunker->>Chunker: optimize_chunk_size
                Chunker-->>DocProcessor: section_chunks
                deactivate Chunker
            end
        else Standard Document
            DocProcessor->>Chunker: structure_aware_chunking(text)
            activate Chunker
            Chunker-->>DocProcessor: chunks
            deactivate Chunker
        end
        
        DocProcessor->>DocProcessor: add_metadata(chunks)
        DocProcessor->>Cache: store_in_cache(url, chunks)
        Cache-->>DocProcessor: stored
    end
    
    deactivate Cache
    DocProcessor-->>FastAPI: chunked_documents
    deactivate DocProcessor
    
    FastAPI->>VectorStore: create_embeddings(chunks)
    activate VectorStore
    
    VectorStore->>VectorStore: estimate_tokens(chunks)
    VectorStore->>VectorStore: create_batches(200K_limit)
    
    loop For each batch
        VectorStore->>OpenAI: generate_embeddings(batch)
        activate OpenAI
        OpenAI-->>VectorStore: embedding_vectors
        deactivate OpenAI
    end
    
    VectorStore->>VectorStore: build_faiss_index
    VectorStore-->>FastAPI: vector_store
    deactivate VectorStore
    
    FastAPI->>FastAPI: create_retriever(vector_store)
    
    par Process Questions in Parallel
        loop For each question
            FastAPI->>FastAPI: process_single_question(q)
            FastAPI->>VectorStore: similarity_search(question)
            activate VectorStore
            VectorStore->>VectorStore: k-NN search
            VectorStore-->>FastAPI: relevant_docs
            deactivate VectorStore
            
            alt Thorough Mode
                FastAPI->>OpenAI: rerank_documents(question, docs)
                activate OpenAI
                OpenAI-->>FastAPI: reranked_docs
                deactivate OpenAI
            else Fast Mode
                FastAPI->>FastAPI: use_top_k_directly
            end
            
            FastAPI->>FastAPI: assemble_context(docs)
            FastAPI->>OpenAI: generate_answer(context, question)
            activate OpenAI
            OpenAI-->>FastAPI: generated_answer
            deactivate OpenAI
            
            FastAPI->>FastAPI: post_process_answer
        end
    end
    
    FastAPI->>FastAPI: collect_all_answers
    FastAPI->>FastAPI: format_response
    FastAPI-->>Client: JSON response
    deactivate FastAPI
```

## Vector Processing Pipeline

```mermaid
graph LR
    A[Document Chunks] --> B[Token Estimation]
    B --> C{Estimate Tokens<br/>3 chars = 1 token}
    C --> D[Batch Creation]
    D --> E{Batch Size<br/>≤ 200K tokens?}
    E -->|No| F[Split Batch]
    F --> E
    E -->|Yes| G[OpenAI Embeddings API]
    
    G --> H[Vector Response<br/>1536 dimensions]
    H --> I[FAISS Index Creation]
    I --> J[Similarity Search Setup]
    
    %% Batch Processing Details
    K[Batch 1<br/>Documents 1-N] --> G
    L[Batch 2<br/>Documents N+1-M] --> G
    M[Batch K<br/>Remaining Docs] --> G
    
    %% FAISS Operations
    I --> N[Index Building]
    N --> O[Memory Optimization]
    O --> P[Search Interface]
    
    %% Search Process
    P --> Q[Query Vector]
    Q --> R[k-NN Search]
    R --> S[Distance Calculation]
    S --> T[Top-K Results]
    
    classDef batchClass fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef vectorClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef faissClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class B,C,D,E,F batchClass
    class G,H,Q,R,S vectorClass
    class I,J,N,O,P,T faissClass
```

## Question Processing Architecture

```mermaid
graph TD
    A[Questions Array] --> B[ThreadPoolExecutor]
    B --> C{Determine Workers}
    C -->|Large Docs >2000| D[3 Workers]
    C -->|Many Questions >3| E[4 Workers]
    C -->|Standard Load| F[2 Workers]
    
    D --> G[Parallel Processing]
    E --> G
    F --> G
    
    G --> H[Worker 1: Question 1]
    G --> I[Worker 2: Question 2]
    G --> J[Worker N: Question N]
    
    %% Individual Question Processing
    H --> K1[Vector Search]
    I --> K2[Vector Search]
    J --> K3[Vector Search]
    
    K1 --> L1{Processing Mode?}
    K2 --> L2{Processing Mode?}
    K3 --> L3{Processing Mode?}
    
    L1 -->|Fast| M1[Top-6 Direct]
    L1 -->|Thorough| N1[LLM Reranking]
    L2 -->|Fast| M2[Top-6 Direct]
    L2 -->|Thorough| N2[LLM Reranking]
    L3 -->|Fast| M3[Top-6 Direct]
    L3 -->|Thorough| N3[LLM Reranking]
    
    %% Context Assembly
    M1 --> O1[Context Assembly]
    N1 --> O1
    M2 --> O2[Context Assembly]
    N2 --> O2
    M3 --> O3[Context Assembly]
    N3 --> O3
    
    %% Answer Generation
    O1 --> P1[OpenAI GPT-4o-mini]
    O2 --> P2[OpenAI GPT-4o-mini]
    O3 --> P3[OpenAI GPT-4o-mini]
    
    P1 --> Q1[Answer 1]
    P2 --> Q2[Answer 2]
    P3 --> Q3[Answer N]
    
    %% Synchronization
    Q1 --> R[Collect Results]
    Q2 --> R
    Q3 --> R
    
    R --> S[Order Answers]
    S --> T[Format Response]
    T --> U[Return JSON Array]
    
    classDef workerClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef searchClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef aiClass fill:#ffffcc,stroke:#ffcc00,stroke-width:2px
    
    class H,I,J,G workerClass
    class K1,K2,K3,M1,M2,M3,N1,N2,N3 searchClass
    class P1,P2,P3 aiClass
```

## Reranking Process Flow

```mermaid
graph LR
    A[Retrieved Documents] --> B{Fast Mode?}
    B -->|Yes| C[Use Top-6 Directly]
    B -->|No| D[Reranking Process]
    
    D --> E[Batch Documents<br/>Size: 3 docs]
    E --> F[For Each Document in Batch]
    F --> G[Create Scoring Prompt]
    G --> H[Send to GPT-4o-mini]
    H --> I[Extract Relevance Score<br/>1-10 scale]
    I --> J{More Documents?}
    J -->|Yes| F
    J -->|No| K[Sort by Score]
    
    K --> L[Rate Limiting Delay<br/>0.05 seconds]
    L --> M{More Batches?}
    M -->|Yes| E
    M -->|No| N[Final Ranking]
    
    N --> O[Return Top-6 Documents]
    C --> O
    
    %% Scoring Details
    P[Scoring Prompt Template] --> Q["Query: {question}<br/>Section: {section}<br/>Content: {content}<br/>Score 1-10 relevance"]
    Q --> H
    
    classDef fastClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef scoringClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef delayClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class C fastClass
    class F,G,H,I,K,N scoringClass
    class L delayClass
```

## Error Handling Flow

```mermaid
graph TD
    A[Request Processing] --> B{Validation Check}
    B -->|Fail| C[422 Validation Error]
    B -->|Pass| D[Document Processing]
    
    D --> E{Document Download}
    E -->|Fail| F[400 Document Error]
    E -->|Success| G[Text Extraction]
    
    G --> H{Extraction Success}
    H -->|Fail| I[400 Processing Error]
    H -->|Success| J[Vector Processing]
    
    J --> K{OpenAI API Call}
    K -->|Rate Limit| L[Retry with Backoff]
    K -->|Other Error| M[500 Server Error]
    K -->|Success| N[Question Processing]
    
    L --> O{Retry Count < 3}
    O -->|Yes| P[Exponential Backoff]
    P --> K
    O -->|No| M
    
    N --> Q{Processing Success}
    Q -->|Fail| R[Fallback Answer]
    Q -->|Success| S[Format Response]
    
    R --> T["Unable to determine from context"]
    S --> U[Return Success Response]
    
    %% Error Response Formatting
    C --> V[Format Error Response]
    F --> V
    I --> V
    M --> V
    
    V --> W[Log Error Details]
    W --> X[Return HTTP Error]
    
    classDef errorClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef retryClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef successClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class C,F,I,M,V,W,X errorClass
    class L,O,P retryClass
    class S,U successClass
```

## Performance Optimization Flow

```mermaid
graph LR
    A[Request Input] --> B{Dataset Size Analysis}
    B -->|Large Docs >2000| C[Performance Mode: Large]
    B -->|Many Questions >3| D[Performance Mode: Many]
    B -->|Standard| E[Performance Mode: Balanced]
    
    %% Large Document Optimizations
    C --> F[Worker Count: 3]
    C --> G[Fast Mode: Enabled]
    C --> H[Chunk Size: Reduced]
    C --> I[Batch Size: Optimized]
    
    %% Many Questions Optimizations
    D --> J[Worker Count: 4]
    D --> K[Parallel Processing: Max]
    D --> L[Reranking: Selective]
    
    %% Balanced Mode
    E --> M[Worker Count: 2]
    E --> N[Thorough Processing]
    E --> O[Full Reranking]
    
    %% Processing Paths
    F --> P[Execute Processing]
    G --> P
    H --> P
    I --> P
    
    J --> P
    K --> P
    L --> P
    
    M --> P
    N --> P
    O --> P
    
    %% Performance Monitoring
    P --> Q[Monitor Response Time]
    P --> R[Monitor Memory Usage]
    P --> S[Monitor API Calls]
    
    Q --> T[Performance Metrics]
    R --> T
    S --> T
    
    T --> U[Adaptive Optimization]
    U --> V[Adjust Parameters]
    V --> A
    
    classDef modeClass fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef optClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef monitorClass fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    
    class C,D,E modeClass
    class F,G,H,I,J,K,L,M,N,O optClass
    class Q,R,S,T,U,V monitorClass
```

## Caching Strategy

```mermaid
graph TD
    A[Document URL Request] --> B[Generate Cache Key]
    B --> C{Check LRU Cache}
    C -->|Hit| D[Return Cached Chunks]
    C -->|Miss| E[Process Document]
    
    E --> F[Download & Extract]
    F --> G[Structure-Aware Chunking]
    G --> H[Add Metadata]
    H --> I[Store in Cache]
    I --> J[Return Processed Chunks]
    
    %% Cache Management
    K[Cache Management] --> L{Cache Full?}
    L -->|Yes| M[LRU Eviction]
    L -->|No| N[Direct Storage]
    
    M --> O[Remove Oldest Entry]
    O --> N
    N --> P[Update Cache Statistics]
    
    %% Cache Key Generation
    Q[URL Input] --> R[Normalize URL]
    R --> S[Generate Hash]
    S --> T[Cache Key]
    
    %% Cache Statistics
    U[Cache Operations] --> V[Hit Rate Monitoring]
    V --> W[Performance Metrics]
    W --> X[Cache Optimization]
    
    classDef cacheClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef processClass fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef mgmtClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    
    class C,D,I,L,M,N,O cacheClass
    class E,F,G,H,J processClass
    class K,P,U,V,W,X mgmtClass
```

## Token Management System

```mermaid
graph LR
    A[Document Chunks] --> B[Token Estimation]
    B --> C[Conservative Calculation<br/>3 chars = 1 token]
    C --> D[Batch Assembly]
    
    D --> E{Total Tokens<br/>≤ 200K limit?}
    E -->|No| F[Split Current Batch]
    E -->|Yes| G[Add to Batch Queue]
    
    F --> H[Create Sub-batch]
    H --> E
    
    G --> I[Process Batch Queue]
    I --> J[Send to OpenAI API]
    J --> K{API Response}
    
    K -->|Success| L[Store Embeddings]
    K -->|Token Error| M[Reduce Batch Size]
    K -->|Rate Limit| N[Exponential Backoff]
    
    M --> O[Split Further]
    O --> J
    
    N --> P[Wait Period]
    P --> J
    
    L --> Q{More Batches?}
    Q -->|Yes| I
    Q -->|No| R[Complete Processing]
    
    %% Token Tracking
    S[Token Tracking] --> T[Batch Statistics]
    T --> U[Rate Monitoring]
    U --> V[Optimization Feedback]
    V --> B
    
    classDef tokenClass fill:#fff8e1,stroke:#f57c00,stroke-width:2px
    classDef batchClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef errorClass fill:#ffebee,stroke:#c62828,stroke-width:2px
    
    class B,C,E,F,H tokenClass
    class D,G,I,L,Q,R batchClass
    class M,N,O,P errorClass
```

These diagrams provide a comprehensive visual understanding of the entire system workflow, from initial request processing through document handling, vector processing, question answering, and response generation. Each diagram focuses on specific aspects of the system to help developers and users understand the complex interactions and optimizations built into the system.