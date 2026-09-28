# ResearchGrant AI

AI-powered research grant matching and proposal analysis using Retrieval-Augmented Generation (RAG).

ResearchGrant AI retrieves relevant grant opportunities from a local grant database using semantic similarity and then uses an LLM to analyze how well a research proposal matches the retrieved grants.

---

## Features

- Semantic search over research grant documents
- Retrieval-Augmented Generation (RAG)
- AI-powered proposal analysis
- Grant relevance matching
- Eligibility requirement analysis
- Missing information detection
- Required document identification
- Proposal-funding alignment analysis
- Suggested proposal improvements
- Retrieval similarity scores
- Maximum funding information
- Evidence from the original grant documents
- Streamlit-based web interface
- Local vector index using FAISS
- Sentence Transformers for embeddings
- Gemini API for language-model analysis

---

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Research Proposal │
                    │      (User Input)   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Embedding Model   │
                    │ all-MiniLM-L6-v2    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Semantic Search   │
                    │       FAISS         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Relevant Grant      │
                    │ Documents           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Gemini LLM     │
                    │ Proposal Analysis   │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Analysis Results          │
              │                                 │
              │ • Relevant Grants               │
              │ • Why They Match                │
              │ • Eligibility                   │
              │ • Missing Requirements          │
              │ • Required Documents            │
              │ • Funding Alignment             │
              │ • Suggested Improvements        │
              └─────────────────────────────────┘
```

---

## Project Structure

```text
ResearchGrant-AI/
│
├── app.py
├── rag.py
├── ingest.py
├── build_index.py
├── evaluate.py
├── test_rag.py
│
├── grant_index.pkl
│
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── grant_1.json
│   ├── grant_2.json
│   ├── grant_3.json
│   ├── grant_4.json
│   ├── grant_5.json
│   ├── grant_6.json
│   ├── grant_7.json
│   ├── grant_8.json
│   └── grant_9.json
│
└── .streamlit/
    └── config.toml
```

---

## Technologies Used

| Technology            | Purpose                         |
| --------------------- | ------------------------------- |
| Python                | Core programming language       |
| Streamlit             | Web application interface       |
| Sentence Transformers | Text embeddings                 |
| all-MiniLM-L6-v2      | Embedding model                 |
| FAISS                 | Vector similarity search        |
| Gemini API            | AI-powered proposal analysis    |
| NumPy                 | Numerical operations            |
| Pandas                | Data processing                 |
| python-dotenv         | Environment variable management |

---

## How the System Works

### 1. Grant Documents

Grant information is stored as JSON files inside the `data/` directory.

Each document contains information such as:

* Grant name
* Objective
* Eligible applicants
* Focus areas
* Funding information
* Required documents
* Other grant requirements

---

### 2. Data Ingestion

The grant documents are processed using the ingestion pipeline.

Run:

```bash
python ingest.py
```

This prepares the grant documents for indexing.

---

### 3. Embedding Generation

The project uses:

```text
all-MiniLM-L6-v2
```

to convert grant documents into numerical vector representations.

The resulting embeddings are stored in the local vector index.

---

### 4. Building the Search Index

Run:

```bash
python build_index.py
```

This creates:

```text
grant_index.pkl
```

The index contains the grant embeddings and associated grant information.

Example output:

```text
INDEX CREATED SUCCESSFULLY

Grant documents : 9
Embeddings      : (9, 384)
Index file      : grant_index.pkl
```

---

### 5. Proposal Retrieval

When a user enters a research proposal:

1. The proposal is converted into an embedding.
2. Semantic similarity is calculated against the grant embeddings.
3. The most relevant grants are retrieved.
4. The similarity scores are displayed.

For example:

```text
Green Robotics Innovation Grant       0.4920
Ocean & Marine Environmental Grant    0.4723
Smart Waste Management Grant          0.4672
Clean Water Technology Grant          0.4388
Environmental AI & IoT Grant          0.3778
```

The similarity score represents semantic similarity between the proposal and the grant document. It is not a funding probability or eligibility score.

---

## AI Analysis

After relevant grants are retrieved, the retrieved grant information and proposal are provided to the configured LLM.

The system analyzes:

### Potentially Relevant Funding Opportunities

Identifies grants that are relevant to the proposal.

### Why They Match

Explains which grant focus areas correspond to the proposal.

### Eligibility Requirements

Compares the applicant information in the proposal against the grant's stated eligibility categories.

### Missing Requirements

Identifies information that is absent from the proposal, such as:

* Budget
* Prototype plan
* Technical methodology
* System architecture
* Problem statement
* Dataset or sensor plan
* Institution approval
* Applicant/team details

### Required Documents

Lists the documents required by the retrieved grants.

### Proposal-Funding Alignment

Explains how the research objective corresponds to each grant's stated objective.

### Suggested Proposal Improvements

Identifies specific additions that could make the proposal more complete relative to the retrieved grant requirements.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/dushyanth896-cpu/ResearchGrant-AI.git
```

Move into the project directory:

```bash
cd ResearchGrant-AI
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

You should see:

```text
(venv)
```

at the beginning of the terminal prompt.

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Gemini API Configuration

The application requires an API key for the configured Gemini model.

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit the `.env` file to GitHub.

The repository contains:

```text
.env.example
```

as a template.

Example:

```text
GEMINI_API_KEY=
```

Copy `.env.example` to `.env` and add your API key.

---

## Important Security Note

Never upload your actual API key to GitHub.

The `.gitignore` file excludes:

```text
.env
.env.*
!.env.example
```

If an API key has already been exposed publicly, revoke or rotate it immediately.

---

# Build the Grant Index

Before running the application, make sure the grant documents are present inside:

```text
data/
```

Then run:

```bash
python build_index.py
```

A successful run should produce:

```text
grant_index.pkl
```

---

# Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the address in a browser.

---

# Example Proposal

You can test the application with a proposal such as:

```text
We are developing an autonomous underwater robot for detecting
and collecting plastic waste from water bodies.

The system uses computer vision and AI-based plastic detection,
underwater robotics, sensors, motors, and an embedded controller.

The project is being developed by undergraduate engineering
students at an academic institution.
```

The system should retrieve grants related to areas such as:

* Robotics
* Environmental technology
* Plastic waste management
* Water pollution
* AI-based environmental monitoring
* Underwater robotics

---

# Testing

The project contains a test script:

```bash
python test_rag.py
```

This can be used to verify the RAG pipeline independently from the Streamlit interface.

---

# Evaluation

The project also contains:

```text
evaluate.py
```

Run:

```bash
python evaluate.py
```

This can be used for evaluating the retrieval or analysis pipeline depending on the configured evaluation implementation.

---

# Updating the Grant Database

To add a new grant:

### Step 1

Add a new JSON document inside:

```text
data/
```

For example:

```text
data/grant_10.json
```

### Step 2

Run:

```bash
python build_index.py
```

### Step 3

Restart Streamlit:

```bash
streamlit run app.py
```

The new grant will then be available to the retrieval system.

---

# RAG Pipeline

The complete pipeline is:

```text
Grant JSON Files
       │
       ▼
Document Ingestion
       │
       ▼
Text Embeddings
       │
       ▼
Vector Index
       │
       ▼
User Research Proposal
       │
       ▼
Proposal Embedding
       │
       ▼
Semantic Similarity Search
       │
       ▼
Top Relevant Grants
       │
       ▼
Context Construction
       │
       ▼
Gemini
       │
       ▼
Structured Proposal Analysis
       │
       ▼
Streamlit UI
```

---

# Key Files

## `app.py`

Provides the Streamlit user interface.

It handles:

* Proposal input
* Analysis button
* Loading results
* Displaying retrieved grants
* Displaying similarity scores
* Displaying AI-generated analysis

---

## `rag.py`

Contains the main Retrieval-Augmented Generation logic.

It handles:

* Loading the vector index
* Embedding the proposal
* Retrieving relevant grants
* Preparing context
* Calling the LLM
* Returning the analysis and retrieved sources

---

## `ingest.py`

Processes the source grant documents.

---

## `build_index.py`

Creates the vector index used for semantic retrieval.

---

## `evaluate.py`

Provides evaluation functionality for the system.

---

## `test_rag.py`

Tests the RAG pipeline.

---

## `grant_index.pkl`

Stores the generated grant embeddings and associated retrieval data.

This file is generated by:

```bash
python build_index.py
```

---

# Example Output

For a proposal involving an autonomous underwater robot for plastic detection, the system may retrieve grants such as:

```text
Green Robotics Innovation Grant
Ocean & Marine Environmental Innovation Grant
Smart Waste Management Innovation Grant
Clean Water Technology Grant
Environmental AI & IoT Grant
```

The application then analyzes:

```text
1. Potentially Relevant Funding Opportunities

2. Why They Match

3. Eligibility Requirements

4. Missing Requirements or Information

5. Required Documents

6. Proposal-Funding Alignment

7. Suggested Proposal Improvements

8. Evidence and Sources
```

---

# Limitations

ResearchGrant AI is an informational research-support tool.

It does not:

* Guarantee grant eligibility
* Guarantee funding approval
* Submit grant applications
* Replace official grant guidelines
* Make official funding decisions
* Guarantee that retrieved grants are currently accepting applications
* Verify the accuracy or current status of external funding programs

Users should verify eligibility, deadlines, funding amounts, application procedures, and required documents against the official grant provider before applying.

---

# API Quota

The AI analysis depends on the configured LLM API quota.

If the API provider reports a quota or rate-limit error, the retrieval system may still function, but the LLM-based analysis cannot be generated until:

* The quota resets
* A suitable API plan is enabled
* The configured model is changed
* Another supported LLM provider is configured

Repeatedly retrying a quota-exceeded request is not recommended.

---

# Privacy and Security

The project is designed to keep API credentials outside the source code.

Do not commit:

```text
.env
```

or any file containing API keys, passwords, tokens, or other credentials.

Before publishing the repository, check:

```bash
git status
```

and verify that `.env` is not listed.

---

# GitHub Deployment

To upload changes to GitHub:

```bash
git add .
git commit -m "Update ResearchGrant AI"
git push
```

The repository can then be accessed from the GitHub repository configured as the project's remote.

---

# Future Improvements

Potential future improvements include:

* Automatic grant deadline tracking
* Grant deadline notifications
* More grant sources
* PDF grant document ingestion
* Web-based grant discovery
* Multi-LLM provider support
* Better structured AI responses
* Grant comparison interface
* Proposal completeness scoring
* Applicant profile matching
* Grant application checklist generation
* Export analysis as PDF
* Export analysis as JSON
* Authentication and user accounts
* Database-backed grant storage
* Automatic grant database updates

---

# Project Status

Current implementation includes:

* Grant document database
* JSON-based grant storage
* Semantic embeddings
* Vector retrieval
* RAG pipeline
* LLM-based proposal analysis
* Eligibility analysis
* Missing requirement detection
* Required document analysis
* Proposal-funding alignment
* Streamlit web interface
* GitHub-ready project structure

---

# Disclaimer

ResearchGrant AI is intended for research and informational purposes.

The generated analysis should not be considered an official funding, eligibility, legal, or financial decision. Always verify the latest requirements and application information with the official grant provider.

---

# License

This project is currently provided for educational and research purposes.
