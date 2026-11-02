# 🛡️ FinGuard — AI-Powered Financial Response Compliance Classifier

**Detect when AI-generated financial responses cross the boundary from informational content into financial advice.**

FinGuard is a full-stack AI-powered application designed to identify the regulatory grey zone in financial conversations. It analyzes financial statements and classifies them into three categories — **Informational, Advisory, and Uncertain** — while providing linguistic explanations for its decisions.

The project addresses an important challenge in AI-driven financial services: distinguishing between providing general financial information and making recommendations that could be interpreted as personalized financial advice.

## 🚀 Live Application

- **Frontend:** https://fin-guard-ten.vercel.app
- **Backend API:** https://finguard-p45t.onrender.com
- **API Health Check:** https://finguard-p45t.onrender.com/api/health
- **API Documentation:** https://finguard-p45t.onrender.com/docs

## 🎯 Problem Statement

AI-generated financial responses can gradually move from neutral explanations to interpretations, recommendations, and prescriptive advice.

For example:

- **Informational:** "Bond prices generally fall when interest rates rise."
- **Advisory:** "Investors may consider reducing their exposure to long-duration bonds."
- **Uncertain:** A statement whose intent or level of recommendation is ambiguous without additional context.

The boundary between these categories is not always clear. FinGuard aims to identify this ambiguity, explain the linguistic signals behind a classification, and help reformulate potentially advisory responses into safer informational language.

## ✨ Key Features

- **AI-Powered Classification:** Analyzes financial text and assigns one of three categories: Informational, Advisory, or Uncertain.
- **Linguistic Explanations:** Provides reasoning behind the classification instead of returning only a label.
- **Grey-Zone Detection:** Identifies statements where the distinction between information and advice may be ambiguous.
- **Safer Reformulation:** Provides suggestions to make potentially advisory language more neutral and informational.
- **Interactive Web Interface:** Allows users to submit financial statements and view classification results.
- **Classification History:** Maintains previous results in browser local storage.
- **REST API:** Exposes the classification functionality through a FastAPI backend.
- **Cloud Deployment:** Frontend and backend are deployed independently for accessibility.

## 🏗️ System Architecture

```text
              USER
               |
               v
       React Web Interface
          (Vite Frontend)
               |
               | HTTP Request
               v
        FastAPI Backend
               |
               v
       Classification Logic
               |
               v
       Prompt Construction
               |
               v
          Groq API
        (LLM Inference)
               |
               v
      Structured AI Response
               |
               v
     Classification + Explanation
               |
               v
       React Web Interface
```

### Application Workflow

1. The user enters a financial statement through the web interface.
2. The React frontend sends the input to the FastAPI classification endpoint.
3. The backend constructs a structured prompt containing the classification instructions and decision rules.
4. The request is sent to the Groq API using the configured language model.
5. The model generates a structured classification response.
6. The backend returns the result to the frontend.
7. The interface displays the category, linguistic explanation, and safer reformulation where provided.

## 🧠 Classification Framework

| Category | Description |
|---|---|
| Informational | Neutral financial facts, definitions, and general explanations. |
| Advisory | Recommendations, personalized interpretations, or prescriptive financial guidance. |
| Uncertain | Ambiguous statements that cannot be confidently assigned to either category. |

The classification framework is designed to recognize linguistic indicators such as:

- Recommendation-oriented language
- Prescriptive expressions
- Personalized financial suggestions
- Interpretative or evaluative statements
- Ambiguous wording and contextual uncertainty

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React.js, Vite, JavaScript, CSS |
| Backend | Python, FastAPI |
| AI Inference | Groq API |
| Language Model | `openai/gpt-oss-120b` |
| API Communication | HTTP / REST |
| Deployment | Vercel, Render |
| Version Control | Git, GitHub |

## 📂 Project Structure

```text
FinGuard/
│
├── frontend/
│   ├── src/
│   │   └── App.jsx
│   ├── package.json
│   └── ...
│
├── src/
│   ├── api.py
│   ├── inspect_validation.py
│   └── validate_results.py
│
├── data/
│   └── processed/
│       └── finguard_dataset.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: This is a high-level representation of the project structure. Additional files and folders may be present in the repository.*

## ⚙️ Local Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Nitisha-27/FinGuard.git
cd FinGuard
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit API keys or expose secret credentials in frontend code.

### 5. Start the Backend

From the project root:

```bash
uvicorn src.api:app --reload
```

The backend will run at:

`http://127.0.0.1:8000`

Access interactive API documentation at:

`http://127.0.0.1:8000/docs`

### 6. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
```

Create a frontend `.env` file:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Start the development server:

```bash
npm run dev
```

The frontend will be available at:

`http://localhost:5173`

## 🌐 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Checks backend availability. |
| POST | `/api/classify` | Classifies a financial statement. |

Interactive API documentation:

https://finguard-p45t.onrender.com/docs

The Swagger interface provides the request schema and allows the classification endpoint to be tested directly.

## ☁️ Deployment

FinGuard uses separate cloud deployments for its frontend and backend.

| Layer | Platform | Deployment |
|---|---|---|
| Frontend | Vercel | React + Vite |
| Backend | Render | FastAPI + Uvicorn |
| AI inference | Groq API | Hosted language model |

### Frontend Configuration

The frontend uses the following environment variable:

```env
VITE_API_BASE_URL=https://finguard-p45t.onrender.com
```

### Backend Configuration

The backend requires:

```env
GROQ_API_KEY=your_groq_api_key
```

The Render service starts using:

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

Cross-Origin Resource Sharing (CORS) is configured to allow requests from the deployed frontend and the local development environment.

## 🔬 Dataset and Validation

FinGuard also includes a dataset-development and validation component.

The project dataset was organized around three classification labels:

- Informational
- Advisory
- Uncertain

The dataset-development process included reviewing source references, checking duplicate examples, and examining near-duplicate text.

Validation scripts are maintained separately from the deployed inference workflow.

**Important:** Dataset validation and live LLM inference are distinct parts of the project. The deployed application currently performs inference through the Groq API; it does not directly load the CSV dataset for every classification request.

## ⚠️ Limitations and Responsible Use

- FinGuard is an experimental compliance-support tool, not a certified regulatory compliance system.
- Classification results are AI-generated and may contain errors.
- Ambiguous financial language may require human review.
- The application does not replace legal, regulatory, or compliance professionals.
- The current application does not provide a persistent server-side database for classification history.
- The application should not be used as a substitute for professional financial advice.

## 🔮 Future Improvements

- Build a larger, independently reviewed financial compliance dataset.
- Introduce multi-annotator evaluation to measure human disagreement.
- Evaluate classification performance using precision, recall, F1-score, and confusion matrices.
- Add confidence calibration and uncertainty-aware decision thresholds.
- Explore retrieval-augmented generation using verified regulatory documents.
- Add persistent storage and audit logs.
- Introduce automated testing and continuous integration.
- Evaluate the system across different financial domains and language models.

## 👩‍💻 Author

**Nitisha Mishra**

B.Tech — Electronics and Communication Engineering (Biomedical Specialization)  
VIT Vellore

GitHub: https://github.com/Nitisha-27

---

*FinGuard — Making AI-generated financial communication more transparent, explainable, and compliance-aware.*