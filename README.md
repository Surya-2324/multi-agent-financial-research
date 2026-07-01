# multi-agent-financial-research
Multi-agent financial research system using LangGraph, Groq, ChromaDB, MLflow — production agentic AI with governance

## Railway environment variables

The AI analysis endpoint uses Groq via `langchain-groq`. Set `GROQ_API_KEY`
in Railway to a valid Groq API key, then redeploy or restart the service. A
missing, expired, or non-Groq key will make `/api/analyse` return a 401
`invalid_api_key` error.
