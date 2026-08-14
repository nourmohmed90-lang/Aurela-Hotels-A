import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from planning.router import route_task

def get_gemini_llm(model_name: str | None = None):
    # Load environment variables
    # Check current dir, then agent/ directory
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / "agent" / ".env")
    
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY found in agent/.env")
        
    return ChatGoogleGenerativeAI(
        api_key=api_key,
        model=model_name,
        temperature=0.2
    )

def run_operations_planning(question: str):
    print(f"Running Hotel Operations Planning Agent for goal/question:")
    print(f"  '{question}'\n")
    
    llm = get_gemini_llm()
    result = route_task(question, llm)
    
    print("=" * 60)
    print(f"Selected Strategy: {result['strategy']}")
    print(f"Success Status:    {result.get('success')}")
    print("=" * 60)
    print("Output / Action Plan:")
    print(result["output"])
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aurelia Hotels Operations Agent Entry Point")
    parser.add_argument("question", nargs="?", default="Guest Sara Mohamed requests extra pillows and laundry service for reservation 2.", help="Hotel operations question or crisis scenario")
    args = parser.parse_args()
    
    run_operations_planning(args.question)
