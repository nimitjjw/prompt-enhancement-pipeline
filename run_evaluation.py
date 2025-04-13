import json
import os
import numpy as np
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (API keys, LangSmith settings, etc.)
load_dotenv()

# Load evaluation dataset from eval_dataset.json
with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)
print("Loaded evaluation dataset with", len(eval_data), "inputs.")

# Load prompt versions from prompts.json
with open("prompts.json", "r") as f:
    prompt_versions = json.load(f)
print("Loaded", len(prompt_versions), "prompt versions.")

# Initialize the embedding model (using a pre-trained model from sentence-transformers)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Ensure you've installed "sentence-transformers"

def get_embedding(text):
    """Return the embedding vector for a given text."""
    return embedding_model.encode([text])[0]

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def run_prompt(prompt_template, user_input):
    """
    Build and run a prompt chain for a given prompt template and user input.
    Returns the generated output from the Gemini model.
    """
    prompt_obj = PromptTemplate(
        template=prompt_template,
        input_variables=["destination"]
    )
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0
    )
    chain = (prompt_obj | llm)
    try:
        response = chain.invoke({"destination": user_input})
        return response.content
    except Exception as e:
        return f"error: {e}"

def automated_evaluate(output, reference):
    """
    Compute a semantic similarity score between the generated output and a reference text.
    The score is based on cosine similarity of their embedding vectors.
    Returns a score between 0 and 1 as a native Python float.
    """
    try:
        embed_output = get_embedding(output)
        embed_reference = get_embedding(reference)
        score = cosine_sim(embed_output, embed_reference)
        # Convert to a native Python float to avoid JSON serialization issues.
        return float(round(score, 2))
    except Exception:
        return 0.0

evaluation_results = []

# Loop through each input in the evaluation dataset and each prompt version
for input_case in eval_data:
    user_input = input_case["destination"]
    reference_text = input_case.get("reference", "")
    for prompt_version in prompt_versions:
        version_id = prompt_version["id"]
        prompt_template = prompt_version["prompt_template"]
        
        # Generate output for the given prompt version and input
        generated_output = run_prompt(prompt_template, user_input)
        # Evaluate the output using cosine similarity between embeddings
        score = automated_evaluate(generated_output, reference_text)
        
        evaluation_record = {
            "prompt_version": version_id,
            "user_input": user_input,
            "reference": reference_text,
            "generated_output": generated_output,
            "automated_score": score,
            "timestamp": datetime.now().isoformat()
        }
        evaluation_results.append(evaluation_record)
        print(f"Prompt {version_id} with input '{user_input[:30]}...' scored {score}")

# Save all evaluation records into a JSON file
os.makedirs("results", exist_ok=True)
with open("results/evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)

print("Automated evaluation complete. Results saved to results/evaluation_results.json")
