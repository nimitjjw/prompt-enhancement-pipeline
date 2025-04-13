import json
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Load evaluation dataset from eval_dataset.json
with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)
print("Loaded evaluation dataset with", len(eval_data), "inputs.")

# Load prompt versions from prompts.json
with open("prompts.json", "r") as f:
    prompt_versions = json.load(f)
print("Loaded", len(prompt_versions), "prompt versions.")

def run_prompt(prompt_template, user_input):
    """
    Runs a given prompt version with a specific user input and returns the generated output.
    """
    # Build a prompt object using the provided template
    prompt_obj = PromptTemplate(
        template=prompt_template,
        input_variables=["destination"]
    )
    # Initialize the Gemini model using ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0
    )
    # Build the execution chain
    chain = (prompt_obj | llm)
    try:
        response = chain.invoke({"destination": user_input})
        return response.content
    except Exception as e:
        return f"error: {e}"

def automated_evaluate(output, reference="A helpful and detailed travel itinerary."):
    """
    Simulates an automated evaluation by calculating a simple similarity score based on output length.
    Replace this logic with a more advanced evaluator from LangSmith if needed.
    """
    try:
        similarity = min(len(output), len(reference)) / max(len(output), len(reference))
        return round(similarity, 2)
    except Exception:
        return 0.0

evaluation_results = []

# Loop over each input in the evaluation dataset and each prompt version
for input_case in eval_data:
    user_input = input_case["destination"]
    for prompt_version in prompt_versions:
        version_id = prompt_version["id"]
        prompt_template = prompt_version["prompt_template"]
        
        # Run the prompt version using the helper function
        generated_output = run_prompt(prompt_template, user_input)
        # Get an automated score comparing the generated output to a reference
        score = automated_evaluate(generated_output)
        
        # Build an evaluation record for this run
        evaluation_record = {
            "prompt_version": version_id,
            "user_input": user_input,
            "generated_output": generated_output,
            "automated_score": score,
            "timestamp": datetime.now().isoformat()
        }
        evaluation_results.append(evaluation_record)
        print(f"Prompt {version_id} with input '{user_input[:30]}...' scored {score}")

# Save the evaluation results to a file
os.makedirs("results", exist_ok=True)
with open("results/evaluation_results.json", "w") as f:
    json.dump(evaluation_results, f, indent=2)

print("Automated evaluation complete. Results saved to results/evaluation_results.json")
