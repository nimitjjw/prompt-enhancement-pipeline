import json
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Load test cases
with open("prompts.json") as f:
    prompt_data = json.load(f)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0
)

results = []

for item in prompt_data:
    prompt = PromptTemplate(
        template=item["prompt_template"],
        input_variables=["destination"]
    )

    # Attach prompt ID as a LangSmith tag for easier tracking
    # chain = prompt | llm.with_config(config={
    #     "tags": [f"prompt-{item['id']}"]
    # })
    chain = (prompt | llm).with_config(config={"tags": [f"prompt-{item['id']}"]})


    # Error handling/Test for pipeline failures
    try:
        response = chain.invoke({"destination": item["input"]})
        output = response.content
    except Exception as e:
        print(f"failed to process prompt {item['id']}: {e}")
        output = None

    print(f"\n--- Prompt ID: {item['id']} ---")
    print(output)

    results.append({
        "id": item["id"],
        "prompt_template": item["prompt_template"],
        "input": item["input"],
        "output": output,
        "timestamp": datetime.now().isoformat()
    })

# Save outputs for future evaluation
os.makedirs("results", exist_ok=True)

with open("results/outputs.json", "w") as f:
    json.dump(results, f, indent=2)
