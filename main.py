from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Sample input that mimics user preference from a travel app
destination_info = """
I'm planning a 3-day trip to Kyoto in April. I love historical places, cherry blossoms, and traditional food.
"""

# Template that the LLM will fill in to generate a custom itinerary
prompt = PromptTemplate(
    template=(
        "You are a travel assistant. Based on the user's preferences, create a 3-day itinerary that includes must-see attractions, "
        "cherry blossom viewing spots, and great places to try traditional Japanese food in {destination}.\n\n"
        "User preferences:\n{destination}\n\nItinerary:"
    ),
    input_variables=["destination"]
)

# Free-to-use Gemini model that supports chat-based responses
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0
)

# Connect the prompt to the model as a chain
chain = prompt | llm

# Run the model using the user input
response = chain.invoke({"destination": destination_info})

print("Generated Itinerary:\n")
print(response.content)
