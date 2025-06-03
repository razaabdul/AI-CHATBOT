README for CococureBot
CococureBot: A conversational assistant designed for handling booking, events, and customer support for Cococure venues.
Features
Event Booking Assistance:

Provides step-by-step guidance for booking tables at Cité and Haus.

Helps with event-specific reservations using relevant links.

Customer Support:

Handles queries about refunds, cancellations, and complaints empathetically.

Adheres to Cococure's cancellation and refund policies from the data.txt file.

Date and Event Parsing:

Parses user queries to identify dates or events using natural language processing.

Provides information based on context (restaurant vs. event venue).

Contextual Memory:

Remembers the conversation history for continuity and improved responses.

Stores past interactions using Redis for session-based memory.

Advanced Search:

Leverages FAISS indexing to retrieve relevant sections from FAQs or policy documents.

Generates embeddings using sentence-transformers for semantic similarity search.

Token Management:

Optimizes conversation lengths using OpenAI token counters to manage costs and ensure completeness.

Integration with Databases:

Uses MongoDB to store conversation logs for persistent history.

Redis for fast, temporary session data storage.

Installation
Dependencies
Ensure the following libraries are installed:

openai

faiss

numpy

re

sentence-transformers

langchain

redis

tiktoken

pymongo

python-dotenv

Install them using:

bash
Copy
Edit
pip install openai faiss numpy sentence-transformers langchain redis tiktoken pymongo python-dotenv
Setup
Environment Variables:
Create a .env file in the root directory with the following variables:

env
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
REDIS_URL=redis://localhost:6379
Database:

Set up a Redis server locally on localhost:6379.

Configure MongoDB for storing persistent chat histories.

Data Preparation:

Place FAQ or policy documents in the specified path (/home/brandon/Desktop/bot-integration/data.txt).

Run the Bot:
Run the main  -  python main.py
