
import openai
import faiss
import numpy as np
import re
import tiktoken
import os 
from datetime import datetime
import redis
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
# from datetime import datetime

# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# import requests
from db import save_conversation, is_message_processed, collection

from langchain_openai import ChatOpenAI

from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from event_bot import process_query, parse_natural_date  # if in another file
from langchain_core.messages.utils import trim_messages
import tiktoken
from langchain_core.messages import get_buffer_string


from langchain_community.chat_message_histories import RedisChatMessageHistory

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

api_key = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=api_key)  # Replace with your actual key

EXPIRY_SECONDS = 300  # 5 minutes
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


def load_faq_content():
    print("load_faq_content----------------")
    with open(r"/home/abdul/Desktop/bot-integration/data.txt", "r", encoding="utf-8") as file:
        return file.read()
replied_users = set()
def split_document(doc, chunk_size=500):
    print("split_document-------------")
    sections = re.split(r'(\n|^)[A-Za-z0-9\!\?\.\-\(\)\&]+[\:\-]*', doc)  # Split by headings or sections
    chunks = []
    current_chunk = ""
    for section in sections:
        if len(current_chunk) + len(section) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += "\n" + section
    # print(chunks,'-----------')
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def get_embeddings(texts):
    print("get_embeddings-----------------")
    # converting text into vector .
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return np.array(embeddings)

def create_faiss_index(embeddings):
    print("create_faiss_index-----------------")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_faiss_index(query, faiss_index, top_k=5):
    print("query_faiss_index---------")
    query_embedding = get_embeddings([query])[0]
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    return I[0]


llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def openai_token_counter(messages) -> int:
    print("openai_token_counter---------")

    # Convert messages to a single string
    buffer_string = get_buffer_string(messages)

    # Use the correct tokenizer
    encoding = tiktoken.encoding_for_model("gpt-4")  # or gpt-3.5-turbo if you're using that
    tokens = encoding.encode(buffer_string)
    return len(tokens)
def generate_response(wid, query,token_counter, relevant_chunks,history=None):
    print("generate_response-------------")
    if history is None:
        history =load_chat_history(wid)
    date = parse_natural_date(query)
    if date or "weekend" in query.lower():
        return process_query(query)

    if history:
        history = trim_messages(history, max_tokens=1000, token_counter=openai_token_counter)

   

    context = "\n\n".join(relevant_chunks) if relevant_chunks else ""

   
    prompt = f"""


# You are CococureBot, an assistant for Cococure.com. Help users book tables, events, and answer questions.

Follow these rules:

# If the user asks about booking a table or making a reservation and does NOT mention a specific event:
Respond with the **restaurant reservation process** and give both Cité and Haus options:

📍 Cité: https://cococure.com/cite-reservations/
📍 Haus: https://cococure.com/haus-reservations/

**Steps:**
1. Redirect to website
2. Select number of guests
3. Select time and check seating
4. Choose a day and time slot
5. Read policies
6. (Optional) Upgrade reservation
7. Fill personal info and agree to terms
8. Make payment to confirm  
⚠️ Max booking is 20 guests. Contact us if more.

# If the user asks about a specific event (e.g., Afrobeats Fridays, Twnty7):
Give the **event reservation link** and these steps:

1. Go to the event page
2. Select date
3. Choose ticket
4. Click “Get Tickets”
5. Fill info and billing address
6. Complete payment

Answer clearly and guide them based on what they’re asking.




# You are a polite and helpful assistant for Cococure, providing accurate information about bookings, cancellations, and refunds based on the user's question.

    You are a polite and helpful assistant for Cococure.

    1. Only provide event listings or say "No events found" if the user **explicitly asks** about events, parties, shows, or dates.
    2. If the user asks about anything else (like refunds, cancellations, complaints, or bad experiences), respond helpfully, empathetically, and do not mention event availability or dates.
    3. Always include contact details for follow-up in refund or cancellation cases:
    📧 Email: send@cococure.com  
    📞 Phone: +44 20 3983 3790  
    🔗 Terms: https://cococure.com/terms-conditions/

    Never say “No events found” or refer to the process_query() in event_bot.py  unless the user clearly asked for a date or event info.




# You are a polite and helpful assistant for Cococure, providing accurate information about bookings, complaints, cancellations, or refunds based on the user's query of yesterday ,tomorrow ,day tonight, after tomorrow response from llm model.


# Use the information from Cococure's policy document to respond clearly.



# You are a smart and polite assistant for Cococure. help user if user ask about refund or cancellations, or refunds refer them the data.txt * Cancellation & Refund Policy *: 
 

Always identify the **user's intent** first before responding. You MUST detect when the user is asking about cancellations, refunds, or booking issues — even if they mention dates or events.

Do NOT check for events if the user's question is about:
- a refund,
- canceling a ticket or reservation,
- changing a booking,
- event cancellation,
- or payment issues.



Instructions:
- Match the query context (restaurant or club).
- If it’s about **cancellation**, explain the required time frame or if cancellation is not possible.
- If it’s about **refunds**, check if the situation qualifies or not.
- If it's about **no entry** or **mistake**, list the relevant non-refundable conditions.
- Always include contact details and terms link for escalation.
- Be empathetic and professional, even when saying no.

# Only show the event that matches BOTH the user's requested venue name and date. 
# Return only one event. Do not list multiple events.
If no match, reply with: ❌ No event found for that venue on that date.


You are an AI assistant for Cococure. A user wants to **book a table** for an event on a specific date.


# Only return the event(s) that match BOTH:
- The venue name or event title mentioned by the user.
- The requested date.

# Do NOT list all events on that date. Return only the matching one(s).
If no match, say "No event found for that venue on that date."

Steps:

1. Parse the date from the user message using `parse_natural_date(message)`.

2. Use the parsed date to call the API via `fetch_event_data(date)` and retrieve events happening **on that specific date only** and show the booking table link for specifc event .

3. If events exist for that date, respond with this format:


# Never guess or make up details. If something is missing (e.g., specific time/day not in the opening hours), politely inform the user or ask for clarification.



# If a user asks about celebrating a birthday at Cococure , refer to the  *Looking to celebrate your birthday in style at Cococure?* in data.txt. Show all list of birthday events . Explain the steps clearly based on whether it's a restaurant (Haus/Cite) or a club event or any specific event name  .

If the user mentions uploading an ID, verifying identity, or completing verification, respond with a polite message and provide the ID verification link 🔗 https://cococure.com/id-verify


For restaurant bookings, direct them to Cococure Restaurant Bookings.

For club/event table bookings, direct them to the Cococure Events Page where they can select an event and book a table accordingly.
# Use only the following labeled links when referring to external resources refere them to the 11. Links of events :
 in data.txt. Do not use markdown links—just mention the label as-is:


# When responding:
# - Confirm if the requested booking date and time are available based on opening hours. If not, let the user know.
# - Clearly explain the reservation and booking process, step by step for both restaurant and club.
# - For any club reservation inquiries, inform the user that Cococure hosts events at **Cité**, **Haus**, and **Twnty7** clubs. Direct them to the respective club info link listed above.
# - If the user asks about cancellation policies for the club, prioritize relevant information from the **Club Policies** section.
# - If the user asks about **dress code**, **door policy**, or **entry rules**, refer to the **Style Guide and Club Door Policy** section in the context.
# - If the user asks about **age limit**, inform them that entry is strictly **18+ only**, and a valid ID is required.
# - If the user asks about the **menu** or **bottle menu**, refer to the **Bottle Menu** link above.
# - Never refer to days/times not listed in the opening hours (e.g., do not allow bookings on Mondays if not open).
# - If the user’s request is ambiguous or unsupported by context, ask a clarifying question.
# - Walk-ins are welcome, but we advise checking online for availability to avoid disappointment.
# - **Seating Inquiries (Gallery/Visuals)**:  
#    If the user asks how the seating looks, venue layout, space, or table arrangements for any location — do not guess. Instead, refer to the relevant gallery links 
    for twnty7-https://cococure.com/twnty7-stratford-gallery/

for haus-https://cococure.com/cococure-haus-stratford-gallery/

for cite-https://cococure.com/cococure-aldgate-gallery/

# - **Menu Queries (Food, Bottle, Off-Peak)**:  
#    If the user asks about food or bottle menus for any venue, provide the appropriate menu link from the knowledge base.

# - **Brunch/Drunch**:  
#    If the user asks about brunch or drunch options, guide them to the relevant brunch information pages based on the venue.



# When a user asks about parking availability, parking facilities, or parking options for Cococure venues or events, respond like this:

"Currently, there is no dedicated parking available at Cococure venues. We recommend using nearby public parking facilities or public transportation to reach the venue. 


You are an assistant helping users find information about events from a structured event database. 

You are an assistant helping users find information about events from the data below.

The event data is organized by numbered events, each with:

- Event Title
- Date & Time
- Venue Name
- Address
- Description (short summary)

se only the provided event data to answer the user's query. Do NOT generate any information that is not present.

If the query is about a specific event or date, List all events on the specified date, numbering them clearly. Do not assume the number of events; instead, base your response entirely on the data.
.

Provide your response in a clear, concise, and informative manner.

# Note: All details and links related to these topics are available in the document chunks and should be referenced directly when relevant.

# Always prioritize being helpful, factual, and grounded in the context.

Context:
{context}

Conversation History:
Use the conversation history below to maintain continuity.


"""

    messages = [{"role": "system", "content": prompt}]

    for msg in history:
        if isinstance(msg, dict):
            messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages.append({"role": role, "content": msg.content})

    # Step 6: Add current user query
    messages.append({"role": "user", "content": query})

  

    # Step 7: Call OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
        max_tokens=800
    )

    # Step 8: Save conversation to DB
    response_text = response.choices[0].message.content


    return response_text

def cococure_bot(query, faq_content):
    print("cococure_bot------------------")
    chunks = split_document(faq_content)

    embeddings = get_embeddings(chunks)

    faiss_index = create_faiss_index(embeddings)
    
    relevant_indices = query_faiss_index(query, faiss_index, top_k=5)

    relevant_chunks = [chunks[i] for i in relevant_indices]


    response = generate_response(query, relevant_chunks)
    print(len(faiss_index.index))
    return response
class CococureBotWithHistory:
    def __init__(self, faq_content, wid, ):
        self.wid = wid
        self.faq_content = faq_content
        self.chunks = split_document(faq_content)
        self.embeddings = get_embeddings(self.chunks)
        self.faiss_index = create_faiss_index(self.embeddings)
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

        self.llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o", openai_api_key="sk-...")

        print(wid,'--------------------idd')
        chat_history = RedisChatMessageHistory(
            url="redis://localhost:6379",
            session_id=wid
        )

        # ✅ Only add old messages if Redis is empty
        redis_key = f"langchain:chat_history:{wid}"
        if not self.redis_client.exists(redis_key):
            past_messages = load_chat_history(wid)

            formatted_messages = []
            for msg in past_messages:
                if msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AIMessage(content=msg["content"]))

            chat_history.add_messages(formatted_messages)
            print("cha-historyy ----------------------------")

        self.memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True
        )

        self.previous_query = None
        self.previous_topic_vector = None

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    def count_message_tokens(messages, model="gpt-4o"):
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message overhead
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
        num_tokens += 2  # assistant priming
        return num_tokens
    def trim_messages(self, messages, max_tokens=3000):
        while self.count_message_tokens(messages) > max_tokens and len(messages) > 1:
            messages.pop(0)
        return messages

    def is_same_topic(self, query):
        if self.previous_topic_vector is None:
            return False
        query_vec = get_embeddings([query])[0]
        similarity = np.dot(query_vec, self.previous_topic_vector) / (
            np.linalg.norm(query_vec) * np.linalg.norm(self.previous_topic_vector)
        )
        return similarity > 0.85
    
    async def answer(self, query, msg_id, inputs):
        if is_message_processed(msg_id):
            return {"response": "Message already processed."}
        
        query_text = inputs["user_message"]

        if not self.is_same_topic(query):
            self.memory.clear()

        if "date" in query_text.lower():
            print("called")
            response = await self.process_query(query)
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response)
            save_conversation(self.wid, msg_id, query, response)
        
           
            return {"response": response}
        
         # Step 2: Load memory or fallback to MongoDB
        if self.memory.chat_memory.messages:
            history = []
            for msg in self.memory.chat_memory.messages[-10:]:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                history.append({"role": role, "content": msg.content})
        else:
            history = load_chat_history(self.wid, limit=10)

        # Step 3: Vector Search (FAISS)


        relevant_indices = query_faiss_index(query, self.faiss_index, top_k=5)
        relevant_chunks = [self.chunks[i] for i in relevant_indices]
        relevant_chunks = filter_chunks_by_date(relevant_chunks, query)
        context = "\n\n".join(relevant_chunks)

       

        response = generate_response(
            self.wid,
            query,
            relevant_chunks=relevant_chunks,
            token_counter=self.count_message_tokens,
            history=history
        )
                # Step 5: Add user message to memory and redis
        self.memory.chat_memory.add_user_message(query)

        redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # redis_client.expire(f"message_store:{self.wid}", 3600)  # expires in 1 hour

        self.memory.chat_memory.add_ai_message(response)
        redis_client.expire(f"message_store:{self.wid}", 3600)

        # Step 6: Update previous query/vector and save to DB

        self.previous_query = query
        self.previous_topic_vector = get_embeddings([query])[0]

        save_conversation(self.wid, msg_id, query, response)

        return {"response": response}   

def load_chat_history(wid, limit=10):
    print("loadddddds")
    # Fetch last 'limit' conversations from DB ordered by timestamp ascending (oldest first)
    chats = collection.find({"wid": wid}).sort("timestamp", 1).limit(limit)
    messages = []
    for chat in chats:
        # Add user message
        messages.append({"role": "user", "content": chat["user_message"]})
        # Add bot response
        messages.append({"role": "assistant", "content": chat["bot_response"]})
    return messages

bot_instances = {}
def get_bot_instance(wid):
    if wid not in bot_instances:
        faq_content = load_faq_content()  # Load it here
        bot = CococureBotWithHistory(faq_content, wid)
        bot_instances[wid] = bot
    return bot_instances[wid]
async def handle_user_message(wid, query, msg_id):
    bot = get_bot_instance(wid)
    inputs = {"user_message": query}  # <-- fix here
    response = await bot.answer(query, msg_id, inputs)
    return response

def filter_chunks_by_date(chunks, query):
    # e.g., extract 29 May from query
    pattern = re.search(r'\b(\d{1,2})\s+(may|june|july|august)\b', query, re.IGNORECASE)
    if not pattern:
        return chunks[:10]  # fallback to first 10

    day, month = pattern.groups()
    date_str = f"{int(day):02d} {month.lower()}"
    return [chunk for chunk in chunks if date_str in chunk.lower()]




import uuid
import asyncio


if __name__ == "__main__":
    async def main():
        wid = "newuser"
        faq_content = load_faq_content()

        # Initialize the bot (handles Redis & Mongo logic inside)
        bot = CococureBotWithHistory(faq_content, wid)

        print("🤖 CococureBot ready!\n")

        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                break

            msg_id = str(uuid.uuid4())  # Unique ID
            inputs = {"user_message": query}

            # Get response
            response = await bot.answer(query, msg_id, inputs)
            print("Bot:", response)

    asyncio.run(main())







# wati_api_key = os.getenv("WATI_API_KEY")
# wati_base_url = "https://live-mt-server.wati.io/101643/api/v1"
# AUTO_REPLY_TEXT = "Thanks for your message! We'll get back to you shortly."

# # Cache message ID with expiry
# processed_message_ids = {}  # Format: { message_id: expiry_time }


# @app.post("/wati/webhook")
# async def wati_incoming_message(request: Request):
#     try:
#         data = await request.json()
#         print(data)
#     except Exception as e:
#         return JSONResponse(content={"error": "Invalid JSON"}, status_code=400)
#     if data.get("eventType") in ["sessionMessageSent", "sessionMessageSent_v2"]:
#         print(f"🟡 Ignored system message: {data.get('text')}")
#         return JSONResponse(content={"message": "Ignored system event"}, status_code=200)

#     #sender_phone = data.get("waId")
#     sender_phone = 16073739682
#     message_body = data.get("text", "").strip()
#     message_id=data.get("id")
    

#     print(f"📨 Incoming message from {sender_phone}: {message_body}")

#     if not sender_phone or not message_body:
#         return JSONResponse(content={"message": "Missing fields"}, status_code=400)
#     if data.get("owner", True):  # default True to skip if missing
#         return JSONResponse(content={"message": "Ignoring bot message"}, status_code=200)
    
#     if is_message_processed(message_id):
#         print(f"🛑 Skipping reply: message id {message_id} already processed")
#         return JSONResponse(content={"message": "Already processed message"}, status_code=200)

#     # Check if user already got a reply
#     if sender_phone in replied_users:
#         print(f"🛑 Skipping auto-reply to {sender_phone} (already replied)")
#         return JSONResponse(content={"message": "User already replied to"}, status_code=200)

#     # Send auto-reply
#     faq_content = load_faq_content()
    
# #     # Step 2: Initialize the bot
#     bot = CococureBotWithHistory(faq_content)
#     response = bot.answer(message_body)
# #     print(f"Bot: {response}\n")
#     save_conversation(sender_phone, message_id, message_body, response)
#     if sender_phone==16073739682:
        
#         send_whatsapp_message(sender_phone, response)
#         return JSONResponse(content={"message": "Auto-reply sent"}, status_code=200)


# def send_whatsapp_message(phone_number: str, message: str):
#     encoded_message = urllib.parse.quote(message)
#     send_url = f"{wati_base_url}/sendSessionMessage/{phone_number}?messageText={encoded_message}"
#     headers = {"Authorization": f"Bearer {wati_api_key}"}

#     try:
#         response = requests.post(send_url, headers=headers)
#         if response.status_code == 200:
#             print(f"✅ Auto-reply sent to {phone_number}")
#         else:
#             print(f"❌ Failed to send auto-reply. Status: {response.status_code}, Error: {response.text}")
#     except Exception as e:
#         print(f"🚨 Exception while sending auto-reply: {str(e)}")
