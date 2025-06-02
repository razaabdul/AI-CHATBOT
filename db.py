from pymongo import MongoClient
from datetime import datetime

# Connect to local MongoDB (or use a cloud URI like MongoDB Atlas)
client = MongoClient("mongodb://localhost:27017")
db = client["cococure_chatbot"]
collection = db["conversations"]

def save_conversation(wid, msg_id, user_message, bot_response):
    if is_message_processed(msg_id):
        print(f"🛑 Duplicate message detected: {msg_id}")
        return
    try:
        collection.insert_one({
            "wid": wid,
            "msg_id": msg_id,
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": datetime.utcnow()
        })
        print(f"✅ Conversation saved: {msg_id}")
    except Exception as e:
        print(f"❌ Error saving conversation: {e}")
def is_message_processed(msg_id: str) -> bool:
    try:
        existing = collection.find_one({"msg_id": msg_id})
        if existing:
            print(f"ℹ️ MongoDB: Message with msg_id={msg_id} already processed")
            return True
        return False
    except Exception as e:
        print(f"❌ MongoDB: Failed to check msg_id={msg_id}: {e}")
        return False  # safer to return False if error, so it tries to process

