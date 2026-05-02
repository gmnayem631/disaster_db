import os
import uuid
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to MongoDB
def get_database():
    client = MongoClient(os.getenv("MONGODB_URI"))
    return client["disaster_db"]

# Insert disaster record
def insert_disaster_record(record):
    db = get_database()
    collection = db["disaster_events"]

    # Duplicate check by source_url
    existing = collection.find_one({"source_url": record.get("source_url")})
    if existing:
        return {"status": "duplicate", "id": str(existing["_id"])}

    # Add metadata
    record["event_id"]            = str(uuid.uuid4())
    record["extraction_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result = collection.insert_one(record)
    return {"status": "inserted", "id": str(result.inserted_id)}

# Test connection
def test_connection():
    try:
        db = get_database()
        db.command("ping")
        print("MongoDB connection successful")
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()