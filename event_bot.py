import requests
import dateparser
from datetime import datetime, timedelta
import re
import redis
import json

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
CACHE_KEY = "cococure:events"
CACHE_TTL = 300  # Cache for 5 minutes (300 seconds)

API_URL = "https://cococure.com/wp-json/tribe/events/v1/events"



def fetch_all_events_cached():
    print("fetching event----------------")
    # Check Redis cache
    cached_data = r.get(CACHE_KEY)
    if cached_data:
        try:
            return json.loads(cached_data)
        except:
            pass  # ignore corrupted cache

    # Fetch from original API
    events = fetch_all_events()

    # Cache the result
    r.setex(CACHE_KEY, CACHE_TTL, json.dumps(events))
    return events

def fetch_all_events():
    events, page = [], 1
    while True:
        try:
            resp = requests.get(API_URL, params={"page": page})
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("events", [])
            if not batch:
                break
            events.extend(batch)
            if page >= data.get("total_pages", 1):
                break
            page += 1
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            break
    return events

# def extract_venue_name(user_input, venue_list, threshold=80):
#     match, score, _ = process.extractOne(user_input, venue_list)
#     return match if score > threshold else None
def format_event(e):
    title = e.get("title", "No Title")
    slug = e.get("slug", "No Slug")
    venue = e.get("venue", {})
    venue_name = venue.get("venue", "Unknown Venue") if isinstance(venue, dict) else "Unknown Venue"
    start_date_str = e.get("start_date", "")
    end_date_str = e.get("end_date", "")

    try:
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d %H:%M:%S")

        dt = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")
        date_part = dt.strftime("%A, %d %B %Y")
        time_part = dt.strftime("%I:%M %p")  # 12-hour format with AM/PM
    except Exception:
        date_part, time_part = start_date_str, ""
    return f"""🎉 {title} (slug: {slug}) at {venue_name} on {date_part} from {time_part}  to {end_dt}
    👉 [Click here to book your table](https://cococure.com/event/{slug})"""

def to_str(val):
    if isinstance(val, str):
        return val.lower()
    elif isinstance(val, dict):
        return to_str(val.get('name') or val.get('title') or val.get('venue') or "")
    elif isinstance(val, list):
        return ' '.join(to_str(x) for x in val)
    else:
        return str(val)
def filter_events_by_date(events, start_date, end_date=None):
    end_date = end_date or start_date
    matched = []
    now = datetime.now()

    for e in events:
        try:
            event_dt = datetime.strptime(e.get("start_date", ""), "%Y-%m-%d %H:%M:%S")
            if start_date <= event_dt.date() <= end_date and event_dt > now:
                matched.append(e)  # ⬅️ return raw event dict, not formatted string
        except:
            continue
    return matched

def parse_natural_date(text):
    settings = {"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False, "RELATIVE_BASE": datetime.now()}
    text_lower = text.lower()
    date = dateparser.parse(text)
    if "today" in text.lower() :
        return datetime.now().date()
    if date:
        return date.date()
    parsed = dateparser.parse(text, settings=settings)
    if parsed:
        return parsed.date()
    
    m = re.search(r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b', text, re.I)
    if m:
        parsed = dateparser.parse(m.group(0), settings=settings)
        if parsed:
            return parsed.date()
    
    weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, day in enumerate(weekdays):
        if day in text_lower:
            today = datetime.now()
            delta = (i - today.weekday() + 7) % 7
            if any(w in text_lower for w in ["next", "coming"]):
                delta += 7
            elif "this" in text_lower and delta == 0:
                pass
            elif delta == 0:
                delta = 7
            return (today + timedelta(days=delta)).date()
    return None
def format_event_with_booking(event):
    title = event.get("title", "No title")
    start_str = event.get("start_date", "")
    end_str = event.get("end_date", "")

    venue = event.get("venue", {})
    venue_name = venue.get("venue", "Unknown Venue") if isinstance(venue, dict) else str(venue)
    
    slug = event.get("slug", "")
    try:
        start_dt = datetime.fromisoformat(start_str)
        end_dt = datetime.fromisoformat(end_str) if end_str else None

        date_part = start_dt.strftime("%A, %d %B %Y")
        start_time = start_dt.strftime("%I:%M %p")
        end_time = end_dt.strftime("%I:%M %p") if end_dt else ""

        if end_time:
            formatted_time = f"{date_part}, {start_time} – {end_time}"
        else:
            formatted_time = f"{date_part}, {start_time}"
    except Exception as e:
        formatted_time = "Date/time unavailable"
    booking_link = f"https://cococure.com/events/{slug}?booking=table" if slug else "Booking link unavailable"

    return (
        f"🎉 **{title}** at *{venue_name}*\n"
        f"📅 Date & Time: {formatted_time}\n"
        f"🔗 [Book a Table]({booking_link})"
    )

def process_query(query):
    query_lower = query.lower()
    events = fetch_all_events_cached()

    date = parse_natural_date(query)
    

    for event in events:
        event_title = to_str(event.get("title", ""))
        venue = to_str(event.get("venue", ""))
        event_date_str = event.get("datetime", "")
     
        try:
            event_date = datetime.fromisoformat(event_date_str).date()
        except:
            continue
        if event_title in query_lower or venue in query_lower:
                if date is None or event_date == date:
                    return format_event_with_booking(event)
    if "weekend" in query_lower:
        today = datetime.now().date()
        saturday = today + timedelta((5 - today.weekday()) % 7)
        sunday = saturday + timedelta(days=1)
        matched = filter_events_by_date(events, saturday, sunday)
        if matched:
            event_texts = [format_event_with_booking(e) for e in matched]
            return "🎉 **Here are the available events this weekend:**\n\n" + "\n\n".join(event_texts)
        else:
            return "❌ No events found this weekend."
        # return "\n\n".join([format_event_with_booking(e) for e in matched]) if matched else "❌ No events found this weekend."

    if "this week" in query_lower:
        return "Sorry, I can't check events by week yet. Please specify a particular date."

    date = parse_natural_date(query)
    if date:
        today = datetime.now().date()

        matched = filter_events_by_date(events, date)
        
        if matched:
            event_texts = [format_event_with_booking(e) for e in matched]
            return f"🎉 **Here are the events on {date.strftime('%A, %d %B %Y')}:**\n\n" + "\n\n".join(event_texts)
        else:
            return  "Sorry there is no event found on this data ! "

    return "Sorry AI bot couldn't understand . If you have any other query feel free to ask ! "

if __name__ == "__main__":
    print("Ask about events (type 'exit' or 'quit' to stop).")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", process_query(q))
