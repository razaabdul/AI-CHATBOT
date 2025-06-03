

import os
import requests
from typing import Optional
from dotenv import load_dotenv, find_dotenv
import dateparser  # for natural language date parsing

# Load environment variables (OpenAI key)
load_dotenv(find_dotenv())

# Step 1: Enhanced function to parse natural language dates and fetch events
def get_events_by_date(natural_date: Optional[str] = None) -> str:
    """
    Fetches Cococure events for a natural language date like '8 Jun'.
    Returns clean text: event name, venue, times, and booking link.
    """
    if not natural_date:
        return "Please provide a date like '8 Jun' or 'next Friday'."

    parsed_date = dateparser.parse(natural_date)
    if not parsed_date:
        return f"Could not understand the date: {natural_date}"

    date_str = parsed_date.strftime("%Y-%m-%d")
    url = f"https://cococure.com/wp-json/tribe/events/v1/events?start_date={date_str}&end_date={date_str}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Failed to fetch events: {response.status_code}"

    events = response.json().get('events', [])
    if not events:
        return f"No events found for {natural_date}."

    result = ""
    for event in events:
        name = event.get('title', 'Unnamed Event')
        start = event.get('start_date', 'Unknown Start')
        end = event.get('end_date', 'Unknown End')
        venue = event.get('venue', {}).get('venue', 'Unknown Venue')
        slug = event.get('slug', '')
        booking_link = f"https://cococure.com/event/{slug}/" if slug else "Link not available"
        
        result += (
            f"\n🎉 *{name}*\n"
            f"📍 Venue: {venue}\n"
            f"🕒 Start: {start}\n"
            f"🕔 End: {end}\n"
            f"🔗 Book here: {booking_link}\n"
            f"{'-'*40}"
        )

    return result.strip()

# Step 2: Wrap function as LangChain tool
from langchain.tools.base import StructuredTool
event_tool = StructuredTool.from_function(get_events_by_date)

# Step 3: Agent setup
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

chat = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [event_tool]
print("called--------------------------")
agent_chain = initialize_agent(
    tools,
    chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    response = agent_chain.run("Are there any events on 5th June?")
    print(response)
