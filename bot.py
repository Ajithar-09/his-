import os
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Union
from bson import ObjectId
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

load_dotenv()

# MongoDB Setup
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client.get_database("HISK")

# Memory Setup
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Helper to format dates
def format_date(dt):
    if isinstance(dt, datetime):
        return dt.strftime("%d/%m/%Y")
    return str(dt)

@tool
def get_teacher_id_by_name(name: str) -> str:
    """Useful when the user mentions a teacher by name.
    Returns the MongoDB ID string or an error message.
    """
    try:
        teacher = db.teachers.find_one({"name": {"$regex": name, "$options": "i"}})
        if teacher:
            return str(teacher["_id"])
        return f"Teacher with name '{name}' not found."
    except Exception as e:
        return f"Error searching for teacher: {str(e)}"

@tool
def get_teacher_name_by_id(teacher_id: str) -> str:
    """Useful to find the name of the teacher when their MongoDB ID is known.
    Returns the teacher's name or an error message.
    """
    try:
        teacher = db.teachers.find_one({"_id": ObjectId(teacher_id)})
        if teacher:
            return teacher.get("name", "Unknown")
        return f"Teacher with ID '{teacher_id}' not found."
    except Exception as e:
        return f"Error searching for teacher name: {str(e)}"

@tool
def list_teacher_periods(teacher_id: str, date_str: Optional[str] = None) -> str:
    """Lists periods for a teacher. 
    Use 'date_str' (DD/MM/YYYY) to filter for a specific day.
    """
    try:
        teacher_oid = ObjectId(teacher_id)
        query = {"teacherId": teacher_oid}
        
        if date_str:
            date_formats = ["%d/%m/%Y", "%d/%m/%y"]
            dt = None
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError: pass
            
            if dt:
                query["date"] = {"$gte": dt.replace(hour=0, minute=0, second=0), "$lt": dt.replace(hour=23, minute=59, second=59)}

        timetables = list(db.timetables.find(query).sort("period", 1))
        
        if not timetables:
            return f"No periods found for teacher ID {teacher_id}" + (f" on {date_str}" if date_str else "")

        results = []
        for entry in timetables:
            class_info = db.classes.find_one({"_id": entry.get("classId")})
            classname = class_info.get("className", "Unknown") if class_info else "Unknown"
            section = class_info.get("section", "") if class_info else ""
            
            results.append(f"- Date: {format_date(entry.get('date'))}, P{entry.get('period')}: {classname} {section} ({entry.get('subject')})")
        
        return "Teacher Schedule:\n" + "\n".join(results)
    except Exception as e:
        return f"Error listing periods: {str(e)}"

@tool
def change_period_schedule(teacher_id: str, old_period: int, new_period: int, old_date: Optional[str] = None, new_date: Optional[str] = None) -> str:
    """Changes a period from an old date/slot to a new date/slot for a teacher ID.
    If dates are not provided, it defaults to the current date.
    Format for dates: DD/MM/YYYY or DD/MM/YY.
    Example: 'change my 5th period to 7th' (assumes today) or 'change my 3rd period on 12/01/24 to 14/01/24 5th period'.
    """
    try:
        teacher_oid = ObjectId(teacher_id)
        today_str = datetime.now().strftime("%d/%m/%Y")
        

        if not old_date: old_date = today_str
        if not new_date: new_date = today_str
        

        date_formats = ["%d/%m/%Y", "%d/%m/%y"]
        old_dt = None
        new_dt = None
        
        for fmt in date_formats:
            try:
                if not old_dt: old_dt = datetime.strptime(old_date, fmt)
            except ValueError: pass
            try:
                if not new_dt: new_dt = datetime.strptime(new_date, fmt)
            except ValueError: pass

        if not old_dt or not new_dt:
            return json.dumps({"error": f"Invalid date format: {old_date} or {new_date}. Use DD/MM/YYYY or DD/MM/YY"})


        query = {
            "teacherId": teacher_oid,
            "period": old_period,
            "date": {"$gte": old_dt.replace(hour=0, minute=0, second=0), "$lt": old_dt.replace(hour=23, minute=59, second=59)}
        }
        
        period_entry = db.timetables.find_one(query)
        
        if not period_entry:
            return json.dumps({"error": f"No period found for teacher on {old_date} at period {old_period}"})
        

        target_query = {
            "teacherId": teacher_oid,
            "period": new_period,
            "date": {"$gte": new_dt.replace(hour=0, minute=0, second=0), "$lt": new_dt.replace(hour=23, minute=59, second=59)}
        }
        conflict = db.timetables.find_one(target_query)
        if conflict:
            return json.dumps({"error": f"Target slot ({new_date} period {new_period}) is already occupied."})


        new_day = new_dt.strftime("%A")
        
        result = db.timetables.update_one(
            {"_id": period_entry["_id"]},
            {"$set": {
                "date": new_dt,
                "period": new_period,
                "day": new_day
            }}
        )
        
        if result.modified_count > 0:
            # Fetch planned topics from monthlyheralds
            topics = "No topic found"
            try:
                class_id = period_entry.get("classId")
                subject = period_entry.get("subject")
                new_year = new_dt.year
                new_month_name = new_dt.strftime("%B").lower()
                new_date_iso = new_dt.strftime("%Y-%m-%d")

                # 1. Find the syllabus ID
                syllabus = db.syllabuses.find_one({"classId": class_id, "subject": subject})
                if syllabus:
                    syllabus_id = syllabus["_id"]
                    

                    herald = db.monthlyheralds.find_one({
                        "classId": class_id,
                        "syllabusId": syllabus_id,
                        "year": new_year
                    })
                    
                    if herald and "period_plan" in herald:
                        month_plan = herald["period_plan"].get(new_month_name, [])

                        for plan_entry in month_plan:
                            if plan_entry.get("date") == new_date_iso and plan_entry.get("period_number") == new_period:
                                plan_topics = plan_entry.get("topics", [])
                                if plan_topics:
                                    topics = ", ".join(plan_topics) if isinstance(plan_topics, list) else str(plan_topics)
                                break
            except Exception as e:
                print(f"Error fetching topics: {e}")

            return json.dumps({
                "date": new_dt.isoformat() + ".000+00:00",
                "period": new_period,
                "subject": period_entry.get("subject", "Unknown"),
                "topics": topics
            })
        else:
            return json.dumps({"error": "Failed to update the database."})
            
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_chatbot_app(teacher_id: Optional[str] = None):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    
    tools = [get_teacher_id_by_name, get_teacher_name_by_id, list_teacher_periods, change_period_schedule]
    
    teacher_context = f"The primary teacher (user) MongoDB ID is: {teacher_id}. Use this for 'my' or 'me' requests." if teacher_id else "No primary teacher ID provided."
    
    system_message = f"""You are a helpful assistant for teachers at HISK school.
    Your job is to help teachers manage their period schedules.

    CONTEXT:
    - Primary Teacher ID: {teacher_id}
    - Today's Date: {datetime.now().strftime('%d/%m/%Y')}

    RULES:
    1. For 'my' or 'me' requests, use the Primary Teacher ID: {teacher_id}.
    2. If a specific name is mentioned (e.g., 'Kavin'), find their ID first using get_teacher_id_by_name.
    3. For schedule lists:
       - Use list_teacher_periods.
       - If asked for 'today' or 'tomorrow', pass the correct date string.
    4. For changes: Use change_period_schedule.
    5. Always respond politely. For successful schedule changes, you MUST provide the updated details in a JSON block at the end.
    Example JSON format:
    ```json
    {{
        "date": "2026-05-11T18:30:00.000+00:00",
        "period": 3,
        "subject": "Maths",
        "topics": "Algrebra, Fractions"
    }}
    ```
    """
    
    app = create_react_agent(llm, tools, prompt=system_message, checkpointer=memory)
    return app

def chat_with_bot(user_input: str, teacher_id: Optional[str] = None):
    app = get_chatbot_app(teacher_id)
   
    session_id = f"session_{teacher_id}" if teacher_id else "session_anonymous"
    config = {"configurable": {"thread_id": session_id}}

    final_state = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    return final_state["messages"][-1].content
