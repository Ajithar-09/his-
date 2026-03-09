from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from bot import chat_with_bot
import json

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = ""
    teacher_id: Optional[str] = None

class ChatResponse(BaseModel):
    status: str = "success"
    response: str
    structured_data: dict = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:

        if not request.message.strip():
            greeting = "Hello I'm His-bot How can I help you?"
            return ChatResponse(response=greeting)
            
        raw_output = chat_with_bot(request.message, request.teacher_id)
        

        structured_info = {}
        if "{" in raw_output and "}" in raw_output:
            try:

                start = raw_output.find("{")
                end = raw_output.rfind("}") + 1
                json_str = raw_output[start:end]
                structured_info = json.loads(json_str)
            except:
                pass
        
        return ChatResponse(response=raw_output, structured_data=structured_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
