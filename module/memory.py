# modules/memory.py → Memory Manager
# Role: Embedding-based semantic memory using FAISS.

# Responsibilities:

# Store & retrieve MemoryItem objects

# Use local embedding server (e.g., Ollama) to vectorize input

# Filter memory based on type/tags/session

# Dependencies:

# faiss, requests, pydantic

# Used by: context.py, loop.py

# Inputs: Queries and tool outputs

# Outputs: Retrieved memory items for context injection

# modules/memory.py

import numpy as np
import requests
from typing import List, Optional, Literal
from pydantic import BaseModel
from datetime import datetime


class MemoryItem(BaseModel):
    text: str
    type: Literal["preference", "tool_output", "fact", "query", "system"] = "fact"
    timestamp: Optional[str] = datetime.now().isoformat()
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    session_id: Optional[str] = None


class MemoryManager:
    def __init__(self):
        self.data: List[MemoryItem] = []

    def add(self, item: MemoryItem):
        self.data.append(item)


    def retrieve(
        self
    ) -> List[MemoryItem]:
        if  len(self.data) == 0:
            return []
        else:
            return self.data

    def bulk_add(self, items: List[MemoryItem]):
        for item in items:
            self.add(item)

