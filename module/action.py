# modules/action.py

from typing import Dict, Any, Union
from pydantic import BaseModel
import ast

# Optional logging fallback
try:
    from agent import log
except ImportError:
    import datetime
    def log(stage: str, msg: str):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] [{stage}] {msg}")


class ToolCallResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Union[str, list, dict]
    raw_response: Any


def parse_function_call(response: str) -> tuple[str, Dict[str, Any]]:
    """Parses FUNCTION_CALL string into tool name and arguments."""
    try:
        if not response.startswith("FUNCTION_CALL:"):
            raise ValueError("Not a valid FUNCTION_CALL")

        _, function_info = response.split(":", 1)
        parts = [p.strip() for p in function_info.split("|")]
        func_name, param_parts = parts[0], parts[1:]
        result = {}
        for part in param_parts:
            if "=" not in part:
                raise ValueError(f"Invalid param: {part}")
            
            if "numbers=" in part:
                part = part.replace("numbers=", "input.a=")

            if "numbers_to_exclude=" in part:
                part = part.replace("numbers_to_exclude=", "input.b=") 

            if "number_to_exclude=" in part:
                part = part.replace("number_to_exclude=", "input.b=") 
            
            if "exclude=" in part:
                part = part.replace("exclude=", "input.b=") 

            if "number=" in part:
                part = part.replace("number=", "input.b=") 

            key, value = part.split("=", 1)

            try:
                parsed_value = ast.literal_eval(value)
            except Exception:
                parsed_value = value.strip()

            #Handle nested keys
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                current = current.setdefault(k, {})
            current[keys[-1]] = parsed_value

        #log("parser", f"Parsed: {func_name} → {param_parts[0]}")
        #return func_name, list(param_parts[0])
        log("parser", f"Parsed: {func_name} → {result}")
        return func_name, result

    except Exception as e:
        log("parser", f"❌ Failed to parse FUNCTION_CALL: {e}")
        raise
