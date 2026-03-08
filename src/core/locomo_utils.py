
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def parse_range_string(range_str: Optional[str]) -> Optional[List[int]]:
    """Parse a range string like '1-4' or '3' or '1,3,5' into a list of integers."""
    if not range_str or range_str.lower() == "all":
        return None
    
    indices = []
    parts = range_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-"))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return sorted(list(set(indices)))

def load_locomo_data(
    input_path: str,
    mode: str, # "ingest" or "eval"
    sample_indices: Optional[List[int]] = None,
    session_ranges: Optional[List[int]] = None, # 1-based session indices
    user_id_override: Optional[str] = None
) -> List[Dict[str, Any]]:
    
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read LoCoMo file {input_path}: {e}")
        return []
        
    results = []
    
    # If data is a list, it's multiple samples.
    if not isinstance(data, list):
        data = [data] # Single sample case? Or different format? LoCoMo usually list.
    
    for idx, user_data in enumerate(data):
        # Sample filtering (0-based index)
        if sample_indices is not None and idx not in sample_indices:
            continue
            
        # Determine user_id
        if user_id_override:
            # If override provided, use it. 
            user_id = user_id_override
        else:
            # Deterministic ID based on sample index
            user_id = f"locomo_user_{idx}"
            
        if mode == "ingest":
            # Process Sessions
            session_container = user_data
            if "conversation" in user_data:
                session_container = user_data["conversation"]
                
            session_keys = []
            for k in session_container.keys():
                if k.startswith("session_") and not k.endswith("_date_time"):
                    parts = k.split("_")
                    if len(parts) == 2 and parts[1].isdigit():
                        session_num = int(parts[1])
                        # Filter by session range (1-based index)
                        if session_ranges is not None and session_num not in session_ranges:
                            continue
                        session_keys.append((session_num, k))
            
            # Sort by session number
            session_keys.sort(key=lambda x: x[0])
            
            for s_num, s_key in session_keys:
                turns = session_container[s_key]
                date_time = session_container.get(f"{s_key}_date_time", "")
                
                transcript_lines = []
                if date_time:
                    transcript_lines.append(f"[Session Date: {date_time}]")
                for turn in turns:
                    speaker = turn.get("speaker", "Unknown")
                    text = turn.get("text", "")
                    transcript_lines.append(f"{speaker}: {text}")
                
                transcript = "\n".join(transcript_lines)
                
                results.append({
                    "text": transcript,
                    "path": f"locomo/{user_id}/{s_key}",
                    "user_id": user_id,
                    "timestamp": date_time
                })
                
        elif mode == "eval":
            # Process QA
            qa_list = user_data.get("qa", [])
            for qa in qa_list:
                results.append({
                    "question": qa.get("question"),
                    "ground_truth": str(qa.get("answer")),
                    "evidence": qa.get("evidence"),
                    "category": qa.get("category"),
                    "user_id": user_id
                })
                
    return results
