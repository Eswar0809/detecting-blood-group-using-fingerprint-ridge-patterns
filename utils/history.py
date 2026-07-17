"""
Lightweight runtime prediction history tracking.
"""

from collections import deque
from datetime import datetime

MAX_HISTORY_ITEMS = 20
_PREDICTION_HISTORY = deque(maxlen=MAX_HISTORY_ITEMS)


def add_prediction_history(image_id, blood_group, confidence_score):
    """Append one prediction event to in-memory history."""
    entry = {
        "image_id": image_id,
        "blood_group": blood_group,
        "confidence_score": float(confidence_score),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _PREDICTION_HISTORY.appendleft(entry)
    return entry


def get_recent_predictions(limit=10):
    """Return latest prediction history entries."""
    return list(_PREDICTION_HISTORY)[:limit]
