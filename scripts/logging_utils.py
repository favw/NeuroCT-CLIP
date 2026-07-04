from datetime import datetime


def timestamped_message(message: str) -> str:
    return f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
