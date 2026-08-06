chat_history = []
MAX_HISTORY = 10

def add_message(role, content):
    global chat_history
    chat_history.append({
        "role": role,
        "content": content
    })

    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)

def get_history():
    return chat_history

def clear_history():
    global chat_history
    chat_history = []