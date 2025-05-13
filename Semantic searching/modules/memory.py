def update_memory(memory, new_query, max_len=5):
    memory.append(new_query)
    return memory[-max_len:]

def get_full_query(memory, current_query):
    if not memory:
        return current_query
    return " ".join(memory[-2:] + [current_query])
