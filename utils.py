import json
import os

def get_ids(path) -> list:
    ids = []

    for file_name in os.listdir(path):
        if file_name.endswith('.json'):
            with open(os.path.join(path, file_name), 'r') as f:
                data = json.load(f)
                ids.append(data.keys())

    return ids

    
def load_config(path) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
        return config