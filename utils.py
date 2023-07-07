import json
import os

def get_ids(folder_path) -> list:
    ids = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as f:
                data = json.load(f)
                ids.append(data.keys())

    return ids

    
def load_config(json_path) -> dict:
    with open(json_path, 'r') as f:
        config = json.load(f)
        return config