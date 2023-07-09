import json
import os

# def get_ids(path) -> list:
#     ids = []

#     for file_name in os.listdir(path):
#         if file_name.endswith('.json'):
#             with open(os.path.join(path, file_name), 'r') as f:
#                 data = json.load(f)
#                 ids_in_file = [key.split('.')[0] for key in data.keys()]
#                 ids.extend(ids_in_file)

#     return ids

def get_ids(path) -> list:
    ids = []
    
    for video in os.listdir(path):
        for face in os.listdir(os.path.join(path, video)):
            ids.append(video + '_' + face)

    return ids
    
def load_config(path) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
        return config