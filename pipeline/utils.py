import json
import os

def get_ids(path) -> list:
    ids = []
    
    for video in os.listdir(path):
        for face in os.listdir(os.path.join(path, video)):
            if face == '0' and len(os.listdir(os.path.join(path, video, face))) >= 24:
                ids.append(video + '_' + face)

    return ids
    
def load_config(path) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
        return config