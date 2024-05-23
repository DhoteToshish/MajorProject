import json
import os

def require_json(relative_path):
    # Construct the absolute file path
    base_path = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(base_path, relative_path)

    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data