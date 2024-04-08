import json

# Sample data for states
states_data = []

for i in range(22):
    states_data.append({"state_id": 1, "step_name":"1-1", "text": "content", "image_path": ""})

# Save data to JSON file
with open('states.json', 'w') as json_file:
    json.dump(states_data, json_file, indent=4)
