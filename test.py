import json

# Open the JSON file
with open('intents.json', 'r') as file:
    # Load the JSON data into a dictionary
    data = json.load(file)

# Print the entire dictionary
print(data)
