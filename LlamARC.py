from llama_stack_client import LlamaStackClient

# Define LlamaStack host/port
host = "192.168.1.15"
port = 5000
client = LlamaStackClient(base_url=f"http://{host}:{port}")

# Define the model to use for inference
model = "Llama3.2-1B-Instruct"

# Function to communicate with the model and get ARC puzzle JSON
def get_arc_puzzle_json(user_message):
    # Send the user message for ARC puzzle generation
    iterator = client.inference.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You only respond with a valid JSON string representing an ARC puzzle.  Do not include any other output."},
            {"role": "user", "content": user_message}
        ],
        stream=True
    )

    # Collect the streamed JSON response from the model
    json_response = ""
    for chunk in iterator:
        json_response += chunk.event.delta
        print(chunk.event.delta, end="", flush=True)
    
    return json_response

import json

# Example usage - you can modify the user input incrementally
user_input = "Create a simple 3x3 ARC puzzle with symmetry. Return only valid JSON."
arc_json_string = get_arc_puzzle_json(user_input)

print("\n\nGenerated ARC Puzzle JSON String:")
print(arc_json_string)

try:
    arc_json = json.loads(arc_json_string)
    print("\n\nParsed ARC Puzzle JSON:")
    print(arc_json)
except json.JSONDecodeError as e:
    print(f"\n\nError decoding JSON: {e}")
    print(f"Raw response: {arc_json_string}")
