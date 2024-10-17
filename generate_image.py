import json
import matplotlib.pyplot as plt
import numpy as np
import os
from llama_stack_client import LlamaStackClient

# Define LlamaStack host/port
host = ""
port = 5000
client = LlamaStackClient(base_url=f"http://{host}:{port}")

model = "Llama3.2-1B-Instruct"

def get_arc_puzzle_json(user_message):
    iterator = client.inference.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": "You only respond with a valid JSON string representing an ARC puzzle. Do not include any other output."},
            {"role": "user", "content": user_message}
        ],
        stream=True,
        tool_prompt_format='json'
    )

    json_response = ""
    for chunk in iterator:
        json_response += chunk.event.delta
        print(chunk.event.delta, end="", flush=True)

    return json_response

def evaluate_and_retry(user_message, raw_response):
    print("\n\nError detected in JSON format. Sending instructions to correct...")
    
    feedback_message = (
        "The previous response was not valid JSON. "
        "Ensure to return a grid with numeric or categorical values "
        "in a 2D structure, with no extra text or errors."
    )
    return get_arc_puzzle_json(feedback_message)

# Function to render ARC puzzle and save as an image file
def render_arc_puzzle(puzzle, output_path="arc_puzzle.png"):
    try:
        # Extract the grid data
        grid = [[cell['value'] for cell in row['row']] for row in puzzle['rows']]
        print("\nExtracted Grid:", grid)

        # Map values to integers
        unique_values = list(set(cell for row in grid for cell in row))
        value_map = {val: idx for idx, val in enumerate(unique_values)}
        numeric_grid = np.array([[value_map[cell] for cell in row] for row in grid])
        print("\nNumeric Grid:\n", numeric_grid)

        # Plot the grid and save to file
        plt.imshow(numeric_grid, cmap='tab20', interpolation='nearest')
        plt.axis('off')  # Hide the axes for clarity

        # Save the image
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        print(f"Image saved at: {os.path.abspath(output_path)}")

        # Close the plot to free resources
        plt.close()

    except Exception as e:
        print(f"Error during rendering: {e}")

# Main logic
symmetry = ""
user_input = f"Create a simple 3x3 ARC puzzle with {symmetry}. Return only valid JSON."
arc_json_string = get_arc_puzzle_json(user_input)

print("\n\nGenerated ARC Puzzle JSON String:")
print(arc_json_string)

try:
    arc_json = json.loads(arc_json_string)
    print("\n\nParsed ARC Puzzle JSON:")
    print(arc_json)

    puzzle = arc_json['puzzle']
    render_arc_puzzle(puzzle, output_path="arc_puzzle.png")

except json.JSONDecodeError as e:
    print(f"\n\nError decoding JSON: {e}")
    print(f"Raw response: {arc_json_string}")

    arc_json_string = evaluate_and_retry(user_input, arc_json_string)

    try:
        arc_json = json.loads(arc_json_string)
        print("\n\nParsed ARC Puzzle JSON after retry:")
        print(arc_json)

        puzzle = arc_json['puzzle']
        render_arc_puzzle(puzzle, output_path="arc_puzzle_retry.png")

    except json.JSONDecodeError as e:
        print(f"\n\nFailed again: {e}")
        print(f"Final raw response: {arc_json_string}")
