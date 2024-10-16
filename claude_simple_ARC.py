# Define the ARC task
task = {
    "train": [
        {
            "input": [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ],
            "output": [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        },
        {
            "input": [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0]
            ],
            "output": [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]
            ]
        }
    ],
    "test": [
        {
            "input": [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ]
        }
    ]
}

# Solution function
def solve_arc_task(input_grid):
    return [[1 for _ in row] for row in input_grid]

# Test the solution
test_input = task["test"][0]["input"]
test_output = solve_arc_task(test_input)

print("Input:")
for row in test_input:
    print(row)

print("\nOutput:")
for row in test_output:
    print(row)