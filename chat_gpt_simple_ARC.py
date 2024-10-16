# what is the simplest ARC problem you can write code for right now?
def transform_grid(grid):
    # Iterate through each row and column of the grid
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            # Replace all occurrences of 1 with 2
            if grid[i][j] == 1:
                grid[i][j] = 2
    return grid

# Example input
grid = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
]

# Transform the grid
output = transform_grid(grid)

# Print the output
for row in output:
    print(row)