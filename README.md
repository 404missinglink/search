# Search Algorithm Implementations

This repository contains implementations of various search algorithms for pathfinding in a grid-based environment.

## Available Algorithms

- **Breadth First Search (BFS)**: Explores all nodes at the present depth before moving to nodes at the next depth level. Guarantees the shortest path in unweighted graphs.
- **Depth First Search (DFS)**: Explores as far as possible along each branch before backtracking. Memory efficient but doesn't guarantee shortest path.
- **A\* Search**: An informed search algorithm that uses heuristics to find the most promising path. Guarantees shortest path and is generally more efficient than BFS.
- **Reflex Agent**: A simple agent that makes decisions based only on the current state, demonstrating basic reactive behavior in a cleaning environment.
- **Resource Collection Search**: An A\* search implementation for collecting resources in a grid world with various terrain types and energy constraints.

## Usage

Each algorithm can be run independently. The grid-based environment uses the following symbols:

- S: Start position
- G: Goal position
- 1 or #: Wall/obstacle
- 0 or space: Empty space
- \*: Path taken by the agent
- .: Explored cells

For the Resource Collection Search:

- \*: Path taken
- W: Water resource
- F: Food resource
- W: Wood resource
- M: Metal resource
- M: Mountain terrain
- S: Swamp terrain
- H: Hazard terrain
- .: Plain terrain

### Running the Algorithms

# Run Breadth First Search

python breadth_first_search.py

# Run Depth First Search

python depth_first_search.py

# Run A\* Search

python astar_search.py

# Run Reflex Agent

python reflex_agent.py

# Run Resource Collection Search

python resource_collection_search.py

### Modifying the Environment

You can modify the maze/grid in any of the search algorithms by changing the maze variable in the **main** block:

maze = [
[0, 0, 0, 1, 0],
['S', 1, 0, 1, 0],
[0, 1, 0, 0, 0],
[0, 0, 0, 1, 'G'],
[0, 1, 0, 1, 0]
]

For the reflex agent, you can modify the environment size and number of dirt locations by changing the parameters in the Environment class initialization.

For the resource collection search, you can modify:

- Grid size
- Required resource amounts
- Initial energy
- Terrain distribution
- Resource placement

## Visualization

All implementations include visualization of the search process, showing:

1. The step-by-step exploration of the environment
2. The final path found (if applicable)
3. Clear visual indicators of walls, explored space, and the solution path

## Requirements

- Python 3.6+
- No external dependencies required
