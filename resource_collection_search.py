from heapq import heappush, heappop
from enum import Enum
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple
import random

class ResourceType(Enum):
    """Different types of resources that can be collected"""
    WATER = 'WATER'
    FOOD = 'FOOD'
    WOOD = 'WOOD'
    METAL = 'METAL'

class TerrainType(Enum):
    """Different terrain types affecting movement cost"""
    PLAIN = 1      # Normal movement cost
    MOUNTAIN = 3   # High movement cost
    SWAMP = 2      # Medium movement cost
    HAZARD = -1    # Dangerous area, high risk

@dataclass
class GameState:
    """Represents the complete state of the search problem"""
    position: Tuple[int, int]           # Current position (x, y)
    energy: int                         # Current energy level
    resources: Dict[ResourceType, int]  # Collected resources
    visited_positions: Set[Tuple]       # Track visited positions for path reconstruction
    
    def __hash__(self):
        """
        Make state hashable for visited set.
        This is needed because GameState needs to be hashable to be used in sets.
        """
        # Convert resources dict to tuple of (name, amount) pairs sorted by name
        # This is necessary because dictionaries aren't hashable
        resources_tuple = tuple(
            sorted(
                [(r_type.name, amount) for r_type, amount in self.resources.items()],
                key=lambda x: x[0]  # Sort by resource name for consistent hashing
            )
        )
        return hash((
            self.position,
            self.energy,
            resources_tuple
        ))
    
    def __lt__(self, other):
        """
        Implementation of less than operator for GameState.
        This is needed for heapq operations.
        We don't actually need to compare GameStates,
        so we use the id of the object as a stable comparison.
        """
        return id(self) < id(other)

class ResourceCollectionProblem:
    def __init__(self, size: int = 10, required_resources: Dict[ResourceType, int] = None):
        """
        Initialize the resource collection problem
        
        Parameters:
            size: Size of the square grid world (size x size)
            required_resources: Dictionary specifying how many of each resource type is needed
        """
        self.size = size
        self.grid = self._generate_world()  # Generate random terrain
        # Default resource requirements if none provided
        self.required_resources = required_resources or {
            ResourceType.WATER: 2,
            ResourceType.FOOD: 3,
            ResourceType.WOOD: 2,
            ResourceType.METAL: 1
        }
        
        # Place resources randomly in the world
        self.resource_locations = self._place_resources()
        self.start_position = (0, 0)  # Start at top-left corner
        self.initial_energy = 100     # Starting energy level
        
        # Movement costs for different terrain types
        self.movement_costs = {
            TerrainType.PLAIN: 1,     # Basic movement cost
            TerrainType.MOUNTAIN: 3,  # Difficult terrain, high cost
            TerrainType.SWAMP: 2,     # Moderately difficult terrain
            TerrainType.HAZARD: 10    # Very dangerous terrain
        }
        
    def _generate_world(self) -> List[List[TerrainType]]:
        """
        Generate a random grid world with various terrain types
        Returns a 2D grid where each cell contains a TerrainType
        """
        # Initialize grid with all plains
        grid = [[TerrainType.PLAIN for _ in range(self.size)] for _ in range(self.size)]
        
        # Randomly add terrain features with different probabilities:
        # 10% chance for mountains
        # 10% chance for swamps
        # 5% chance for hazards
        for i in range(self.size):
            for j in range(self.size):
                rand = random.random()
                if rand < 0.1:
                    grid[i][j] = TerrainType.MOUNTAIN
                elif rand < 0.2:
                    grid[i][j] = TerrainType.SWAMP
                elif rand < 0.25:
                    grid[i][j] = TerrainType.HAZARD
        
        return grid
    
    def _place_resources(self) -> Dict[ResourceType, List[Tuple[int, int]]]:
        """
        Place resources randomly in the world
        Returns a dictionary mapping resource types to lists of coordinates
        """
        # Initialize empty lists for each resource type
        resources = {resource_type: [] for resource_type in ResourceType}
        
        # Place twice as many resources as required to make problem solvable
        for resource_type in ResourceType:
            num_to_place = self.required_resources[resource_type] * 2
            placed = 0
            
            while placed < num_to_place:
                x = random.randint(0, self.size - 1)
                y = random.randint(0, self.size - 1)
                
                # Only place resources on non-hazardous terrain
                if self.grid[x][y] != TerrainType.HAZARD:
                    resources[resource_type].append((x, y))
                    placed += 1
        
        return resources

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds"""
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size

    def _get_possible_moves(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all possible moves from current position (including diagonals)"""
        x, y = position
        return [
            (x+1, y), (x-1, y), (x, y+1), (x, y-1),  # Cardinal directions
            (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)  # Diagonals
        ]

    def _calculate_new_energy(self, state: GameState, next_pos: Tuple[int, int]) -> int:
        """Calculate remaining energy after a move"""
        terrain_type = self.grid[next_pos[0]][next_pos[1]]
        move_cost = self.movement_costs[terrain_type]
        return state.energy - move_cost

    def _collect_resources_at_position(self, position: Tuple[int, int], current_resources: Dict[ResourceType, int]) -> Dict[ResourceType, int]:
        """Collect any available resources at the given position"""
        new_resources = current_resources.copy()
        for resource_type, locations in self.resource_locations.items():
            if position in locations and new_resources[resource_type] < self.required_resources[resource_type]:
                new_resources[resource_type] += 1
        return new_resources

    def get_successors(self, state: GameState) -> List[GameState]:
        """Get all possible next states from current state"""
        successors = []
        
        for next_pos in self._get_possible_moves(state.position):
            # Skip invalid positions
            if not self._is_valid_position(next_pos):
                continue
                
            # Check energy requirements
            new_energy = self._calculate_new_energy(state, next_pos)
            if new_energy <= 0:
                continue
            
            # Create new state with collected resources
            new_resources = self._collect_resources_at_position(next_pos, state.resources)
            new_visited = state.visited_positions | {next_pos}
            
            successor = GameState(
                position=next_pos,
                energy=new_energy,
                resources=new_resources,
                visited_positions=new_visited
            )
            
            successors.append(successor)
        
        return successors

    def is_goal(self, state: GameState) -> bool:
        """
        Check if all required resources have been collected
        Returns True if mission complete, False otherwise
        """
        return all(
            state.resources.get(resource_type, 0) >= required_amount
            for resource_type, required_amount in self.required_resources.items()
        )

    def heuristic(self, state: GameState) -> float:
        """
        Estimate minimum cost to goal state using:
        1. Manhattan distance to nearest needed resources
        2. Number of remaining resources needed
        Returns combined heuristic value
        """
        # Calculate remaining resources needed
        remaining_resources = {
            r_type: max(0, self.required_resources[r_type] - state.resources.get(r_type, 0))
            for r_type in ResourceType
        }
        
        # If no resources needed, we're done
        if sum(remaining_resources.values()) == 0:
            return 0
            
        # Calculate distances to nearest needed resources
        total_distance = 0
        for r_type, needed in remaining_resources.items():
            if needed > 0:
                # Calculate Manhattan distance to each resource location
                distances = [
                    abs(state.position[0] - x) + abs(state.position[1] - y)
                    for x, y in self.resource_locations[r_type]
                ]
                if distances:
                    total_distance += min(distances)  # Use distance to nearest resource
        
        # Combine distance and remaining resource count
        return total_distance + sum(remaining_resources.values()) * 5

    def _get_cell_symbol(self, position: Tuple[int, int], solution: GameState) -> str:
        """Get the symbol to display for a given cell position"""
        # Check path first
        if position in solution.visited_positions:
            return '*'
        
        # Check for resources
        for r_type, locations in self.resource_locations.items():
            if position in locations:
                return r_type.value[0]
        
        # Check terrain
        terrain_symbols = {
            TerrainType.MOUNTAIN: 'M',
            TerrainType.SWAMP: 'S',
            TerrainType.HAZARD: 'H',
            TerrainType.PLAIN: '.'
        }
        return terrain_symbols[self.grid[position[0]][position[1]]]

    def visualize_solution(self, solution: GameState):
        """
        Create ASCII visualization of the solution path and world
        Legend:
        * = Path taken
        W/F/W/M = Resource locations (first letter)
        M = Mountain, S = Swamp, H = Hazard, . = Plain
        """
        for i in range(self.size):
            row = []
            for j in range(self.size):
                symbol = self._get_cell_symbol((i, j), solution)
                row.append(symbol)
            print(' '.join(row))

    def print_solution_details(self, solution: GameState):
        """Print details about the found solution"""
        print("\nSolution found!")
        print(f"Final energy: {solution.energy}")
        print("Resources collected:")
        for resource_type, amount in solution.resources.items():
            print(f"{resource_type.value}: {amount}")
        print("\nPath visualization:")
        self.visualize_solution(solution)

def resource_collection_search(problem: ResourceCollectionProblem):
    """
    A* search implementation for resource collection problem
    Uses priority queue for frontier and maintains explored set
    Returns solution state if found, None otherwise
    """
    # Create initial state with starting position and empty resources
    initial_state = GameState(
        position=problem.start_position,
        energy=problem.initial_energy,
        resources={resource_type: 0 for resource_type in ResourceType},
        visited_positions={problem.start_position}
    )
    
    # Priority queue for frontier: (priority, cost, state)
    # Priority = cost + heuristic
    frontier = [(problem.heuristic(initial_state), 0, initial_state)]
    explored = set()  # Set of explored states
    
    while frontier:
        _, cost, current_state = heappop(frontier)
        
        # Check if we've reached the goal
        if problem.is_goal(current_state):
            return current_state
            
        # Skip if already explored
        if current_state in explored:
            continue
            
        explored.add(current_state)
        
        # Explore successors
        for successor in problem.get_successors(current_state):
            if successor not in explored:
                new_cost = cost + 1  # Increment path cost
                priority = new_cost + problem.heuristic(successor)
                heappush(frontier, (priority, new_cost, successor))
    
    return None  # No solution found

if __name__ == "__main__":
    # Create and solve a new resource collection problem
    problem = ResourceCollectionProblem(size=10)
    print("Searching for solution...")
    solution = resource_collection_search(problem)
    
    if solution:
        problem.print_solution_details(solution)
    else:
        print("No solution found!") 