from heapq import heappush, heappop
from enum import Enum
from dataclasses import dataclass
from typing import List, Set, Dict, Tuple, Optional, Iterator
import random

class ResourceType(Enum):
    """
    Different types of resources that can be collected
    Each resource type has a unique string value for visualization
    """
    WATER = 'WATER'
    FOOD = 'FOOD'
    WOOD = 'WOOD'
    METAL = 'METAL'

class TerrainType(Enum):
    """
    Different terrain types affecting movement cost
    The value represents the energy cost to traverse this terrain
    """
    PLAIN = 1      # Normal movement cost (1 energy)
    MOUNTAIN = 3   # High movement cost (3 energy)
    SWAMP = 2      # Medium movement cost (2 energy)
    HAZARD = -1    # Dangerous area, high risk (10 energy in movement_costs)

# Type aliases for better readability
Position = Tuple[int, int]
Resources = Dict[ResourceType, int]
Grid = List[List[TerrainType]]
ResourceLocations = Dict[ResourceType, List[Position]]
FrontierItem = Tuple[float, int, 'GameState']  # priority, cost, state

@dataclass
class GameState:
    """
    Represents the complete state of the search problem
    Includes current position, energy level, collected resources, and path history
    """
    position: Position          # Current position as (x, y) coordinates
    energy: int                # Remaining energy units
    resources: Resources       # Dictionary tracking collected resource amounts
    visited_positions: Set[Position]  # Set of positions in path to this state
    
    def __hash__(self) -> int:
        """
        Make state hashable for visited set.
        Required for using GameState in sets and as dictionary keys.
        """
        # Convert mutable resources dict to immutable tuple for hashing
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
    
    def __lt__(self, other: 'GameState') -> bool:
        """
        Implementation of less than operator for GameState.
        Compares states based on energy and collected resources.
        """
        if self.energy != other.energy:
            return self.energy > other.energy
        
        # Compare total resources collected
        self_resources = sum(self.resources.values())
        other_resources = sum(other.resources.values())
        return self_resources > other_resources

class ResourceCollectionProblem:
    """
    Represents a resource collection problem instance.
    Includes the world grid, resource locations, and search parameters.
    """
    
    def __init__(self, size: int = 10, required_resources: Optional[Resources] = None) -> None:
        self.size: int = size
        self.grid: Grid = self._generate_world()
        
        # Default resource requirements if none provided
        self.required_resources: Resources = required_resources or {
            ResourceType.WATER: 2,
            ResourceType.FOOD: 3,
            ResourceType.WOOD: 2,
            ResourceType.METAL: 1
        }
        
        # Game world settings
        self.resource_locations: ResourceLocations = self._place_resources()
        self.start_position: Position = (0, 0)
        self.initial_energy: int = 100
        
        # Movement costs for different terrain types
        self.movement_costs: Dict[TerrainType, int] = {
            TerrainType.PLAIN: 1,
            TerrainType.MOUNTAIN: 3,
            TerrainType.SWAMP: 2,
            TerrainType.HAZARD: 10
        }

    def _generate_world(self) -> Grid:
        """
        Generate a random grid world with various terrain types.
        
        Terrain distribution:
        - 75% Plains (default)
        - 10% Mountains
        - 10% Swamps
        - 5% Hazards
        
        Returns:
            2D grid where each cell contains a TerrainType
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
    
    def _place_resources(self) -> ResourceLocations:
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

    def _is_valid_position(self, position: Position) -> bool:
        """Check if a position is within grid bounds"""
        x, y = position
        return 0 <= x < self.size and 0 <= y < self.size

    def _get_possible_moves(self, position: Position) -> Iterator[Position]:
        """
        Get all possible moves from current position (including diagonals)
        
        Args:
            position: Current (x, y) position
            
        Yields:
            All eight possible adjacent positions, including diagonals
        """
        x, y = position
        moves = [
            (x+dx, y+dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if (dx, dy) != (0, 0)  # Exclude current position
        ]
        yield from moves

    def _calculate_new_energy(self, state: GameState, next_pos: Position) -> int:
        """
        Calculate remaining energy after a move
        
        Args:
            state: Current game state
            next_pos: Position being moved to
            
        Returns:
            Remaining energy after the move
        """
        # Calculate if move is diagonal
        dx = abs(state.position[0] - next_pos[0])
        dy = abs(state.position[1] - next_pos[1])
        is_diagonal = dx == 1 and dy == 1
        
        terrain_type = self.grid[next_pos[0]][next_pos[1]]
        move_cost = self.movement_costs[terrain_type]
        
        # Diagonal moves should cost more (multiply by sqrt(2) ≈ 1.4)
        if is_diagonal:
            move_cost = int(move_cost * 1.4)
        
        return state.energy - move_cost

    def _collect_resources_at_position(self, position: Position, current_resources: Resources) -> Resources:
        """Collect any available resources at the given position"""
        new_resources = current_resources.copy()
        for resource_type, locations in self.resource_locations.items():
            if position in locations and new_resources[resource_type] < self.required_resources[resource_type]:
                new_resources[resource_type] += 1
        return new_resources

    def get_successors(self, state: GameState) -> List[GameState]:
        """Get all possible next states from current state"""
        successors = []
        
        # Get needed resources
        needed_resources = {
            r_type for r_type in ResourceType
            if state.resources.get(r_type, 0) < self.required_resources[r_type]
        }
        
        # Get all possible moves
        possible_moves = list(self._get_possible_moves(state.position))
        
        # Sort moves by proximity to needed resources
        if needed_resources:
            possible_moves.sort(key=lambda pos: min(
                abs(pos[0] - rx) + abs(pos[1] - ry)
                for r_type in needed_resources
                for rx, ry in self.resource_locations[r_type]
            ))
        
        for next_pos in possible_moves:
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
        Estimate minimum cost to goal state
        
        Uses:
        1. Manhattan distance to nearest needed resources
        2. Number of remaining resources needed
        3. Terrain costs to reach resources
        
        Returns:
            Estimated minimum cost to goal
        """
        # Calculate remaining resources needed
        remaining_resources = {
            r_type: max(0, self.required_resources[r_type] - state.resources.get(r_type, 0))
            for r_type in ResourceType
        }
        
        # If no resources needed, we're done
        if sum(remaining_resources.values()) == 0:
            return 0
            
        # Calculate minimum cost to collect each needed resource
        total_cost = 0
        current_pos = state.position
        
        for r_type, needed in remaining_resources.items():
            if needed > 0:
                # Find distance to nearest resource of this type
                min_cost = float('inf')
                for rx, ry in self.resource_locations[r_type]:
                    # Manhattan distance
                    distance = abs(current_pos[0] - rx) + abs(current_pos[1] - ry)
                    
                    # Estimate terrain costs (minimum possible cost)
                    terrain_cost = distance * self.movement_costs[TerrainType.PLAIN]
                    
                    min_cost = min(min_cost, terrain_cost)
                
                total_cost += min_cost * needed
        
        return total_cost

    def _get_cell_symbol(self, position: Position, solution: GameState) -> str:
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

def resource_collection_search(problem: ResourceCollectionProblem, max_explored: int = 10000):
    """
    A* search implementation for resource collection problem
    Uses priority queue for frontier and maintains explored set
    Returns solution state if found, None otherwise
    
    Args:
        problem: The resource collection problem instance
        max_explored: Maximum number of states to explore before giving up
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
        if len(explored) >= max_explored:
            print(f"Search terminated: Exceeded maximum explored states ({max_explored})")
            return None
            
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