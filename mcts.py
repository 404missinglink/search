from __future__ import annotations
import math
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Location:
    """Represents a delivery location with coordinates and time windows"""
    id: str
    x: float
    y: float
    earliest_time: float  # Earliest allowed delivery time
    latest_time: float   # Latest allowed delivery time
    service_time: float  # Time needed to complete delivery

@dataclass
class State:
    """Represents the current state of the delivery route"""
    current_location: Location
    unvisited: List[Location]
    time: float
    total_distance: float
    route: List[Location]
    arrival_times: Dict[str, float]  # Track arrival time for each location

    def is_terminal(self) -> bool:
        """Check if all locations have been visited"""
        return len(self.unvisited) == 0

    def get_possible_moves(self) -> List[Location]:
        """Get all possible next locations that can be visited"""
        possible_moves = []
        for location in self.unvisited:
            travel_time = self.calculate_travel_time(self.current_location, location)
            arrival_time = self.time + travel_time
            
            # Wait if we arrive before the earliest allowed time
            actual_service_start = max(arrival_time, location.earliest_time)
            
            # Check if we can complete service before the latest allowed time
            if actual_service_start + location.service_time <= location.latest_time:
                possible_moves.append(location)
        return possible_moves

    @staticmethod
    def calculate_travel_time(loc1: Location, loc2: Location) -> float:
        """Calculate travel time between two locations"""
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        distance = math.sqrt(dx*dx + dy*dy)
        # Assume constant speed of 1 unit per time
        return distance

class Node:
    """Represents a node in the Monte Carlo Search Tree"""
    def __init__(self, state: State, parent: Optional[Node] = None, move: Optional[Location] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children: List[Node] = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, move: Location, state: State) -> Node:
        """Add a child node"""
        child = Node(state, self, move)
        self.children.append(child)
        return child

    def update(self, reward: float):
        """Update node statistics"""
        self.visits += 1
        self.value += (reward - self.value) / self.visits

class MCTS:
    """Monte Carlo Tree Search implementation for route planning"""
    def __init__(self, exploration_weight: float = 1.0):
        self.exploration_weight = exploration_weight

    def choose_move(self, root_state: State, iterations: int = 1000) -> Optional[Location]:
        """Choose the best move from the current state"""
        root = Node(root_state)

        # Check if there are any possible moves
        if not root_state.get_possible_moves():
            return None

        for _ in range(iterations):
            node = self.select(root)
            reward = self.simulate(node.state)
            self.backpropagate(node, reward)

        # If no children were created, return None
        if not root.children:
            return None

        # Choose the move with the highest value
        best_child = max(root.children, key=lambda c: c.value / c.visits)
        return best_child.move

    def select(self, node: Node) -> Node:
        """Select a node to expand"""
        while not node.state.is_terminal():
            if len(node.children) < len(node.state.get_possible_moves()):
                return self.expand(node)
            if not node.children:  # If no children are available
                return node
            node = self.uct_select(node)
        return node

    def expand(self, node: Node) -> Node:
        """Expand a node by adding a child"""
        moves = node.state.get_possible_moves()
        if not moves:  # If no moves are available
            return node
        tried_moves = {child.move for child in node.children}
        untried_moves = [move for move in moves if move not in tried_moves]
        
        if not untried_moves:  # If no untried moves are available
            return node
        
        move = random.choice(untried_moves)
        
        # Create new state
        new_unvisited = [loc for loc in node.state.unvisited if loc != move]
        travel_time = node.state.calculate_travel_time(node.state.current_location, move)
        new_time = max(node.state.time + travel_time, move.earliest_time) + move.service_time
        new_total_distance = node.state.total_distance + travel_time
        new_route = node.state.route + [move]
        
        new_state = State(
            current_location=move,
            unvisited=new_unvisited,
            time=new_time,
            total_distance=new_total_distance,
            route=new_route,
            arrival_times=dict(node.state.arrival_times)
        )
        
        return node.add_child(move, new_state)

    def simulate(self, state: State) -> float:
        """Simulate a random playout"""
        current_state = state
        total_distance = state.total_distance
        num_visited = len(state.route)
        
        while not current_state.is_terminal():
            moves = current_state.get_possible_moves()
            if not moves:
                # Penalize based on how many locations were left unvisited
                remaining = len(current_state.unvisited)
                return float('-inf') if remaining == len(state.unvisited) else -1000 * remaining
            
            move = random.choice(moves)
            travel_time = current_state.calculate_travel_time(current_state.current_location, move)
            arrival_time = current_state.time + travel_time
            actual_service_start = max(arrival_time, move.earliest_time)
            new_time = actual_service_start + move.service_time
            
            new_unvisited = [loc for loc in current_state.unvisited if loc != move]
            new_total_distance = current_state.total_distance + travel_time
            new_arrival_times = dict(current_state.arrival_times)
            new_arrival_times[move.id] = arrival_time
            
            current_state = State(
                current_location=move,
                unvisited=new_unvisited,
                time=new_time,
                total_distance=new_total_distance,
                route=current_state.route + [move],
                arrival_times=new_arrival_times
            )
        
        # Reward complete routes more than partial routes
        return 1000.0 + (1.0 / total_distance)  # Complete routes get base reward plus distance bonus

    def backpropagate(self, node: Node, reward: float):
        """Backpropagate the reward up the tree"""
        while node is not None:
            node.update(reward)
            node = node.parent

    def uct_select(self, node: Node) -> Node:
        """Select a child node using UCT formula"""
        log_n_visits = math.log(node.visits)
        
        def uct(child: Node) -> float:
            exploitation = child.value / child.visits
            exploration = math.sqrt(log_n_visits / child.visits)
            return exploitation + self.exploration_weight * exploration
        
        return max(node.children, key=uct)

# Example usage
def create_example_problem() -> Tuple[Location, List[Location]]:
    """Create an example delivery problem"""
    depot = Location("depot", 0, 0, 0, 24, 0)
    
    # Create some delivery locations with time windows
    locations = [
        Location("A", 2, 3, 0, 20, 0.5),    # Location A: deliverable between 0-20
        Location("B", -1, 5, 0, 20, 0.5),   # Location B: deliverable between 0-20
        Location("C", 5, -2, 0, 20, 0.5),   # Location C: deliverable between 0-20
        Location("D", -3, -4, 0, 20, 0.5)   # Location D: deliverable between 0-20
    ]
    
    return depot, locations

def main():
    # Create problem
    depot, locations = create_example_problem()
    
    # Create initial state
    initial_state = State(
        current_location=depot,
        unvisited=locations,
        time=0,
        total_distance=0,
        route=[depot],
        arrival_times={depot.id: 0}
    )
    
    # Create MCTS solver
    mcts = MCTS(exploration_weight=2.0)  # Increased exploration
    
    # Plan the route
    current_state = initial_state
    while not current_state.is_terminal():
        next_move = mcts.choose_move(current_state, iterations=2000)  # Increased iterations
        if next_move is None:
            print("No valid moves available. Route cannot be completed.")
            break
            
        print(f"Next delivery: Location {next_move.id}")
        
        # Update state
        travel_time = current_state.calculate_travel_time(current_state.current_location, next_move)
        arrival_time = current_state.time + travel_time
        actual_service_start = max(arrival_time, next_move.earliest_time)
        new_time = actual_service_start + next_move.service_time
        new_unvisited = [loc for loc in current_state.unvisited if loc != next_move]
        new_total_distance = current_state.total_distance + travel_time
        
        new_arrival_times = dict(current_state.arrival_times)
        new_arrival_times[next_move.id] = arrival_time
        
        current_state = State(
            current_location=next_move,
            unvisited=new_unvisited,
            time=new_time,
            total_distance=new_total_distance,
            route=current_state.route + [next_move],
            arrival_times=new_arrival_times
        )
    
    # Print final route
    print("\nFinal Route:")
    for location in current_state.route:
        arrival = current_state.arrival_times[location.id]
        print(f"Location {location.id} (arrived at time {arrival:.2f})")
    print(f"Total distance: {current_state.total_distance:.2f}")

if __name__ == "__main__":
    main()