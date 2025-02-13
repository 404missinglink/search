from datasets import load_dataset
import numpy as np
from heapq import heappush, heappop
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import multiprocessing as mp
from tqdm import tqdm
import time
from numpy.typing import NDArray

# Number of CPU cores to use (leave one core free for system)
NUM_CORES = max(1, mp.cpu_count() - 1)
BATCH_SIZE = 1000  # Increased batch size for better performance

@dataclass
class AirbnbListing:
    """Represents a single Airbnb listing with its properties and computed scores."""
    id: str
    price: float
    satisfaction: float
    attr_score: float  # Attraction score
    rest_score: float  # Restaurant score
    distance: float    # Distance to city center in km
    lat: float        # Latitude
    lng: float        # Longitude
    room_type: str
    normalized_score: float = 0.0
    
    def __lt__(self, other: 'AirbnbListing') -> bool:
        """Enable sorting listings by normalized score (higher is better)."""
        return self.normalized_score > other.normalized_score
    
    def get_maps_link(self) -> str:
        """Generate a Google Maps link for this listing's location."""
        return f"https://www.google.com/maps?q={self.lat},{self.lng}"

def process_batch(args):
    """Process a batch of listings"""
    batch, max_price, max_distance, weights = args
    results = []
    
    # Vectorize the normalization calculations
    prices = np.array([float(item['realSum']) for item in batch])
    satisfactions = np.array([float(item['guest_satisfaction_overall']) for item in batch])
    attr_scores = np.array([float(item['attr_index_norm']) for item in batch])
    rest_scores = np.array([float(item['rest_index_norm']) for item in batch])
    distances = np.array([float(item['dist']) for item in batch])
    
    # Normalize all at once
    norm_prices = 1 - (prices / max_price)
    norm_satisfactions = satisfactions / 100
    norm_locations = (attr_scores + rest_scores) / 2
    norm_distances = 1 - (distances / max_distance)
    
    # Calculate scores vectorized
    scores = (weights[0] * norm_prices + 
             weights[1] * norm_satisfactions + 
             weights[2] * norm_locations + 
             weights[3] * norm_distances)
    
    for i, item in enumerate(batch):
        listing = AirbnbListing(
            id=str(item['_id']),
            price=prices[i],
            satisfaction=satisfactions[i],
            attr_score=attr_scores[i],
            rest_score=rest_scores[i],
            distance=distances[i],
            lat=float(item['lat']),
            lng=float(item['lng']),
            room_type=str(item['room_type']),
            normalized_score=scores[i]
        )
        results.append(listing)
    
    return results

class AirbnbSearchProblem:
    def __init__(self, 
                 dataset: Dict,
                 price_weight: float = 0.3,
                 satisfaction_weight: float = 0.3,
                 location_weight: float = 0.2,
                 distance_weight: float = 0.2) -> None:
        """
        Initialize the search problem with dataset and scoring weights.
        
        Args:
            dataset: The Airbnb dataset containing listings
            price_weight: Weight for price normalization (0-1)
            satisfaction_weight: Weight for satisfaction score (0-1)
            location_weight: Weight for location quality (0-1)
            distance_weight: Weight for distance to center (0-1)
        """
        print("\nInitializing search problem...")
        start_time = time.time()
        
        self.dataset_list = list(dataset['train'])
        self.weights = np.array([
            price_weight,
            satisfaction_weight,
            location_weight,
            distance_weight
        ])
        
        # Calculate maximum values for normalization
        print("Calculating maximum values...")
        self._calculate_max_values()
        
        # Process and score all listings
        print(f"Processing {len(self.dataset_list)} listings...")
        self.listings = self._process_listings()
        
        # Sort listings by score for efficient access
        self.listings.sort(reverse=True)
        self.start = self.listings[0]  # Best scoring listing is our start state
        
        print(f"Initialization completed in {time.time() - start_time:.2f} seconds")
    
    def _calculate_max_values(self) -> None:
        """Calculate maximum values needed for score normalization."""
        self.max_price = max(float(x['realSum']) for x in self.dataset_list)
        self.max_distance = max(float(x['dist']) for x in self.dataset_list)
    
    def _extract_numeric_features(self) -> Tuple[NDArray, ...]:
        """Extract numeric features from dataset into numpy arrays."""
        return (
            np.array([float(x['realSum']) for x in self.dataset_list]),
            np.array([float(x['guest_satisfaction_overall']) for x in self.dataset_list]),
            np.array([float(x['attr_index_norm']) for x in self.dataset_list]),
            np.array([float(x['rest_index_norm']) for x in self.dataset_list]),
            np.array([float(x['dist']) for x in self.dataset_list])
        )
    
    def _normalize_features(self, 
                          prices: NDArray,
                          satisfactions: NDArray,
                          attr_scores: NDArray,
                          rest_scores: NDArray,
                          distances: NDArray) -> Tuple[NDArray, ...]:
        """Normalize all features to [0,1] range."""
        return (
            1 - (prices / self.max_price),           # Lower price is better
            satisfactions / 100,                     # Scale 0-100 to 0-1
            (attr_scores + rest_scores) / 2,         # Average location scores
            1 - (distances / self.max_distance)      # Closer is better
        )
    
    def _calculate_scores(self,
                         norm_prices: NDArray,
                         norm_satisfactions: NDArray,
                         norm_locations: NDArray,
                         norm_distances: NDArray) -> NDArray:
        """Calculate final scores using weighted sum of normalized features."""
        return (self.weights[0] * norm_prices + 
                self.weights[1] * norm_satisfactions + 
                self.weights[2] * norm_locations + 
                self.weights[3] * norm_distances)
    
    def _process_listings(self) -> List[AirbnbListing]:
        """Process all listings and compute their normalized scores."""
        # Extract features
        prices, satisfactions, attr_scores, rest_scores, distances = self._extract_numeric_features()
        
        # Normalize features
        norm_prices, norm_satisfactions, norm_locations, norm_distances = self._normalize_features(
            prices, satisfactions, attr_scores, rest_scores, distances
        )
        
        # Calculate final scores
        scores = self._calculate_scores(
            norm_prices, norm_satisfactions, norm_locations, norm_distances
        )
        
        # Create AirbnbListing objects
        listings = []
        for i, item in enumerate(tqdm(self.dataset_list, desc="Creating listing objects")):
            listing = AirbnbListing(
                id=str(item['_id']),
                price=prices[i],
                satisfaction=satisfactions[i],
                attr_score=attr_scores[i],
                rest_score=rest_scores[i],
                distance=distances[i],
                lat=float(item['lat']),
                lng=float(item['lng']),
                room_type=str(item['room_type']),
                normalized_score=scores[i]
            )
            listings.append(listing)
        
        return listings
    
    def get_successors(self, state: AirbnbListing) -> List[AirbnbListing]:
        """Get the most promising successors based on score similarity
        
        In this A* implementation, successors are nearby listings in the sorted score list.
        This is based on the assumption that good solutions will have similar scores.
        We look at a small window around the current listing's position to find promising next steps.
        
        Args:
            state: Current listing state
            
        Returns:
            List of neighboring listings that could be promising next steps
        """
        # Find current listing's position in the sorted list
        current_idx = next(i for i, l in enumerate(self.listings) if l.id == state.id)
        
        # Look at nearby listings in the sorted list (2 before, 2 after)
        window_size = 2
        start_idx = max(0, current_idx - window_size)
        end_idx = min(len(self.listings), current_idx + window_size + 1)
        
        # Return neighbors, excluding the current state
        return [l for l in self.listings[start_idx:end_idx] if l.id != state.id]
    
    @lru_cache(maxsize=1024)
    def heuristic(self, state_id: str) -> float:
        """Estimate how far we are from an optimal solution
        
        The heuristic function is crucial for A* search efficiency.
        Here we use (1 - normalized_score) as our heuristic because:
        1. Higher scores are better, so distance to goal = 1 - score
        2. The normalized_score combines all our optimization criteria
        3. It's admissible (never overestimates) since scores are normalized [0,1]
        
        Args:
            state_id: ID of the listing to evaluate
            
        Returns:
            Heuristic estimate of distance to goal (0 = optimal, 1 = worst)
        """
        state = next(l for l in self.listings if l.id == state_id)
        return 1 - state.normalized_score

def find_optimal_listing(dataset: Dict, 
                        weights: Optional[Dict[str, float]] = None) -> Tuple[List[AirbnbListing], float]:
    """Find an optimal listing path using A* search
    
    A* search is a best-first search algorithm that finds the optimal path by:
    1. Using a priority queue (frontier) to always explore the most promising nodes first
    2. Combining actual path cost with a heuristic estimate of remaining cost
    3. Maintaining an explored set to avoid revisiting nodes
    
    Args:
        dataset: The Airbnb dataset containing listings
        weights: Optional dictionary of weights for different criteria
                Keys: 'price', 'satisfaction', 'location', 'distance'
                Values: Weights between 0 and 1, should sum to 1
    
    Returns:
        Tuple of (best path found, score of best path)
    """
    # Default weights if none provided
    if weights is None:
        weights = {
            'price': 0.3,
            'satisfaction': 0.3,
            'location': 0.2,
            'distance': 0.2
        }
    
    print("\nStarting optimal listing search...")
    start_time = time.time()
    
    # Initialize search problem with weights
    problem = AirbnbSearchProblem(
        dataset,
        price_weight=weights['price'],
        satisfaction_weight=weights['satisfaction'],
        location_weight=weights['location'],
        distance_weight=weights['distance']
    )
    
    # Initialize A* search components
    start_state = problem.start
    frontier: List[Tuple[float, AirbnbListing, List[AirbnbListing]]] = [
        (-start_state.normalized_score, start_state, [start_state])
    ]
    explored: set[str] = {start_state.id}
    best_score = start_state.normalized_score
    best_path = [start_state]
    
    # Main A* search loop
    print("\nExploring listings...")
    max_explored = 25  # Limit exploration for efficiency
    with tqdm(total=max_explored, desc="Search progress") as pbar:
        while frontier and len(explored) < max_explored:
            # Get the most promising unexplored state
            score, current_state, current_path = heappop(frontier)
            
            # Update best solution if current is better
            if current_state.normalized_score > best_score:
                best_score = current_state.normalized_score
                best_path = current_path
            
            # Explore successors (nearby listings with similar scores)
            for next_state in problem.get_successors(current_state):
                if next_state.id not in explored:
                    explored.add(next_state.id)
                    
                    # Only consider promising states that could improve our solution
                    if next_state.normalized_score > best_score:
                        heappush(frontier, (
                            -next_state.normalized_score,  # Negative for max-heap behavior
                            next_state,
                            current_path + [next_state]
                        ))
                    pbar.update(1)
    
    print(f"\nSearch completed in {time.time() - start_time:.2f} seconds")
    return best_path, best_score

def print_listing_path(path: List[AirbnbListing], score: float):
    """Print the path of listings and their properties"""
    print(f"\nFound optimal path with score: {score:.3f}")
    print("\nListing Path:")
    for i, listing in enumerate(path, 1):
        print(f"\nStep {i}:")
        print(f"ID: {listing.id}")
        print(f"Room Type: {listing.room_type}")
        print(f"Price: ${listing.price:.2f}")
        print(f"Satisfaction: {listing.satisfaction:.1f}/100")
        print(f"Attraction Score: {listing.attr_score:.2f}")
        print(f"Restaurant Score: {listing.rest_score:.2f}")
        print(f"Distance to Center: {listing.distance:.2f}km")
        print(f"Normalized Score: {listing.normalized_score:.3f}")
        print(f"Location: {listing.lat:.6f}, {listing.lng:.6f}")
        print(f"View on Maps: {listing.get_maps_link()}")
        print("-" * 50)

if __name__ == "__main__":
    mp.freeze_support()
    
    print("Loading Airbnb dataset...")
    start_time = time.time()
    ds = load_dataset("kraina/airbnb", "all")
    print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")
    
    # Find optimal listing path
    best_path, best_score = find_optimal_listing(ds)
    
    # Print results
    print_listing_path(best_path, best_score) 