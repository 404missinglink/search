import multiprocessing as mp
from datasets import load_dataset
from airbnb_search import find_optimal_listing, print_listing_path

# Weights for different criteria (must sum to 1.0)
WEIGHTS = {
    # How much to prioritize lower prices
    # Higher value = prefer cheaper listings
    # Example: 0.4 would strongly prefer cheaper places
    'price': 0.3,
    
    # How much to prioritize guest satisfaction ratings
    # Higher value = prefer better-rated listings
    # Based on actual guest reviews (0-100 rating)
    'satisfaction': 0.3,
    
    # How much to prioritize location quality
    # Higher value = prefer areas with more attractions/restaurants
    # Based on density of attractions and restaurants nearby
    'location': 0.2,
    
    # How much to prioritize distance to city center
    # Higher value = prefer listings closer to center
    # Based on kilometers from city center
    'distance': 0.2
}

def main():
    print("Loading Airbnb dataset...")
    try:
        ds = load_dataset("kraina/airbnb", "all", cache_dir=".cache")
        
        # Find optimal listing path using the weights defined above
        best_path, best_score = find_optimal_listing(ds, weights=WEIGHTS)
        
        # Print results
        print_listing_path(best_path, best_score)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    # Verify weights sum to 1.0
    if abs(sum(WEIGHTS.values()) - 1.0) > 0.001:
        raise ValueError("Weights must sum to 1.0")
    
    # Initialize multiprocessing support
    mp.freeze_support()
    main()