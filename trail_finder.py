from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import random
from queue import PriorityQueue
import folium

@dataclass
class Trail:
    name: str
    distance: float  # in miles
    elevation_gain: int  # in feet
    difficulty: int  # 1-5 scale
    scenic_rating: int  # 1-5 scale
    features: List[str]  # e.g., ["waterfall", "lake", "forest"]
    coordinates: Tuple[float, float]  # (latitude, longitude)
    
    def __str__(self):
        return (f"{self.name}: {self.distance}mi, {self.elevation_gain}ft gain, "
                f"Difficulty: {self.difficulty}/5, Scenic: {self.scenic_rating}/5")

class TrailNode:
    def __init__(self, trail: Optional[Trail] = None):
        self.trail = trail
        self.children: Dict[str, TrailNode] = {}
        self.decision_criteria: Optional[str] = None
        
    def add_child(self, criteria: str, node: 'TrailNode'):
        self.children[criteria] = node

class TrailDecisionTree:
    def __init__(self):
        self.root = TrailNode()
        
    def build_tree(self, trails: List[Trail]):
        """Build decision tree based on trail characteristics"""
        self.root = self._build_tree_recursive(trails)
        
    def _build_tree_recursive(self, trails: List[Trail]) -> TrailNode:
        if not trails:
            return TrailNode()
            
        if len(trails) == 1:
            return TrailNode(trails[0])
            
        # Choose splitting criteria based on variance
        criteria_options = {
            'distance': self._variance([t.distance for t in trails]),
            'elevation': self._variance([t.elevation_gain for t in trails]),
            'difficulty': self._variance([t.difficulty for t in trails]),
            'scenic': self._variance([t.scenic_rating for t in trails])
        }
        
        best_criteria = max(criteria_options.items(), key=lambda x: x[1])[0]
        node = TrailNode()
        node.decision_criteria = best_criteria
        
        # Split trails based on criteria
        if best_criteria == 'distance':
            values = [t.distance for t in trails]
            median = sorted(values)[len(values)//2]
            easier = [t for t in trails if t.distance < median]
            harder = [t for t in trails if t.distance >= median]
            
            # If we can't split effectively, just return a leaf node with a random trail
            if not easier or not harder:
                return TrailNode(random.choice(trails))
                
            node.add_child('shorter', self._build_tree_recursive(easier))
            node.add_child('longer', self._build_tree_recursive(harder))
            
        elif best_criteria == 'elevation':
            values = [t.elevation_gain for t in trails]
            median = sorted(values)[len(values)//2]
            easier = [t for t in trails if t.elevation_gain < median]
            harder = [t for t in trails if t.elevation_gain >= median]
            
            if not easier or not harder:
                return TrailNode(random.choice(trails))
                
            node.add_child('less_elevation', self._build_tree_recursive(easier))
            node.add_child('more_elevation', self._build_tree_recursive(harder))
            
        elif best_criteria == 'difficulty':
            values = [t.difficulty for t in trails]
            median = sorted(values)[len(values)//2]
            easier = [t for t in trails if t.difficulty < median]
            harder = [t for t in trails if t.difficulty >= median]
            
            if not easier or not harder:
                return TrailNode(random.choice(trails))
                
            node.add_child('easier', self._build_tree_recursive(easier))
            node.add_child('harder', self._build_tree_recursive(harder))
            
        else:  # scenic
            values = [t.scenic_rating for t in trails]
            median = sorted(values)[len(values)//2]
            less_scenic = [t for t in trails if t.scenic_rating < median]
            more_scenic = [t for t in trails if t.scenic_rating >= median]
            
            if not less_scenic or not more_scenic:
                return TrailNode(random.choice(trails))
                
            node.add_child('less_scenic', self._build_tree_recursive(less_scenic))
            node.add_child('more_scenic', self._build_tree_recursive(more_scenic))
            
        return node
    
    @staticmethod
    def _variance(values: List[float]) -> float:
        """Calculate variance of a list of values"""
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def find_trail(self, preferences: Dict[str, str]) -> Optional[Trail]:
        """Search for a trail matching given preferences"""
        current = self.root
        
        while current.trail is None and current.children:
            if not current.decision_criteria:
                break
                
            criterion = current.decision_criteria
            if criterion not in preferences:
                # If preference not specified, choose random path
                choice = random.choice(list(current.children.keys()))
            else:
                choice = preferences[criterion]
                
            if choice in current.children:
                current = current.children[choice]
            else:
                break
                
        return current.trail

def visualize_trail(trail: Trail) -> None:
    """Create an interactive map showing the trail location"""
    # Create a map centered on the trail
    trail_map = folium.Map(location=trail.coordinates, zoom_start=13)
    
    # Add a marker for the trail
    folium.Marker(
        trail.coordinates,
        popup=f"<b>{trail.name}</b><br>"
              f"Distance: {trail.distance}mi<br>"
              f"Elevation Gain: {trail.elevation_gain}ft<br>"
              f"Difficulty: {trail.difficulty}/5<br>"
              f"Scenic Rating: {trail.scenic_rating}/5<br>"
              f"Features: {', '.join(trail.features)}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(trail_map)
    
    # Save the map
    map_file = f"{trail.name.lower().replace(' ', '_')}_map.html"
    trail_map.save(map_file)
    print(f"\nMap has been saved to {map_file}")

def generate_sample_trails() -> List[Trail]:
    """Generate sample trail data"""
    return [
        Trail("Emerald Lake", 3.2, 700, 2, 5, ["lake", "forest", "mountains"], 
             (40.3099, -105.6457)),
        Trail("Sky Pond", 9.8, 1780, 4, 5, ["lake", "waterfall", "mountains"],
             (40.2989, -105.6601)),
        Trail("Bear Lake Loop", 0.7, 50, 1, 3, ["lake", "forest"],
             (40.3121, -105.6473)),
        Trail("Longs Peak", 14.8, 5100, 5, 5, ["mountains", "alpine"],
             (40.2549, -105.6167)),
        Trail("Alberta Falls", 1.6, 200, 2, 4, ["waterfall", "forest"],
             (40.3102, -105.6401)),
        Trail("Deer Mountain", 6.2, 1210, 3, 4, ["mountains", "forest"],
             (40.3869, -105.6011)),
        Trail("Cascade Falls", 7.4, 1001, 3, 4, ["waterfall", "forest"],
             (40.3527, -105.6102)),
        Trail("Chasm Lake", 8.5, 2500, 4, 5, ["lake", "mountains", "alpine"],
             (40.2720, -105.5561)),
        Trail("Gem Lake", 3.1, 990, 3, 4, ["lake", "rocks"],
             (40.4083, -105.5194)),
        Trail("Ouzel Falls", 5.3, 870, 2, 4, ["waterfall", "forest"],
             (40.2019, -105.5866))
    ]

def display_all_trails(trails: List[Trail]) -> None:
    """Display all available trails with their details"""
    print("\nAvailable Trails:")
    print("-" * 80)
    for i, trail in enumerate(trails, 1):
        features_str = ", ".join(trail.features)
        print(f"{i}. {trail.name}")
        print(f"   Distance: {trail.distance}mi")
        print(f"   Elevation Gain: {trail.elevation_gain}ft")
        print(f"   Difficulty: {trail.difficulty}/5")
        print(f"   Scenic Rating: {trail.scenic_rating}/5")
        print(f"   Features: {features_str}")
        print()

def get_user_preferences() -> Dict[str, str]:
    """Get trail preferences from user input"""
    preferences = {}
    
    print("\nPlease enter your preferences (press Enter to skip any preference):")
    
    # Distance preference
    distance = input("Distance preference (shorter/longer): ").lower().strip()
    if distance in ['shorter', 'longer']:
        preferences['distance'] = distance
    
    # Elevation preference
    elevation = input("Elevation preference (less_elevation/more_elevation): ").lower().strip()
    if elevation in ['less_elevation', 'more_elevation']:
        preferences['elevation'] = elevation
    
    # Difficulty preference
    difficulty = input("Difficulty preference (easier/harder): ").lower().strip()
    if difficulty in ['easier', 'harder']:
        preferences['difficulty'] = difficulty
    
    # Scenic preference
    scenic = input("Scenic preference (less_scenic/more_scenic): ").lower().strip()
    if scenic in ['less_scenic', 'more_scenic']:
        preferences['scenic'] = scenic
    
    return preferences

def main():
    # Generate sample trails
    trails = generate_sample_trails()
    
    while True:
        print("\n=== Trail Finder ===")
        print("1. View all trails")
        print("2. Search for trails based on preferences")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            display_all_trails(trails)
            
        elif choice == '2':
            # Build decision tree
            tree = TrailDecisionTree()
            tree.build_tree(trails)
            
            # Get user preferences
            preferences = get_user_preferences()
            
            if not preferences:
                print("\nNo preferences specified. Please try again with at least one preference.")
                continue
            
            # Find matching trail
            recommended_trail = tree.find_trail(preferences)
            
            if recommended_trail:
                print("\nRecommended trail based on your preferences:")
                print(recommended_trail)
                
                # Ask if user wants to see the map
                show_map = input("\nWould you like to see this trail on a map? (yes/no): ").lower().strip()
                if show_map.startswith('y'):
                    visualize_trail(recommended_trail)
            else:
                print("\nNo trail found matching all preferences.")
                
        elif choice == '3':
            print("\nThank you for using Trail Finder!")
            break
            
        else:
            print("\nInvalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()