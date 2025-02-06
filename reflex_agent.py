import random
from enum import Enum

class Action(Enum):
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.agent_pos = (0, 0)
        self.dirt_locations = set()
        self.history = []  # Track history for visualization
        
        # Randomly place dirt
        for _ in range(size * 2):
            x = random.randint(0, size-1)
            y = random.randint(0, size-1)
            self.dirt_locations.add((x, y))
    
    def is_dirty(self, pos):
        return pos in self.dirt_locations
    
    def clean_location(self, pos):
        self.dirt_locations.discard(pos)
    
    def move_agent(self, action):
        x, y = self.agent_pos
        
        if action == Action.UP and x > 0:
            self.agent_pos = (x-1, y)
        elif action == Action.DOWN and x < self.size-1:
            self.agent_pos = (x+1, y)
        elif action == Action.LEFT and y > 0:
            self.agent_pos = (x, y-1)
        elif action == Action.RIGHT and y < self.size-1:
            self.agent_pos = (x, y+1)
    
    def display(self, clear_screen=True):
        """Enhanced display with better visuals"""
        if clear_screen:
            import os
            os.system('clear' if os.name == 'posix' else 'cls')
        
        print("+" + "-" * (self.size * 2 + 1) + "+")
        for i in range(self.size):
            print("|", end=' ')
            for j in range(self.size):
                if (i, j) == self.agent_pos:
                    print('A', end=' ')  # Agent
                elif (i, j) in self.dirt_locations:
                    print('D', end=' ')  # Dirt
                else:
                    print('.', end=' ')  # Clean space
            print("|")
        print("+" + "-" * (self.size * 2 + 1) + "+")
        
        # Display legend
        print("\nLegend:")
        print("A - Agent")
        print("D - Dirt")
        print(". - Clean space")
        print()
    
    def save_state(self):
        """Save current state for history"""
        self.history.append({
            'agent_pos': self.agent_pos,
            'dirt_locations': self.dirt_locations.copy()
        })

class ReflexVacuumAgent:
    def __init__(self):
        self.last_action = None
    
    def get_action(self, pos, is_dirty):
        """
        Simple reflex agent decision making:
        - If current location is dirty, clean it
        - Otherwise, move to a random adjacent location
        """
        if is_dirty:
            return 'CLEAN'
        
        return random.choice(list(Action))

def run_simulation(steps=40):
    env = Environment(size=5)
    agent = ReflexVacuumAgent()
    
    print("Initial environment:")
    env.display()
    env.save_state()
    
    import time
    
    for step in range(steps):
        print(f"Step {step + 1}:")
        
        # Get current state
        pos = env.agent_pos
        is_dirty = env.is_dirty(pos)
        
        # Get agent's action
        action = agent.get_action(pos, is_dirty)
        
        # Execute action and display status
        if action == 'CLEAN':
            print(f"Agent CLEANING at position {pos}")
            env.clean_location(pos)
        else:
            print(f"Agent MOVING {action.value} from {pos}")
            env.move_agent(action)
        
        # Save and display current state
        env.save_state()
        env.display()
        time.sleep(0.5)  # Pause to show each step
        
        # Display statistics
        print(f"Dirt remaining: {len(env.dirt_locations)}")
        print(f"Positions cleaned: {step + 1}")
        print()
        
        # Check if all dirt is cleaned
        if not env.dirt_locations:
            print("🎉 All locations are clean! 🎉")
            print(f"Cleaned in {step + 1} steps")
            break
    
    # Final statistics
    print("\nSimulation complete!")
    print(f"Final dirty locations: {len(env.dirt_locations)}")
    print(f"Total steps taken: {steps}")

if __name__ == "__main__":
    run_simulation() 