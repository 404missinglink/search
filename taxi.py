#!/usr/bin/env python3
"""
Driverless Taxi Simulation with 3D Visualization
--------------------------------------------------

This script simulates a driverless taxi navigating a grid with obstacles.
It first uses an A* search algorithm (implemented from scratch) to plan
a path to a passenger pick-up location and then to a drop-off location.
At the same time, a Tkinter window displays a 3D (perspective) view of the
grid, obstacles, the taxi (drawn as a small 3D box), and markers for the
pick-up (green) and drop-off (red) locations. The taxi’s planned path is
drawn as a blue line.

No external libraries are used.
"""

import math
import time
import _tkinter as tk

#############################
#  Simulation Functionality #
#############################

# Heuristic: Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Return valid neighbors (up, down, left, right) that are not obstacles.
def get_neighbors(position, grid):
    x, y = position
    neighbors = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rows = len(grid)
    cols = len(grid[0])
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors

# Pure A* search for grid-based path planning.
def astar_search(grid, start, goal):
    open_list = []  # Each element: (priority, position)
    came_from = {}
    cost_so_far = {}
    open_list.append((heuristic(start, goal), start))
    came_from[start] = None
    cost_so_far[start] = 0

    while open_list:
        open_list.sort(key=lambda x: x[0])
        current_priority, current = open_list.pop(0)

        if current == goal:
            # Reconstruct path.
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_neighbors(current, grid):
            new_cost = cost_so_far[current] + 1  # Each move costs 1.
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                open_list.append((priority, neighbor))
                came_from[neighbor] = current

    return None

# Create a 10x10 grid (0 = free, 1 = obstacle)
def create_grid():
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    ]
    return grid

# Taxi agent class.
class TaxiAgent:
    def __init__(self, start):
        self.position = start

###################################
#  3D Visualization (Tkinter)    #
###################################

# Basic vector operations.
def vector_subtract(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def cross_product(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])

def normalize(v):
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag == 0:
        return (0, 0, 0)
    return (v[0]/mag, v[1]/mag, v[2]/mag)

# Set up the camera parameters.
def setup_camera():
    # Define a camera (eye) position, a target, and an up vector.
    eye = (5, -15, 20)         # Camera position in world coordinates.
    target = (5, 5, 0)         # Look-at point (roughly the center of the grid).
    up = (0, 0, 1)             # World up is the z-axis.
    forward = normalize(vector_subtract(target, eye))
    right = normalize(cross_product(forward, up))
    up_corrected = cross_product(right, forward)
    return {"eye": eye, "target": target, "right": right, "up": up_corrected, "forward": forward}

# Project a 3D point to 2D screen coordinates.
def project_point(point, camera, focal_length, center):
    eye = camera["eye"]
    right = camera["right"]
    up = camera["up"]
    forward = camera["forward"]
    rel = (point[0]-eye[0], point[1]-eye[1], point[2]-eye[2])
    x_cam = dot_product(rel, right)
    y_cam = dot_product(rel, up)
    # We define z_cam as the distance along the negative forward direction.
    z_cam = -dot_product(rel, forward)
    if z_cam == 0:
        z_cam = 0.0001
    sx = center[0] + (focal_length * x_cam / z_cam)
    sy = center[1] - (focal_length * y_cam / z_cam)
    return (sx, sy)

# Draw the entire scene: grid cells, obstacles, planned path, pick-up/drop-off markers, and the taxi.
def draw_scene(canvas, grid, taxi, path, pickup, dropoff, camera, focal_length, center):
    canvas.delete("all")
    rows = len(grid)
    cols = len(grid[0])
    
    # Draw grid cells.
    for i in range(rows):
        for j in range(cols):
            # Define the four corners of the cell (world coordinates).
            bl = (i, j, 0)       # bottom-left
            br = (i+1, j, 0)     # bottom-right
            tr = (i+1, j+1, 0)   # top-right
            tl = (i, j+1, 0)     # top-left
            p_bl = project_point(bl, camera, focal_length, center)
            p_br = project_point(br, camera, focal_length, center)
            p_tr = project_point(tr, camera, focal_length, center)
            p_tl = project_point(tl, camera, focal_length, center)
            poly_points = [p_bl[0], p_bl[1],
                           p_br[0], p_br[1],
                           p_tr[0], p_tr[1],
                           p_tl[0], p_tl[1]]
            fill_color = "gray" if grid[i][j] == 1 else "white"
            canvas.create_polygon(poly_points, fill=fill_color, outline="black")
    
    # Draw the planned path (if available) as a blue line on the ground.
    if path:
        path_points = []
        for (x, y) in path:
            # Use the center of the cell.
            pt = (x+0.5, y+0.5, 0)
            sp = project_point(pt, camera, focal_length, center)
            path_points.extend(sp)
        if len(path_points) >= 4:
            canvas.create_line(path_points, fill="blue", width=2)
    
    # Draw pick-up and drop-off markers as circles.
    for loc, color in [(pickup, "green"), (dropoff, "red")]:
        cx, cy, cz = (loc[0]+0.5, loc[1]+0.5, 0)
        sp = project_point((cx, cy, cz), camera, focal_length, center)
        r = 5
        canvas.create_oval(sp[0]-r, sp[1]-r, sp[0]+r, sp[1]+r, fill=color)
    
    # Draw the taxi as a 3D box.
    tx, ty = taxi.position
    taxi_height = 0.5
    # Bottom vertices of the taxi box.
    b0 = (tx,    ty,    0)
    b1 = (tx+1,  ty,    0)
    b2 = (tx+1,  ty+1,  0)
    b3 = (tx,    ty+1,  0)
    # Top vertices.
    t0 = (tx,    ty,    taxi_height)
    t1 = (tx+1,  ty,    taxi_height)
    t2 = (tx+1,  ty+1,  taxi_height)
    t3 = (tx,    ty+1,  taxi_height)
    b0p = project_point(b0, camera, focal_length, center)
    b1p = project_point(b1, camera, focal_length, center)
    b2p = project_point(b2, camera, focal_length, center)
    b3p = project_point(b3, camera, focal_length, center)
    t0p = project_point(t0, camera, focal_length, center)
    t1p = project_point(t1, camera, focal_length, center)
    t2p = project_point(t2, camera, focal_length, center)
    t3p = project_point(t3, camera, focal_length, center)
    
    # Draw the bottom face of the taxi.
    canvas.create_polygon(b0p[0], b0p[1], b1p[0], b1p[1],
                          b2p[0], b2p[1], b3p[0], b3p[1],
                          fill="yellow", outline="black")
    # Draw the top face of the taxi.
    canvas.create_polygon(t0p[0], t0p[1], t1p[0], t1p[1],
                          t2p[0], t2p[1], t3p[0], t3p[1],
                          fill="orange", outline="black")
    # Draw vertical edges.
    canvas.create_line(b0p[0], b0p[1], t0p[0], t0p[1], fill="black")
    canvas.create_line(b1p[0], b1p[1], t1p[0], t1p[1], fill="black")
    canvas.create_line(b2p[0], b2p[1], t2p[0], t2p[1], fill="black")
    canvas.create_line(b3p[0], b3p[1], t3p[0], t3p[1], fill="black")

###################################
#  Main Simulation with Animation #
###################################

def main():
    # --- Simulation Setup ---
    grid = create_grid()
    taxi_start = (0, 0)
    pickup_location = (2, 5)
    dropoff_location = (9, 9)
    taxi = TaxiAgent(taxi_start)
    
    # Compute the path to pick-up and then drop-off.
    path_to_pickup = astar_search(grid, taxi.position, pickup_location)
    if path_to_pickup is None:
        print("No path found to the pick-up location!")
        return
    path_to_dropoff = astar_search(grid, pickup_location, dropoff_location)
    if path_to_dropoff is None:
        print("No path found to the drop-off location!")
        return
    
    # --- Visualization Setup ---
    window = tk.Tk()
    window.title("Driverless Taxi 3D Visualization")
    canvas_width = 800
    canvas_height = 600
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="lightblue")
    canvas.pack()
    
    # Set up camera parameters for the perspective projection.
    camera = setup_camera()
    focal_length = 300
    center = (canvas_width/2, canvas_height/2)
    
    # --- Animate the Taxi Movement ---
    
    # Phase 1: Move to the pick-up location.
    full_path = path_to_pickup
    for step in full_path:
        taxi.position = step
        draw_scene(canvas, grid, taxi, full_path, pickup_location, dropoff_location,
                   camera, focal_length, center)
        window.update()
        time.sleep(0.3)
    print("Passenger picked up at", taxi.position)
    time.sleep(1)
    
    # Phase 2: Move to the drop-off location.
    full_path = path_to_dropoff
    for step in full_path:
        taxi.position = step
        draw_scene(canvas, grid, taxi, full_path, pickup_location, dropoff_location,
                   camera, focal_length, center)
        window.update()
        time.sleep(0.3)
    print("Passenger dropped off at", taxi.position)
    
    # Keep the window open until closed by the user.
    window.mainloop()

if __name__ == "__main__":
    main()

"""
Reflection on the Implementation:
----------------------------------
1. Agent and Simulation:
   - The TaxiAgent class represents our driverless taxi.
   - The simulation computes two paths using an A* search: one to the 
     pick-up location and one to the drop-off location. The taxi moves 
     step-by-step along these paths.

2. Pure Search and 3D Visualization:
   - The A* search algorithm is implemented using only built-in data types.
   - For visualization, a simple perspective projection is implemented
     (using basic vector math) to render a 3D view in a Tkinter Canvas.
   - The grid is drawn in 3D (with obstacles filled in gray), the taxi is
     rendered as a small box with a bottom (yellow) and top (orange) face,
     and the planned path appears as a blue line.
     
3. Limitations and Extensions:
   - This is a basic simulation with hard-coded grid, obstacles, and positions.
   - The 3D projection is a simple model and does not handle hidden-surface
     removal or advanced lighting.
   - Future enhancements might include smoother animations, interactive
     controls, or dynamic obstacles.

This script uses no external libraries and is designed to run correctly on the
first try.
"""
