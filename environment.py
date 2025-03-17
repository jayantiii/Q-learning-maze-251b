import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy

class MazeEnvironment:    
    def __init__(self, maze, init_position, goal, vision_range=2, masked=False):  # Masked Q-Learning
        # Number of rows and columns in the maze
        x = len(maze)
        y = len(maze)
        
        self.boundary = np.asarray([x, y]) # Store maze dimensions
        self.init_position = init_position
        self.current_position = np.asarray(init_position)
        self.goal = goal
        self.maze = maze  # Maze structure in 2D grid
        
        # Track visited cells to penalize revisits
        self.visited = set()
        # Add initial position to visited set
        self.visited.add(tuple(self.current_position))
                
        # Identify all empty cells (value 0) in the maze
        self.allowed_states = np.asarray(np.where(self.maze == 0)).T.tolist()

        # Compute Euclidean distances from each allowed state to the goal
        self.distances = np.sqrt(np.sum((np.array(self.allowed_states) -
                                         np.asarray(self.goal))**2,
                                         axis = 1))
        
        # Remove the goal itself from the allowed states
        del(self.allowed_states[np.where(self.distances == 0)[0][0]])
        self.distances = np.delete(self.distances, np.where(self.distances == 0)[0][0])

        # Mapping of actions to movement vectors     
        self.action_map = {0: [0, 1], # Right
                           1: [0, -1], # Left
                           2: [1, 0], # Down
                           3: [-1, 0]} # Up
        
        # Mapping of direction symbols for visualization
        self.directions = {0: '→',
                           1: '←',
                           2: '↓ ',
                           3: '↑'}
        
        
        
        # Masked Q-Learning parameters
        self.vision_range = vision_range  
        self.masked = masked  

    # Reset policy: high epsilon the initial position is nearer to the goal 
    # (useful for large mazes)
    def reset_policy(self, eps, reg = 7):
        # Softmax distribution for initialization based on proximity to goal
        return sp.softmax(-self.distances/(reg*(1-eps**(2/reg)))**(reg/2)).squeeze()
    
    # Reset the environment when the game is completed 
    def reset(self, epsilon, prand = 0):
        if np.random.rand() < prand: # Random reset with probability prand
            idx = np.random.choice(len(self.allowed_states))
        else: # Use reset policy for targeted starting points
            p = self.reset_policy(epsilon)
            idx = np.random.choice(len(self.allowed_states), p = p)

        # Update new starting position
        self.current_position = np.asarray(self.allowed_states[idx])
        # Clear visited set
        self.visited = set()
        # Add starting point to visited set
        self.visited.add(tuple(self.current_position))

        return self.state()
    
    # Update the agent's state based on the chosen action
    def state_update(self, action):
        isgameon = True
        
        # Default move penalty: each move costs -0.05
        reward = -0.05
        
        # Map action to movement vector
        move = self.action_map[action]
        # Compute next position
        next_position = self.current_position + np.asarray(move)
        
        # Check if the goal is reached
        if (self.current_position == self.goal).all():
            reward = 1 # Reward for reaching the goal
            isgameon = False # Game over condition
            return [self.state(), reward, isgameon]
            
        # Penalize revisiting cells with reward -0.2
        else:
            if tuple(self.current_position) in self.visited:
                reward = -0.2
        
        # Penalize invalid moves with reward -1 
        # (if the move goes out of the maze or to a wall)
        if self.is_state_valid(next_position):
            self.current_position = next_position
        else:
            reward = -1
        
        # Mark new cell as visited
        self.visited.add(tuple(self.current_position))

        return [self.state(), reward, isgameon]

    # return the current state for the agent
    def state(self):  
        if self.masked:  # Use masked state
            return self.get_masked_state()
        else: # Provide the full maze view
            state = copy.deepcopy(self.maze)
            state[tuple(self.current_position)] = 2 # Mark agent position in maze
            return state

    # Masked Q-Learning: Extract a local view of the maze
    def get_masked_state(self):  
        # Agent's current position
        x, y = self.current_position
        # Define vision boundaries (within maze limits)
        x_min, x_max = max(0, x - self.vision_range), min(self.boundary[0], x + self.vision_range + 1)
        y_min, y_max = max(0, y - self.vision_range), min(self.boundary[1], y + self.vision_range + 1)

        # Initialize unseen areas as -1
        masked_view = np.full(self.maze.shape, -1)  
        # Fill visible areas
        masked_view[x_min:x_max, y_min:y_max] = self.maze[x_min:x_max, y_min:y_max]
        # Mark agent position
        masked_view[tuple(self.current_position)] = 2  

        return masked_view
    
    # Check if the given position is outside maze boundaries
    def check_boundaries(self, position):
        # Negative coordinates
        out = len([num for num in position if num < 0])
        # Beyond limits
        out += len([num for num in (self.boundary - np.asarray(position)) if num <= 0])
        return out > 0
    
    # Check if the given position is a wall
    def check_walls(self, position):
        return self.maze[tuple(position)] == 1
    
    # Validate if a given state is within maze limits and not a wall
    def is_state_valid(self, next_position):
        if self.check_boundaries(next_position):
            return False
        elif self.check_walls(next_position):
            return False
        return True

    # Action masking
    def get_valid_actions(self, position):
        """
        Returns a binary array where valid action = 1 and invalid action = 0.
        """
        # Initialize
        valid_actions = np.zeros(4, dtype=np.int32)
    
        # Check each direction (up, right, down, left)
        for action in range(4):
            move = self.action_map[action]
            next_position = position + np.array(move)
            
            # Mark 0 or 1
            if self.is_state_valid(next_position):
                valid_actions[action] = 1
        
        return valid_actions
    
    # Draw the maze
    def draw(self, filename):
        plt.figure()
        # Maze visualization
        im = plt.imshow(self.maze, interpolation='none', aspect='equal', cmap='Greys');
        ax = plt.gca() # Get current axis

        # Hide x and y-axis ticks
        plt.xticks([], [])
        plt.yticks([], [])

        # Plot goal in blue
        ax.plot(self.goal[1], self.goal[0], 'bs', markersize = 4)
        # Plot agent in red
        ax.plot(self.current_position[1], self.current_position[0], 'rs', markersize = 4)
        plt.savefig(filename, dpi = 300, bbox_inches = 'tight')
        plt.show()

