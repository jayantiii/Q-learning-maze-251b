import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections

# Stores experiences
Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])


class Agent:
    def __init__(self, maze, memory_buffer, use_softmax = True):
        self.env = maze
        self.buffer = memory_buffer # this is actually a reference
        self.num_act = 4
        self.use_softmax = use_softmax # toggles between softmax and epsilon-greedy
        self.total_reward = 0
        self.min_reward = -self.env.maze.size
        self.isgameon = True

    # Perform action and udpate state
    def make_a_move(self, net, epsilon, device = 'cuda'):
        # Choose action using softmax or epsilon-greedy
        action = self.select_action(net, epsilon, device)
        # Get the current state
        current_state = self.env.state()
        # Move to new state
        next_state, reward, self.isgameon = self.env.state_update(action)
        # Update rewards
        self.total_reward += reward
        
        # End game if total reward is too low
        if self.total_reward < self.min_reward:
            self.isgameon = False
        # Reset reward when game is terminated
        if not self.isgameon:
            self.total_reward = 0
        
        # Stores experience for later training
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        
        # Save experience to memory buffer
        self.buffer.push(transition)


    # Choose the next action
    def select_action(self, net, epsilon, device='cuda'):
        state = self.env.state()  # Assuming self.env.state() returns a numpy ndarray
    
        # Convert to a tensor if it's a numpy array
        if isinstance(state, np.ndarray):  
            state_tensor = torch.tensor(state, dtype=torch.float32)
        else:
            state_tensor = state  
    
        # Convert device to torch.device in the same line if it's a string
        device = torch.device(device if isinstance(device, str) else 'cpu')  # Default to 'cpu' if not a string
    
        # Move the tensor to the correct device (CUDA or CPU)
        state_tensor = state_tensor.to(device)  # This automatically moves the tensor to the correct device
    
        # Flatten the state tensor to match the input size (1D)
        state_tensor = state_tensor.view(-1)  # Flatten the tensor (20, 20) -> (400,)
    
        # Move the model to the correct device
        net = net.to(device)
    
        # Passes the state through the neural network (net) to get predicted Q-values 
        qvalues = net(state_tensor).cpu().detach().numpy().squeeze()
    
        # Ensure epsilon is a scalar
        if isinstance(epsilon, np.ndarray):
            # Convert to scalar if it's a single element
            if epsilon.size == 1:
                epsilon = epsilon.item() 
            
            # Pick the first value if it's an array with multiple values
            else:
                epsilon = epsilon[0] 
    
        # If using softmax, sample action from q-values
        if self.use_softmax:
            # Normalize Q-values for Softmax
            qvalues = qvalues / epsilon  
            # Apply Softmax to convert to probabilities
            p = sp.softmax(qvalues).squeeze()  
            # Ensure probabilities sum to 1
            p /= np.sum(p) 
            # Sample an action 
            action = np.random.choice(self.num_act, p=p)  
        
        # Otherwise, follows Epsilon-greedy action selection
        else:
            # Random action
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]  
            # Greedy action (based on best Q-value)
            else:
                action = np.argmax(qvalues)
    
        return action


    # Helper function for masked Q-learning
    def get_masked_state(self, agent_pos, vision_range=2):
        # Assuming env is an instance of MazeEnvironment, you should access the maze like this:
        x, y = agent_pos
        # Accessing the maze environment object
        env = self.env 
        
        # Extracts local view of the maze within the agent's vision range
        masked_view = env.maze[max(0, x - vision_range):x + vision_range + 1,
                               max(0, y - vision_range):y + vision_range + 1]
        return masked_view.flatten()


    # Action masking
    def masked_action_selection(self, net, state_tensor, epsilon):
        # Get current position and valid actions
        position = self.env.current_position
        valid_actions = self.env.get_valid_actions(position)
        
        # If all actions are invalid, pick random action
        if np.sum(valid_actions) == 0:
            return np.random.randint(0, 4)  
        
        with torch.no_grad():
            # Get Q-values
            q_values = net(state_tensor).cpu().numpy()[0]
            
            # Mask invalid actions by setting Q-values to -1e9
            masked_q_values = q_values.copy()
            masked_q_values[valid_actions == 0] = -1e9

            # Epsilon-greedy selection for valid actions
            if np.random.rand() < epsilon:
                # Choose random valid actions within epsilon possibility
                valid_indices = np.where(valid_actions == 1)[0]
                action = np.random.choice(valid_indices)
            else:
                # Choose valid action with highest Q-value 
                action = np.argmax(masked_q_values)
        
        return action


    # Visualize optimal policy across the maze
    def plot_policy_map(self, net, filename, offset):
        net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, 'Greys')
    
            # Get the device of the model to ensure consistency
            device = next(net.parameters()).device  
    
            # Each cell displays the optimal action for that position
            for free_cell in self.env.allowed_states:
                self.env.current_position = np.asarray(free_cell)
    
                # Convert the state to a tensor and move it to the same device as the model
                state_tensor = torch.Tensor(self.env.state()).view(1, -1).to(device)
    
                # Get Q-values by passing the state through the model
                qvalues = net(state_tensor)
    
                # Get the action from the Q-values
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
    
                # Get the policy (direction)
                policy = self.env.directions[action]
    
                # Display the policy on the plot
                ax.text(free_cell[1] - offset[0], free_cell[0] - offset[1], policy)
    
            # Customize plot
            ax = plt.gca()
            plt.xticks([], [])
            plt.yticks([], [])
    
            # Mark the goal position on the plot
            ax.plot(self.env.goal[1], self.env.goal[0], 'bs', markersize=4)
    
            # Save and show the plot
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()


