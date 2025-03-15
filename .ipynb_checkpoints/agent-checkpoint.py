import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections

Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])


class Agent:
    def __init__(self, maze, memory_buffer, use_softmax = True):
        self.env = maze
        self.buffer = memory_buffer # this is actually a reference
        self.num_act = 4
        self.use_softmax = use_softmax
        self.total_reward = 0
        self.min_reward = -self.env.maze.size
        self.isgameon = True

        
    def make_a_move(self, net, epsilon, device = 'cuda'):
        action = self.select_action(net, epsilon, device)
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)
        self.total_reward += reward
        
        if self.total_reward < self.min_reward:
            self.isgameon = False
        if not self.isgameon:
            self.total_reward = 0
        
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        
        self.buffer.push(transition)



    def select_action(self, net, epsilon, device='cuda'):
        state = self.env.state()  # Assuming self.env.state() returns a numpy ndarray
    
        # Convert state to torch.Tensor if it isn't already one
        if isinstance(state, np.ndarray):  # If it's a numpy array, convert to a tensor
            state_tensor = torch.tensor(state, dtype=torch.float32)
        else:
            state_tensor = state  # If it's already a torch tensor, no need to convert
    
        # Convert device to torch.device in the same line if it's a string
        device = torch.device(device if isinstance(device, str) else 'cpu')  # Default to 'cpu' if not a string
    
        # Move the tensor to the correct device (CUDA or CPU)
        state_tensor = state_tensor.to(device)  # This automatically moves the tensor to the correct device
    
        # Flatten the state tensor to match the input size expected by the network
        state_tensor = state_tensor.view(-1)  # Flatten the tensor (20, 20) -> (400,)
    
        # Move the model to the correct device
        net = net.to(device)
    
        # Now use the network to get Q-values (output of the model)
        qvalues = net(state_tensor).cpu().detach().numpy().squeeze()
    
        # Ensure epsilon is a scalar (if it's not a scalar, pick the first element)
        if isinstance(epsilon, np.ndarray):
            if epsilon.size == 1:
                epsilon = epsilon.item()  # Convert to scalar if it's a single element
            else:
                epsilon = epsilon[0]  # Pick the first value if it's an array with multiple values
    
        # If using softmax, sample action from q-values
        if self.use_softmax:
            qvalues = qvalues / epsilon  # Normalize with epsilon
            p = sp.softmax(qvalues).squeeze()  # Softmax normalization
            p /= np.sum(p)  # Ensure probabilities sum to 1
            action = np.random.choice(self.num_act, p=p)  # Sample action based on softmax probabilities
        else:
            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]  # Random action
            else:
                action = np.argmax(qvalues)  # Greedy action (based on best Q-value)
    
        return action


    # In Agent class, make sure get_masked_state() is defined correctly
    def get_masked_state(self, agent_pos, vision_range=2):
        # Assuming env is an instance of MazeEnvironment, you should access the maze like this:
        x, y = agent_pos
        env = self.env  # Accessing the maze environment object
        
        # Ensure that `env` is the maze environment and subscriptable
        masked_view = env.maze[max(0, x - vision_range):x + vision_range + 1,
                               max(0, y - vision_range):y + vision_range + 1]
        return masked_view.flatten()

    
    def plot_policy_map(self, net, filename, offset):
        net.eval()
        with torch.no_grad():
            fig, ax = plt.subplots()
            ax.imshow(self.env.maze, 'Greys')
    
            # Get the device of the model to ensure consistency
            device = next(net.parameters()).device  # This automatically gets the device of the model (CPU or GPU)
    
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


