from environment import MazeEnv
from agent import QAgent

# Set masked=True to enable Masked Q-Learning
USE_MASKED_Q_LEARNING = True  

# Initialize environment with masked flag
env = MazeEnv(grid_size=10, vision_range=2, masked=USE_MASKED_Q_LEARNING)
state_size = env.grid_size ** 2  # Flattened state space
action_size = 4  # UP, DOWN, LEFT, RIGHT
agent = QAgent(state_size, action_size)

# Train the agent
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
