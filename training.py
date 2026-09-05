import gymnasium
import highway_env
import numpy as np
import torch
import random
import os 
import json 

from ppo_agent import PPOAgent


# Set the seed and create the environment
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

MAX_STEPS = 60000  
ROLLOUT_STEPS = 2048  
SAVE_EVERY_EPISODES = 10

env_name = "highway-fast-v0"  # We use the 'fast' env just for faster training, if you want you can use "highway-v0"


'''
to train agents in different environments (e.g. nr of lanes), modify the following line
'''
env = gymnasium.make(env_name,config={'action': {'type': 'DiscreteMetaAction'}, 'lanes_count': 3, 'duration': 40, "vehicles_count": 50})

env.action_space.seed(0)

state, _ = env.reset(seed=0)
state = state.reshape(-1)

state_dim = state.shape[0]
action_dim = env.action_space.n

#create output folders
os.makedirs("weights", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Initialize your model
agent = PPOAgent(
    state_dim =state_dim,
    action_dim = action_dim,
    hidden_dim= 256,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_epsilon = 0.2,
    policy_lr = 3e-4,
    value_coef = 0.5,
    update_epochs= 10,
    batch_size = 64
)


episode = 1
episode_steps = 0
episode_return = 0

training_log = []

# Training loop
for t in range(MAX_STEPS):
    episode_steps += 1
    
    # Linear learning rate annealing with a floor at 5e-5
    frac = 1.0 - (t / MAX_STEPS)
    lrnow = max(frac * 3e-4, 5e-5)
    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = lrnow

    # Select the action to be performed by the agent
    action, log_prob, value = agent.select_action(state)

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = next_state.reshape(-1)
    
    #if the episode ends due to a crash, we penalize it, in order to prioritize survival over speed and lane changes
    shaped_reward = reward
    
    '''
    to include / remove the crash malus from the reward function uncomment / comment the following two lines
    '''
    if done:
        shaped_reward -= 5.0
    
    terminal = done or truncated

    agent.store_transition(state=state,action=action,reward=shaped_reward, done=terminal,log_prob=log_prob,value=value)

    state = next_state
    episode_return += reward
    
    #PPO update as soon as the buffer is long enough
    if len(agent.buffer) >= ROLLOUT_STEPS:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            _, _, next_value = agent.model.get_action_and_value(state_tensor)
        agent.update(next_value=next_value.item())

    if terminal:
        print(f"Total T: {t} Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f} Crash: {done}")
        
        training_log.append({"total_steps": t,"episode": episode,"episode_steps": episode_steps,"episode_return": float(episode_return),"crash": bool(done)})

        if episode % SAVE_EVERY_EPISODES == 0:
            agent.save("weights//ppo_agent_latest.pt")
            
            with open("results/training_log.json", "w") as f:
                json.dump(training_log, f, indent=4)

        state, _ = env.reset()
        state = state.reshape(-1)
        
        episode += 1
        episode_steps = 0
        episode_return = 0
        
#final update for remaining transition
agent.update()

#save final model and training log
agent.save("weights//ppo_agent_final.pt")

with open("results/training_log.json", "w") as f:
    json.dump(training_log, f, indent=4)

env.close()
