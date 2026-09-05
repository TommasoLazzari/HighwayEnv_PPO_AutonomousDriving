import gymnasium
import highway_env


# Remember to save what you will need for the plots

env_name = "highway-v0"
env = gymnasium.make(env_name,
                     config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                     render_mode='human')

env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

while episode <= 10:
    episode_steps += 1

    # Hint: take a look at the docs to see the difference between 'done' and 'truncated'
    _, reward, done, truncated, _ = env.step(env.action_space.sample())  # With manual control these actions are ignored
    env.render()

    episode_return += reward

    if done or truncated:
        print(f"Episode Num: {episode} Episode T: {episode_steps} Return: {episode_return:.3f}, Crash: {done}")

        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()

'''
Episode Num: 2 Episode T: 40 Return: 32.919, Crash: False
Episode Num: 3 Episode T: 40 Return: 31.213, Crash: False
Episode Num: 4 Episode T: 40 Return: 34.069, Crash: False
Episode Num: 5 Episode T: 40 Return: 34.417, Crash: False
Episode Num: 6 Episode T: 40 Return: 38.571, Crash: False
Episode Num: 7 Episode T: 40 Return: 34.224, Crash: False
Episode Num: 8 Episode T: 23 Return: 21.828, Crash: True
Episode Num: 9 Episode T: 40 Return: 36.498, Crash: False
Episode Num: 10 Episode T: 40 Return: 37.351, Crash: False
'''