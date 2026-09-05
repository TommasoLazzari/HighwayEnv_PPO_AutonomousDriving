import json
import random
from pathlib import Path

import gymnasium
import highway_env  
import numpy as np
import torch


# Reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


# Environment
env_name = "highway-fast-v0"

env = gymnasium.make(env_name,config={"action": {"type": "DiscreteMetaAction"},"duration": 40,"vehicles_count": 50,"lanes_count":3},)

env.action_space.seed(0)


class SafetyFirstBaselinePolicy:
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

    LANE_THRESHOLD = 0.25
    CAUTION_FRONT = 0.30 #ahead car too close: change lane or slow down
    SAFE_FRONT = 0.30 #minimum required free space in front in a target lane for a safe lane change 
    SAFE_REAR = 0.10 #minimum required free space behind in a target lane for a safe lane change 

    def __init__(self, env):
        self.env = env

    def _available_actions(self):
        try:
            #returns the set of actions that are available (e.g. if the ego vehicle is in the leftmost lane, it cannot turn left)
            return set(self.env.unwrapped.action_type.get_available_actions())
        except AttributeError:
            #Fallback: assume alla ctions are available
            return set(range(self.env.action_space.n))

    def _format_observation(self, state):
        #converts the state info into a 5 x 5 matrix:
        #row 0 is the ego vehicle
        #rows 1-4 are nearby vehicles
        #each row represents one vehicle: [presence, x, y, vx, vy]
        state = np.asarray(state, dtype=np.float32)

        #if the observation was flattened, reshape back to 5 x 5
        if state.ndim == 1:
            state = state.reshape(-1, 5)

        return state

    def _closest_front_distance(self, vehicles):
        #list of distances of vehicles on the same lane of ego vehicle
        front_distances = []

        for vehicle in vehicles[1:]:
            #for each vehicle, 
            presence, x, y, vx, vy = vehicle

            #if there is at least one vehicle
            if presence < 0.5:
                continue

            #mask out those which are not in the same lane as the ego vehicle
            same_lane = abs(y) <= self.LANE_THRESHOLD

            #and if they are in the same lane and in front of it (x > 0), append their distance to the list
            if same_lane and x > 0:
                front_distances.append(x)

        if len(front_distances) == 0:
            return None

        #return the distance of the closest vehicle in the same lane
        return min(front_distances)

    def _is_lane_safe(self, vehicles, target_lane):
        
        #checks whether a target lane ("right" or "left") is safe
        for vehicle in vehicles[1:]:
            presence, x, y, vx, vy = vehicle

            if presence < 0.5: #no vehicles
                continue

            #approximates lane membership using lateral position y
            if target_lane == "left":
                in_target_lane = y < -self.LANE_THRESHOLD
            elif target_lane == "right":
                in_target_lane = y > self.LANE_THRESHOLD
            else:
                raise ValueError("target_lane must be either 'left' or 'right'.")

            #ignores vehicles that are not in the target lane
            if not in_target_lane:
                continue

            #a target lane is unsafe if there is a vehicle:
            #- too close in front: 0 < x < SAFE_FRONT
            #- too close behind: - SAFE_READ < x <0
            too_close_front = 0 < x < self.SAFE_FRONT
            too_close_rear = -self.SAFE_REAR < x < 0

            if too_close_front or too_close_rear:
                return False

        return True

    def select_action(self, state):
        #select one action accordingly to the policy
        
        '''
        1. If a vehicle is too close in front:
           a. try changing to the right lane if available and safe (higher rewards on the right lane)
           b. otherwise try changing to the left lane if available and safe
           c. otherwise slow down
        2. if the road ahead is clear:
           a. accelerate if possible
           b. otherwise stay idle
        '''
        vehicles = self._format_observation(state)
        available_actions = self._available_actions()

        #detect the closest front vehicle in the current lane
        front_distance = self._closest_front_distance(vehicles)

        front_vehicle_too_close = (front_distance is not None and front_distance < self.CAUTION_FRONT)

        #if the front vehicle is too close
        if front_vehicle_too_close:
            #check whether lane-change actions are allowed
            right_is_available = self.LANE_RIGHT in available_actions
            left_is_available = self.LANE_LEFT in available_actions

            #check whether adjacent lanes are safe
            right_is_safe = self._is_lane_safe(vehicles, target_lane="right")
            left_is_safe = self._is_lane_safe(vehicles, target_lane="left")

            #prefer changing to the right if possible
            if right_is_available and right_is_safe:
                return self.LANE_RIGHT
            #otherwise change to the left
            elif left_is_available and left_is_safe:
                return self.LANE_LEFT
            #if there are no safe lanes, slow down
            elif self.SLOWER in available_actions:
                return self.SLOWER
            #fallback
            else:
                return self.IDLE

        #if there is no close front vehicle, accelerate
        if self.FASTER in available_actions:
            return self.FASTER

        #fallback
        return self.IDLE


#initialize the agent
agent = SafetyFirstBaselinePolicy(env)

state, _ = env.reset(seed=0)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0.0

episode_results = []

while episode <= 10:
    episode_steps += 1

    action = agent.select_action(state)

    next_state, reward, done, truncated, info = env.step(action)

    state = next_state
    episode_return += reward

    if done or truncated:
        print(
            f"Total T: {episode_steps} Episode Num: {episode} "
            f"Episode T: {episode_steps} Return: {episode_return:.3f} "
            f"Crash: {done}"
        )

        episode_results.append(
            {
                "episode": episode,
                "episode_steps": episode_steps,
                "episode_return": float(episode_return),
                "done": bool(done),
                "truncated": bool(truncated),
                "crash": bool(done),
            }
        )

        state, _ = env.reset()
        done, truncated = False, False

        episode += 1
        episode_steps = 0
        episode_return = 0.0

env.close()

returns = [ep["episode_return"] for ep in episode_results]
steps = [ep["episode_steps"] for ep in episode_results]
crashes = [ep["crash"] for ep in episode_results]

results = {
    "env_name": env_name,
    "seed": 0,
    "policy": "safety_first_baseline",
    "config": {
        "action": {"type": "DiscreteMetaAction"},
        "duration": 40,
        "vehicles_count": 50,
    },
    "thresholds": {
        "LANE_THRESHOLD": agent.LANE_THRESHOLD,
        "CAUTION_FRONT": agent.CAUTION_FRONT,
        "SAFE_FRONT": agent.SAFE_FRONT,
        "SAFE_REAR": agent.SAFE_REAR,
    },
    "num_episodes": len(episode_results),
    "mean_return": float(np.mean(returns)) if returns else None,
    "std_return": float(np.std(returns)) if returns else None,
    "mean_episode_steps": float(np.mean(steps)) if steps else None,
    "crash_rate": float(np.mean(crashes)) if crashes else None,
    "episodes": episode_results,
}

output_path = Path("baseline_results.json")
with output_path.open("w") as f:
    json.dump(results, f, indent=4)

print(f"Saved results to {output_path}")