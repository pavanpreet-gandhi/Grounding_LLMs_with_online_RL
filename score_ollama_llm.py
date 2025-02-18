import warnings
warnings.filterwarnings("ignore")

import gym
import babyai_text
import ollama
from tqdm.auto import tqdm  # Uses appropriate version for notebook/terminal


# Maps BabyAI actions to text:
action_to_text = {
    0: "turn left",
    1: "turn right",
    2: "go forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}
text_to_action = {v: k for k, v in action_to_text.items()}


# Helper functions for prompting
def get_instructions(goal: str):
    return f"""
You are an agent playing a simple navigation game. Your goal is to **{goal}**. The following are the possible actions you can take in the game, followed by a short description of each action:

turn left: turn to the left,
turn right: turn to the right,
go forward: take one step forward,
pick up: pick up the object below you,
drop: drop the object that you are holding,
toggle: manipulate the object in front of you.

In a moment I will present you an observation.

Tips:
- Once the desired object you want to interact or pickup in front of you, you can use the 'toggle' action to interact with it.
- It doesn't make sense to repeat the same action over and over if the observation doesn't change.

PLAY!"""

def parse_observation(obs, info) -> str:
    """
    Returns a string with the textual descriptions stored in info['descriptions']
    """
    out = ""
    for description in info["descriptions"]:
        out += description + "\n"
    return out


def parse_action(action: str) -> int:
    """
    Returns the action ID corresponding to the given action text, or None if the action is invalid.
    """
    return text_to_action.get(action, None)

invalid_action_message = "Invalid action, the valid actions are: " + ", ".join(action_to_text.values()) + ".\n"
invalid_action_message += "Please output one of the above actions and nothing else."


def score_ollama_llm(model_name="llama3.2", env_name="BabyAI-GoToObj-v0", num_episodes=10, max_invalid_per_step=5, verbose=False):
    
    env = gym.make("BabyAI-GoToObj-v0")
    env.seed(0) # for consistancy
    rewards = []
    invalid_actions_per_episode = []
    
    for episode in tqdm(range(num_episodes), desc="Episodes", unit="episode"):
        
        obs, info = env.reset()
        goal = obs["mission"]
        done = False
        instructions = get_instructions(goal)
        obs_text = parse_observation(obs, info)
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": obs_text},
        ]
        invalid_actions = 0  # Counter for invalid actions in this episode
        if verbose:
            print(instructions)
            print(obs_text)

        while not done:
            
            # get action from LLM
            action_text = None
            invalid_actions_this_step = 0  # Counter for invalid actions in current step
            
            while action_text not in text_to_action:
                if action_text is not None:
                    invalid_actions += 1
                    invalid_actions_this_step += 1
                    if invalid_actions_this_step >= max_invalid_per_step:
                        if verbose:
                            print(f"Exceeded maximum invalid actions ({max_invalid_per_step}) in one step. Ending episode.")
                        done = True
                        reward = 0
                        break
                    messages.append({"role": "user", "content": invalid_action_message})
                    if verbose:
                        print(invalid_action_message)
                response = ollama.chat(model_name, messages)
                action_text = response.message.content.lower().strip()
                messages.append({"role": "assistant", "content": action_text})
                if verbose:
                    print(action_text)
            
            if done:  # Early exit due to too many invalid actions
                rewards.append(reward)
                invalid_actions_per_episode.append(invalid_actions)
                break

            # apply action
            action = text_to_action[action_text]
            obs, reward, done, info = env.step(action)
            obs_text = parse_observation(obs, info)
            messages.append({"role": "user", "content": obs_text})
            if verbose:
                print(obs_text)
            
            if done:
                if verbose:
                    print("Done!")
                    print(f"Reward: {reward}")
                    print(f"Invalid actions this episode: {invalid_actions}")
                rewards.append(reward)
                invalid_actions_per_episode.append(invalid_actions)
                break
    
    return rewards, invalid_actions_per_episode


if __name__ == "__main__":
    rewards, invalid_actions = score_ollama_llm(num_episodes=2, max_invalid_per_step=5, verbose=False)
    print("Rewards:", rewards)
    print("Invalid actions per episode:", invalid_actions)