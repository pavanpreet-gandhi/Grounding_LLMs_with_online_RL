import sys
import gym
import warnings
import babyai_text
from transformers import pipeline

warnings.filterwarnings("ignore")

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


def parse_observation(info):
    """
    Prints the textual descriptions stored in info['descriptions'],
    exactly like in human_play.py.
    """
    print("Observation:")
    if "descriptions" in info and info["descriptions"]:
        for description in info["descriptions"]:
            print(description)
    else:
        print("(No textual descriptions found in info['descriptions'])")
    print()


def llm_choose_action(generator, mission, descriptions):
    """
    Uses a small LLM (distilgpt2) to decide which action to take next,
    given the mission and current textual descriptions.

    Returns an integer action ID corresponding to the chosen action.
    """

    # Build a short prompt:
    possible_actions = ", ".join(
        text_to_action.keys()
    )  # e.g., "turn left, turn right, go forward..."
    obs_text = "\n".join(descriptions) if descriptions else "No descriptions"

    prompt = (
        f"You are an agent in a simple navigation game. Your goal is to: {mission}.\n\n"
        f"Current observation:\n{obs_text}\n\n"
        f"You can choose ONE of these actions: {', '.join(text_to_action.keys())}.\n"
        "Which action will you take next? Respond with exactly one action from the list."
    )

    # Generate a short response
    response = generator(
        prompt,
        max_new_tokens=20,  # limit the generation length
        do_sample=True,
        top_p=0.9,
        truncation=True,
    )[0]["generated_text"].lower()

    # Attempt to find a valid action in the LLM output
    for action_str, action_id in text_to_action.items():
        if action_str in response:
            return action_id

    # If nothing matches, default to "done" (or any fallback you want)
    print("No valid action found in LLM response. Defaulting to 'done'.")
    return 6  # 'done'


def run_episode(env, generator):
    """
    Runs one BabyAI episode using the LLM to select actions.
    Prints the mission, textual observations, chosen actions, etc.
    """

    # Reset environment
    obs, info = env.reset()
    done = False
    step = 0

    # Print the mission (goal)
    mission = obs.get("mission", "No mission provided")
    print("Mission:")
    print(mission)
    print()

    while not done:
        step += 1

        # Print textual observation
        parse_observation(info)

        # Decide on an action using the LLM
        descriptions = info.get("descriptions", [])
        action_id = llm_choose_action(generator, mission, descriptions)
        action_str = action_to_text.get(action_id, f"Unknown({action_id})")

        # Take the step in the environment
        obs, reward, done, info = env.step(action_id)

        # Log
        print(
            f"Step {step}: Action = {action_str} (ID={action_id}), Reward={reward}, Done={done}"
        )
        print()

    print(f"Episode finished in {step} steps.\n")


def main():
    print("Loading small LLM from Hugging Face (distilgpt2)...")
    generator = pipeline("text-generation", model="distilgpt2")
    print("LLM loaded.\n")

    # Create your BabyAI environment
    env = gym.make("BabyAI-MixedTrainLocal-v0")

    # Run a couple of episodes with LLM-driven actions
    num_episodes = 2
    for i in range(num_episodes):
        print(f"=== Starting Episode {i+1} ===")
        run_episode(env, generator)
        print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
