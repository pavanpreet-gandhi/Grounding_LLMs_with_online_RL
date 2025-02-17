import sys
import gym
import babyai_text

import warnings
warnings.filterwarnings('ignore')

######################
## Helper Functions ##
######################

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

action_to_text = {
    0: 'turn left',
    1: 'turn right',
    2: 'go forward',
    3: 'pick up',
    4: 'drop',
    5: 'toggle',
    6: 'done',
}

text_to_action = {v: k for k, v in action_to_text.items()}

def parse_action():
    print("Action:")
    is_valid = False
    while not is_valid:
        action = input('Enter action: ')
        if action == 'exit':
            sys.exit()
        if action in text_to_action:
            is_valid = True
        else:
            print(f"{action} is not a valid action, please try again.")
    print(action)
    print()
    return action

def parse_observation(info):
    print("Observation: ")
    for description in info['descriptions']:
        print(description)
    print()

##########
## Play ##
##########

env = gym.make("BabyAI-MixedTrainLocal-v0")
obs, info = env.reset()
done = False
goal = obs['mission']

print("Instructions:")
instructions = get_instructions(goal)
print(instructions)
print()

while not done:
    
    parse_observation(info)
    
    action = parse_action()
    
    obs, reward, done, info = env.step(text_to_action[action])
    print(f"Reward: {reward} | Done: {done}")
    print()