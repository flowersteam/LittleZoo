import gymnasium as gym
import numpy as np
import random
import re

from little_zoo.playground.reward_function import get_reward_from_state
from little_zoo.playground.descriptions import generate_all_descriptions
from little_zoo.playground.env_params import get_env_params

class LittleZoo(gym.Env):
    '''
        Little Zoo environment
    '''
    
    def __init__(self,
                 nb_objects=4,
                 train=True,
                 seed=None,
                ):
        
        # The playground environment
        self.playground = gym.make('PlaygroundNavigation-v1', max_nb_objects=nb_objects)
        
        # Dict containing all the playground environment parameters
        self.env_params = get_env_params()
                        
        # Generate all the descriptions/goals for the environment
        train_descriptions, test_descriptions, _ = generate_all_descriptions(self.env_params)
        
        # Remove all the 'Go to <position>' goals and general goals
        general = ['animal', 'thing', 'living_thing', 'carnivore', 'herbivore']
        train_descriptions = [s for s in train_descriptions if not s.startswith('Go') and s.split(' ')[-1] not in general]
        test_descriptions = [s for s in test_descriptions if not s.startswith('Go') and s.split(' ')[-1] not in general]
        
        # Whether we use the train or test descriptions
        self.train = train
        
        # Observation and action space
        # Here for indication, the observation and action space are much smaller
        self.observation_space = gym.spaces.Text(int(1e6))
        self.action_space = gym.spaces.Text(int(1e6))
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        self.playground.unwrapped.seed(seed)
     
     
        
    # --- Gym methods ---
    
    def reset(self, env_desc=None):
        '''
            Reset the environment with a new goal
            env_desc: A list of strings. The first string is the goal,
            the other strings are the objects in the environment.
        '''
        if env_desc is not None:
            self.env_desc = env_desc
        else:
            raise ValueError('You need to specify a goal')
        
        self.playground.reset()
        self.playground.unwrapped.reset_with_goal(self.env_desc)
        o, _, _, _, _ = self.playground.step(np.array([0, 0, 0])) # Init step
        self.current_step = 0
        
        # Define the episode horizon (1.5 x the length of the optimal trajectory)
        if self.env_desc[0].startswith('Grasp'):
            self.max_steps = 3
        elif self.env_desc[0].startswith('Grow'):
            obj = self.env_desc[0].split(' ')[-1]
            if obj in self.env_params['categories']['plant']:
                self.max_steps = 6
            elif obj in self.env_params['categories']['herbivore']:
                self.max_steps = 10
            elif obj in self.env_params['categories']['carnivore']:
                self.max_steps = 15
            else: # Impossible grow
                self.max_steps = 10
        
        self.update_obj_info()
        observation, info = self.generate_description()
        
        self.inventory = info['inventory']
            
        return observation, info
    
    
    def step(self, action_str):
        if action_str[:5].lower() == 'go to':
            action = self.go_to(action_str[6:])
        elif action_str.lower() == 'grasp':
            action = self.grasp()
        elif action_str[:7].lower() == 'release':
            if 'all' in action_str:
                release_id = 4
            else:
                obj_to_release = action_str[8:]
                if obj_to_release == self.inventory[0]:
                    release_id = 2
                else:
                    release_id = 3
                    
            action = self.release(release_id=release_id)
        else:
            raise ValueError('The action ' + action_str + ' is incorrect')
                
        # Take a step in the playgroud environment
        o, _, _, _, _ = self.playground.step(action)
        
        # There is a problem if you move directly to one of the objects, the state of the object is not updated
        # So we need to take a step with no action to update the state of the objects
        if action[0] != 0 or action[1] != 0: # If we moved
            o, _, _, _, _ = self.playground.step(np.array([0, 0, self.playground.unwrapped.gripper_state]))
        
        self.current_step += 1
        
        goal_reached = get_reward_from_state(o, self.env_desc[0], self.env_params)
        
        truncated = self.current_step == self.max_steps
        done = truncated or goal_reached
        
        self.update_obj_info()
        observation, info = self.generate_description()
        
        self.inventory = info['inventory']
        
        # Reset the size of obj help to find the obj grown in the current step
        self.playground.unwrapped.reset_size()
        
        return observation, float(goal_reached), done, truncated, info
        
    def render(self):
        raise NotImplementedError("Not implemented yet")
    
    
    
    
    # --- Utils ---
    
    def update_obj_info(self):
        '''
            Store in a dict the position and grasped state of all the environment objects
            Ex: {'red cow': {'position': (0.1, 0.2), 'grasped': False}, ...}
        '''
        self.obj_dict = {}
        i = 1
        for obj in self.playground.unwrapped.objects:
            if obj is None: 
                continue
            agent_on = np.linalg.norm(obj.position - self.playground.unwrapped.agent_pos) < (obj.size + obj.agent_size) / 2 and not obj.grasped
            
            if obj.object_descr['categories'] == 'plant' and not obj.grown_once:
                key = obj.object_descr['types'] + ' seed'
            elif ('carnivore' in obj.object_descr['categories'] or 'herbivore' in obj.object_descr['categories']) and not obj.grown_once:
                key = 'baby ' + obj.object_descr['types']
            else:
                key = obj.object_descr['types']
            if key not in self.obj_dict.keys():
                self.obj_dict[key] = {'position': obj.position, 'grasped': obj.grasped, 'agent_on': agent_on, 'grown': obj.grown_once}
            else: # If there are multiple objects with the same description
                self.obj_dict[key + str(i)] = {'position': obj.position, 'grasped': obj.grasped, 'agent_on': agent_on, 'grown': obj.grown_once}
                i += 1
            
    def generate_description(self):
        '''
            Return a natural language description of the scene
        '''
        desc = 'You see: '
        desc += ', '.join(self.rm_trailing_number(obj) for obj in self.obj_dict.keys() if not self.obj_dict[obj]['grasped'])
        agent_on = [self.rm_trailing_number(obj) for obj in self.obj_dict.keys() if self.obj_dict[obj]['agent_on']]
        desc += f'\nYou are standing on: {", ".join(agent_on) if len(agent_on) > 0 else "nothing"}'
        obj_held = [self.rm_trailing_number(obj) for obj in self.obj_dict.keys() if self.obj_dict[obj]['grasped']]
        nb_held = 0
        for obj in self.obj_dict.keys():
            if self.obj_dict[obj]['grasped']:
                nb_held += 1
        desc += f'\nInventory ({nb_held}/2): {", ".join(obj_held) if len(obj_held) > 0 else "empty"}'
        
        possible_actions = ['Grasp'] + ['Go to ' + self.rm_trailing_number(obj) for obj in self.obj_dict.keys() if not self.obj_dict[obj]['grasped']] + ['Release ' + obj for obj in obj_held]
        
        info = {'goal': self.env_desc[0], 'possible_actions': possible_actions, 'inventory': obj_held}
        
        return desc, info
    
    def rm_trailing_number(self, input_str):
        return re.sub(r'\d+$', '', input_str)
    
    
    
    # --- Actions ---
    
    def go_to(self, obj_desc):
        '''
            Return the action to move to the object described by obj_desc
        '''
        for obj in self.obj_dict.keys():
            if obj.startswith(obj_desc) and not self.obj_dict[obj]['grasped']:
                target_pos = self.obj_dict[obj]['position']
                return np.array([target_pos[0] - self.playground.unwrapped.agent_pos[0],
                                target_pos[1] - self.playground.unwrapped.agent_pos[1],
                                -1])

        else:
            raise ValueError(obj_desc + " not in the environment")
    
    def grasp(self):
        '''
            Return the action to grasp an object
        '''
        return np.array([0, 0, 1])
    
    def release(self, release_id = -1):
        '''
            Return the action to release an object
        '''
        return np.array([0, 0, release_id])
