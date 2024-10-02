import gymnasium as gym
import numpy as np
import random
import re

from little_zoo.playground.reward_function import get_reward_from_state, sample_descriptions_from_state
from little_zoo.playground.descriptions import generate_all_descriptions
from little_zoo.playground.env_params import get_env_params

class LittleZoo(gym.Env):
    '''
        PlayGroundText is a wrapper for the PlayGroundNavigation environment.
        It convert natural language commands into actions for the PlayGroud environment
        and convert observations into natural language descriptions.
    '''
    def __init__(self,
                 nb_objects=8,
                 train=True,
                 seed=None,
                 remove_test_from_obj=True,
                 impossible_goals=False
                 ):
        
        # The playground environment
        self.playground = gym.make('PlaygroundNavigation-v1', max_nb_objects=nb_objects)
        
        # Dict containing all the playground environment parameters
        self.env_params = get_env_params()
        
        # Ensure that the objects from the test are removed from the scene
        self.playground.unwrapped.remove_test = remove_test_from_obj
                        
        # Generate all the descriptions/goals for the environment
        train_descriptions, test_descriptions, _ = generate_all_descriptions(self.env_params)
        
        # Remove all the 'Go to <position>' goals and general goals
        general = ['animal', 'thing', 'living_thing', 'carnivore', 'herbivore']
        train_descriptions = [s for s in train_descriptions if not s.startswith('Go') and s.split(' ')[-1] not in general]
        test_descriptions = [s for s in test_descriptions if not s.startswith('Go') and s.split(' ')[-1] not in general]
        
        self.train_descriptions = train_descriptions.copy()
        self.test_descriptions = test_descriptions.copy()
        if impossible_goals:
            for desc in train_descriptions:
                if desc.startswith('Grasp') and desc.split(' ')[-1] in self.env_params['categories']['furniture']:
                    self.train_descriptions.append(desc.replace('Grasp', 'Grow'))
            
            for desc in test_descriptions:
                if desc.startswith('Grasp') and desc.split(' ')[-1] in self.env_params['categories']['furniture']:
                    self.test_descriptions.append(desc.replace('Grasp', 'Grow'))

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
    
    def reset(self, goal_str=None):
        if goal_str is not None:
            self.goal_str = goal_str
        elif self.train:
            self.goal_str = random.choice(self.train_descriptions)
        else:
            self.goal_str = random.choice(self.test_descriptions)
        
        self.playground.reset()
        self.playground.unwrapped.reset_with_goal(self.goal_str)
        o, _, _, _, _ = self.playground.step(np.array([0, 0, 0])) # Init step
        self.current_step = 0
        
        # Define the episode horizon (1.5 x the length of the optimal trajectory)
        if self.goal_str.startswith('Grasp'):
            self.max_steps = 3
        elif self.goal_str.startswith('Grow'):
            obj = self.goal_str.split(' ')[-1]
            if obj in self.env_params['categories']['plant'] or obj == 'plant':
                self.max_steps = 6
            elif obj in self.env_params['categories']['small_herbivore'] or obj == 'small_herbivore':
                self.max_steps = 10
            elif obj in self.env_params['categories']['big_herbivore'] or obj == 'big_herbivore':
                self.max_steps = 18
            elif obj in self.env_params['categories']['small_carnivore'] or obj == 'small_carnivore':
                self.max_steps = 15
            elif obj in self.env_params['categories']['big_carnivore'] or obj == 'big_carnivore':
                self.max_steps = 27
            else: # Impossible grow
                self.max_steps = 15
        
        self.update_obj_info()
        observation, info = self.generate_description()
        
        self.hindsights_list = [] # Used to construct sequential hindsights
        self.hindsights_mem = [] # Used to make sure we don't repeat the same hindsights
        hindsight = self.get_hindsight(o)
        if len(hindsight) != 0:
            self.hindsights_list.extend(hindsight.copy())
            
        info['hindsight'] = hindsight
        
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
        
        # Used for the hindsight method
        grasp = action_str.lower() == 'grasp' and self.playground.unwrapped.gripper_state != 1
        
        # Take a step in the playgroud environment
        o, _, _, _, _ = self.playground.step(action)
        
        # There is a problem if you move directly to one of the objects, the state of the object is not updated
        # So we need to take a step with no action to update the state of the objects
        if action[0] != 0 or action[1] != 0: # If we moved
            o, _, _, _, _ = self.playground.step(np.array([0, 0, self.playground.unwrapped.gripper_state]))
        
        self.current_step += 1
        
        goal_reached = get_reward_from_state(o, self.goal_str, self.env_params)
        
        truncated = self.current_step == self.max_steps
        done = truncated or goal_reached
        
        self.update_obj_info()
        observation, info = self.generate_description()
        
        # Gather hindsights for the current state
        hindsights = self.get_hindsight(o, grasp)
        hindsights = list(set(hindsights) - set(self.hindsights_mem))
            
        info['hindsight'] = hindsights
        
        self.hindsights_mem.extend(hindsights)
        
        self.inventory = info['inventory']
        
        # Reset the size of obj help to find the obj grown in the current step
        self.playground.unwrapped.reset_size()
        
        return observation, float(goal_reached), done, truncated, info
        
    def render(self):
        raise NotImplementedError("Not implemented yet")
    
    
    
    
    # --- Utils ---
    
    def get_hindsight(self, o, grasp=False):
        hindsights = [hindsight for hinsights in sample_descriptions_from_state(o, self.env_params) for hindsight in hinsights]
        return [hindsight for hindsight in hindsights if not hindsight.startswith("Go") and (grasp or not hindsight.startswith("Grasp")) and hindsight in self.train_descriptions]
        
    
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
                key = obj.object_descr['colors'] + ' ' + obj.object_descr['types'] + ' seed'
            elif ('carnivore' in obj.object_descr['categories'] or 'herbivore' in obj.object_descr['categories']) and not obj.grown_once:
                key = 'baby ' + obj.object_descr['colors'] + ' ' + obj.object_descr['types']
            else:
                key = obj.object_descr['colors'] + ' ' + obj.object_descr['types']
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
        if len(obj_held) == 2:
            possible_actions.append('Release all')
        
        info = {'goal': self.goal_str, 'possible_actions': possible_actions, 'inventory': obj_held}
        
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
