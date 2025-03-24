# LittleZoo 🦁  

LittleZoo is a Gym environment built on top of the [Playground environment](https://github.com/flowersteam/playground_env). It features natural language interactions, where actions are given as text commands, and observations are returned as textual scene descriptions.  

## Installation  

First, create a Conda environment:  

```sh
conda create -n littlezoo python=3.10
conda activate littlezoo
```  

Then, install the dependencies:  

```sh
git clone https://github.com/LorisGaven/LittleZoo.git
cd LittleZoo/little_zoo
pip install -e .
```  

## Environment Description  

### Observations  

The environment provides observations and additional information in the following formats:  

- **Observation** (str):  
  ```text
  You see: <obj_1>, <obj_2>, <obj_3>, <obj_4>
  You are standing on: <obj_i> | "nothing"
  Inventory (n/2): <obj_i>, <obj_j> | "empty"
  ```  

- **Info** (dict):  
  ```python
  {
      'goal': <goal>,
      'possible_actions': [
          'Grasp', 'Release <obj_1>', 'Release <obj_2>', 'Release both',
          'Go to <obj_1>', 'Go to <obj_2>', ..., 'Go to <obj_n>'
      ]
  }
  ```  

### Actions  

Agents interact with the environment by passing a string representing the desired action to the `step()` function. Valid actions include:  

- **`"Go to <obj_i>"`**: Move to the location of object `<obj_i>`.  
- **`"Grasp"`**: Pick up the object the agent is standing on, if the inventory is not full.  
- **`"Release <obj_i>"`**: Drop the specified object from the inventory.  
- **`"Release both"`**: Drop both objects in the inventory.  

### Goals  

There are four categories of goals, each with increasing difficulty:  

1. **Grasp {obj}**: Navigate to the object and grasp it.  
2. **Grow {plant}**: Grow a plant by providing it with water.  
3. **Grow {herbivore}**: First, grow a plant, then feed it to a herbivore to help it grow.  
4. **Grow {carnivore}**: Grow a herbivore, then feed it to a carnivore.  

### Object Categories  

#### **Furniture**  
- Door, Chair, Desk, Lamp, Table, Cupboard, Sofa, Bookshelf, Bed  

#### **Plants**  
- Carrot, Potato, Berry, Lettuce, Tomato, Cucumber, Spinach, Broccoli, Onion  

#### **Herbivores**  
- Cow, Elephant, Rabbit, Deer, Sheep, Giraffe, Goat, Horse, Bison  

#### **Carnivores**  
- Lion, Tiger, Bobcat, Panther, Coyote, Wolf, Leopard, Hyena, Jackal  

#### **Supplies**  
- Water  
