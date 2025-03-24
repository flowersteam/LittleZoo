# 🌿 LittleZoo 🦁

**A natural language Gym environment for goal-conditioned reinforcement learning**

LittleZoo is a Gym environment built on top of the [Playground environment](https://github.com/flowersteam/playground_env). It enables agents to interact through natural language, where actions are given as text commands, and observations are returned as textual scene descriptions. This makes it an ideal testbed for LLM agents.

---

## 🚀 Installation

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

---

## 🌍 Environment Overview

### 🔍 Observations

The environment provides observations in a structured textual format, along with additional information:

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

### 🎮 Actions

Agents interact by passing a string to the `step()` function. Valid actions include:

- **`"Go to <obj_i>"`** → Move to the object `<obj_i>`.
- **`"Grasp"`** → Pick up an object if inventory space allows.
- **`"Release <obj_i>"`** → Drop a specific object.
- **`"Release both"`** → Drop all held objects.

### 🎯 Goals

LittleZoo offers four categories of hierarchical goals, increasing in complexity:

1. **Grasp {obj}** → Navigate and grasp an object.
2. **Grow {plant}** → Provide water to grow a plant.
3. **Grow {herbivore}** → Grow a plant, then feed it to a herbivore.
4. **Grow {carnivore}** → Grow a herbivore, then feed it to a carnivore.

- **Reward Structure:**
  - Achieving a goal grants **+1 reward** and ends the episode.
  - Otherwise, the reward is **0**.
  
- **Step Limits:**
  - **Grasp:** 3 steps
  - **Grow Plant:** 6 steps
  - **Grow Herbivore:** 10 steps
  - **Grow Carnivore:** 15 steps
  - **Impossible goals (e.g., "Grow table")**: 10 steps

---

## 📦 Object Categories

### 🏠 **Furniture**
Door, Chair, Desk, Lamp, Table, Cupboard, Sofa, Bookshelf, Bed

### 🌱 **Plants**
Carrot, Potato, Berry, Lettuce, Tomato, Cucumber, Spinach, Broccoli, Onion

### 🐮 **Herbivores**
Cow, Elephant, Rabbit, Deer, Sheep, Giraffe, Goat, Horse, Bison

### 🦁 **Carnivores**
Lion, Tiger, Bobcat, Panther, Coyote, Wolf, Leopard, Hyena, Jackal

### 💧 **Supplies**
Water

---

## 📖 Citation

If you use LittleZoo in your research, please cite:

```bibtex
@article{gaven2025magellan,
  title={MAGELLAN: Metacognitive predictions of learning progress guide autotelic LLM agents in large goal spaces},
  author={Gaven, Loris and Carta, Thomas and Romac, Cl{\'e}ment and Colas, C{\'e}dric and Lamprier, Sylvain and Sigaud, Olivier and Oudeyer, Pierre-Yves},
  journal={arXiv preprint arXiv:2502.07709},
  year={2025}
}
```

---

## 💡 Contribute

If you’d like to contribute to LittleZoo, feel free to open an issue or submit a pull request on [GitHub](https://github.com/LorisGaven/LittleZoo). 🚀
