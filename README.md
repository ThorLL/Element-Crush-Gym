# Element Crush (Gymnasium)
•    A fully functioning match 3 or more tiles game, inspired by Bejeweled and Candy Crush

•    Implemented using **Python** and the **Pygame** library 

•    Features combos where getting a chain of matches after your initial match multiplies your turn score

<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/46468236/65367719-f2960f80-dbf2-11e9-9444-9810f5ecd5cc.gif">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/46468236/65367395-202c8a00-dbee-11e9-9658-8d6ab4859e2a.gif">
</p>

# About
Fork of the [Element Crush](https://github.com/theharrychen/Element-Crush) repo by [theharrychen](https://github.com/theharrychen) ported to a custom Gymnasium environment, with a lot of changes on the side (might update code with better comments and proper documentation)

# Setup
```python
from Match3Env import Match3Env
import numpy as np

env = Match3Env(render_mode='human')
env.metadata['animation_speed'] = 5
observation = env.init()
while True:
    action = np.random.choice(env.n_actions)
    observation, reward, done, won, _ = env.step(action)
    if done:
        if won:
            print('Won game')
        else:
            print('Lost game')
        observation, _ = env.reset()
    env.render()
```

# Images Used: 
https://www.deviantart.com/v-pk/art/Element-symbols-428049065
