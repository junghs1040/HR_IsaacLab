# HR_IsaacLab
    
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RK](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)  
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

  
#### Humanoid Robot Learning Environment (Manager Based Env)  
     

## Installation  
  
## How to Train  

You can train the policy using this code  
  
<pre>
<code>     
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0     
</code>
</pre>
  
## How to Play  

<pre>
<code>     
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-H1-v0 --checkpoint logs/rsl_rl/h1_rough/2025-07-20_21-19-44\model_2999.pt --num_envs 256     
</code>
</pre>       