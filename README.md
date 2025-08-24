# HR_IsaacLab
    
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![RSL_RK](https://img.shields.io/badge/RSL_RL-2.3.1-silver)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

  
#### Humanoid Robot Learning Environment (Manager Based Env)  
     
configuration script : [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/HRLAB_rough_env_cfg.py]
(https://github.com/junghs1040/HR_IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/HRLAB_rough_env_cfg.py)

## Installation  
  
## How to Train  

You can train the policy using this code  
  
<pre>
<code>     
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-H1-v0     
</code>
</pre>
  
<br>  
  
<img width="1435" height="928" alt="learning" src="https://github.com/user-attachments/assets/b014aa28-3591-4093-ad0f-17e9bc50cb84" />

## How to Play  
   
You can play the policy using this code  

<pre>
<code>     
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-H1-v0 --checkpoint logs/rsl_rl/h1_rough/2025-07-20_21-19-44\model_2999.pt --num_envs 256     
</code>
</pre>       
  
<img width="1435" height="852" alt="inference" src="https://github.com/user-attachments/assets/361f735b-1617-4942-bd14-03fbac89403f" />   
  
## View Training Logs with TensorBoard  
    
You can use TensorBoard to monitor the current training progress in real time.  
   
Linux :    
<pre>
<code>   
./isaaclab.sh -p -m tensorboard.main --logdir=logs
</code>
</pre>
  
window :
<pre>
<code>   
isaaclab.bat -p -m tensorboard.main --logdir=logs
</code>
</pre>   
   
<img width="1890" height="928" alt="tensorboard" src="https://github.com/user-attachments/assets/62f22f9f-a57e-4a3f-9518-24b7320f3609" />  
  