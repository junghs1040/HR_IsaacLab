# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import threading
import time
from collections import deque

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard-control", action="store_true", default=False, help="Enable keyboard control for velocity commands.")
# Add custom velocity command arguments
parser.add_argument("--vel_x", type=float, default=None, help="Custom linear velocity in x direction (m/s)")
parser.add_argument("--vel_y", type=float, default=None, help="Custom linear velocity in y direction (m/s)")
parser.add_argument("--vel_yaw", type=float, default=None, help="Custom angular velocity around z axis (rad/s)")
parser.add_argument("--heading", type=float, default=None, help="Custom heading angle (rad)")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


class KeyboardController:
    """Keyboard controller for velocity commands."""
    
    def __init__(self):
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_yaw = 0.0
        
        # Velocity limits
        self.max_lin_vel = 2.0  # m/s
        self.max_ang_vel = 2.0  # rad/s
        self.vel_step = 0.1     # m/s per key press
        self.ang_step = 0.1     # rad/s per key press
        
        # Key states
        self.keys_pressed = set()
        self.running = True
        
        # Start keyboard listener thread
        self.listener_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.listener_thread.start()
        
        print("[INFO] Keyboard control enabled!")
        print("[INFO] Controls:")
        print("  - W/S: Forward/Backward (X velocity)")
        print("  - A/D: Left/Right (Y velocity)")
        print("  - Q/E: Turn Left/Right (Yaw velocity)")
        print("  - Space: Stop all movement")
        print("  - ESC: Exit")
        print(f"  - Max linear velocity: {self.max_lin_vel} m/s")
        print(f"  - Max angular velocity: {self.max_ang_vel} rad/s")
        print(f"[INFO] Initial velocity: X: {self.vel_x:6.2f} m/s, Y: {self.vel_y:6.2f} m/s, Yaw: {self.vel_yaw:6.2f} rad/s")
    
    def _keyboard_listener(self):
        """Listen for keyboard input in a separate thread."""
        try:
            import msvcrt  # Windows
            while self.running:
                if msvcrt.kbhit():  # type: ignore
                    try:
                        # Get the raw byte
                        raw_key = msvcrt.getch()  # type: ignore
                        
                        # Handle special keys first
                        if raw_key == b'\x00':  # Extended key
                            extended_key = msvcrt.getch()  # type: ignore
                            if extended_key == b'H':  # Up arrow
                                self._handle_key('w')
                            elif extended_key == b'P':  # Down arrow
                                self._handle_key('s')
                            elif extended_key == b'K':  # Left arrow
                                self._handle_key('a')
                            elif extended_key == b'M':  # Right arrow
                                self._handle_key('d')
                            continue
                        
                        # Try to decode as UTF-8, fallback to other encodings
                        try:
                            key = raw_key.decode('utf-8').lower()
                        except UnicodeDecodeError:
                            try:
                                # Try Windows-1252 (common Windows encoding)
                                key = raw_key.decode('cp1252').lower()
                            except UnicodeDecodeError:
                                try:
                                    # Try ASCII
                                    key = raw_key.decode('ascii', errors='ignore').lower()
                                except:
                                    # Skip if we can't decode
                                    continue
                        
                        self._handle_key(key)
                    except Exception as e:
                        # Skip problematic keys
                        continue
                time.sleep(0.01)
        except ImportError:
            try:
                import sys
                import tty
                import termios
                
                # Unix/Linux/Mac
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(sys.stdin.fileno())
                    while self.running:
                        if sys.stdin.readable():
                            key = sys.stdin.read(1).lower()
                            self._handle_key(key)
                        time.sleep(0.01)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except ImportError:
                print("[WARNING] Keyboard control not supported on this platform")
                self.running = False
    
    def _handle_key(self, key):
        """Handle individual key presses."""
        old_vel_x, old_vel_y, old_vel_yaw = self.vel_x, self.vel_y, self.vel_yaw
        
        if key == 'w':
            self.vel_x = min(self.vel_x + self.vel_step, self.max_lin_vel)
        elif key == 's':
            self.vel_x = max(self.vel_x - self.vel_step, -self.max_lin_vel)
        elif key == 'a':
            self.vel_y = min(self.vel_y + self.vel_step, self.max_lin_vel)
        elif key == 'd':
            self.vel_y = max(self.vel_y - self.vel_step, -self.max_lin_vel)
        elif key == 'q':
            self.vel_yaw = min(self.vel_yaw + self.ang_step, self.max_ang_vel)
        elif key == 'e':
            self.vel_yaw = max(self.vel_yaw - self.ang_step, -self.max_ang_vel)
        elif key == ' ':
            # Space bar - stop all movement
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.vel_yaw = 0.0
        elif key == '\x1b':  # ESC key
            self.running = False
        
        # Print current velocity only if it changed
        if (old_vel_x != self.vel_x or old_vel_y != self.vel_y or old_vel_yaw != self.vel_yaw):
            print(f"\r[INFO] Velocity changed - X: {self.vel_x:6.2f} m/s, Y: {self.vel_y:6.2f} m/s, Yaw: {self.vel_yaw:6.2f} rad/s")
    
    def get_velocity(self):
        """Get current velocity command."""
        return self.vel_x, self.vel_y, self.vel_yaw
    
    def stop(self):
        """Stop the keyboard controller."""
        self.running = False


class CustomVelocityCommandWrapper(VecEnv):
    """Wrapper to override velocity commands with custom values."""
    
    def __init__(self, env, vel_x=None, vel_y=None, vel_yaw=None, heading=None, keyboard_controller=None):
        super().__init__()
        self.env = env
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_yaw = vel_yaw
        self.heading = heading
        self.keyboard_controller = keyboard_controller
        
        # Check if we have custom velocity commands
        self.has_custom_commands = any(v is not None for v in [vel_x, vel_y, vel_yaw, heading]) or keyboard_controller is not None
        
        if self.has_custom_commands:
            if keyboard_controller:
                print(f"[INFO] Using keyboard-controlled velocity commands")
            else:
                print(f"[INFO] Using custom velocity commands:")
                if vel_x is not None:
                    print(f"  - Linear velocity X: {vel_x} m/s")
                if vel_y is not None:
                    print(f"  - Linear velocity Y: {vel_y} m/s")
                if vel_yaw is not None:
                    print(f"  - Angular velocity Yaw: {vel_yaw} rad/s")
                if heading is not None:
                    print(f"  - Heading: {heading} rad")
    
    def _override_velocity_commands(self):
        """No-op: command_manager override disabled for this task."""
        return
    
    @property
    def num_envs(self):
        """Return the number of environments."""
        return self.env.num_envs
    
    @property
    def device(self):
        """Return the device."""
        return self.env.device
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env.unwrapped
    
    def get_observations(self):
        """Get observations and override velocity commands if custom ones are set."""
        # Get observations
        obs_tensor, info = self.env.get_observations()
        
        if self.has_custom_commands:
            # The observation structure is: [base_lin_vel, base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions, height_scan]
            # For H1 humanoid, velocity_commands should be 3 values (x, y, yaw)
            obs_dim = obs_tensor.shape[1]
            
            # Debug: print observation shape to understand structure

            print(f"[DEBUG] Observation shape: {obs_tensor.shape}")
            print(f"[DEBUG] First observation: {obs_tensor[0]}")
            print(f"[DEBUG] Observation dimension: {obs_dim}")
            
            # Try to identify velocity_commands by looking for patterns
            # Look for the first few values that might be velocity_commands
            print(f"[DEBUG] First 20 values: {obs_tensor[0, :20]}")
            print(f"[DEBUG] Values around index 9-11: {obs_tensor[0, 9:12]}")
            print(f"[DEBUG] Values around index 3-6: {obs_tensor[0, 3:6]}")
            
            # Check if there are any zero values that might indicate velocity_commands
            zero_indices = torch.where(obs_tensor[0] == 0.0)[0]
            print(f"[DEBUG] Indices with zero values: {zero_indices}")
                
            
            # Common observation structures:
            if obs_dim >= 48:  # Likely has velocity_commands
                vel_cmd_start_idx = 9  # After base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3)
                vel_cmd_end_idx = 12   # 3 values: x, y, yaw
            else:
                # Fallback to original assumption
                vel_cmd_start_idx = 3
                vel_cmd_end_idx = 6
            
            # Ensure indices are within bounds
            if vel_cmd_end_idx <= obs_dim:
                # Create custom velocity command
                custom_vel_cmd = torch.zeros_like(obs_tensor[:, vel_cmd_start_idx:vel_cmd_end_idx])
                
                if self.keyboard_controller:
                    # Use keyboard controller values
                    vel_x, vel_y, vel_yaw = self.keyboard_controller.get_velocity()
                    custom_vel_cmd[:, 0] = vel_x
                    custom_vel_cmd[:, 1] = vel_y
                    custom_vel_cmd[:, 2] = vel_yaw
                    
                else:
                    # Use static custom values
                    if self.vel_x is not None:
                        custom_vel_cmd[:, 0] = self.vel_x
                    if self.vel_y is not None:
                        custom_vel_cmd[:, 1] = self.vel_y
                    if self.vel_yaw is not None:
                        custom_vel_cmd[:, 2] = self.vel_yaw
                
                # Replace the velocity commands in the observation
                obs_tensor = obs_tensor.clone()
                obs_tensor[:, vel_cmd_start_idx:vel_cmd_end_idx] = custom_vel_cmd
                # Print custom velocity command (first env) for debugging
                try:
                    print(f"[CUSTOM_VEL_CMD] X: {custom_vel_cmd[0,0]:6.2f} | Y: {custom_vel_cmd[0,1]:6.2f} | Yaw: {custom_vel_cmd[0,2]:6.2f}")
                except Exception:
                    pass
                
                # Real-time observation output - focus on velocity commands
                if hasattr(self, '_debug_printed') and self._debug_printed:
                    # Print velocity commands in real-time (main focus)
                    print(f"\r[VEL] X: {obs_tensor[0, vel_cmd_start_idx]:6.2f} | Y: {obs_tensor[0, vel_cmd_start_idx+1]:6.2f} | Yaw: {obs_tensor[0, vel_cmd_start_idx+2]:6.2f}", end="")
        
        return obs_tensor, info
    
    def reset(self, **kwargs):
        """Reset environment and ensure robot starts in stopped state."""
        obs_tensor, info = self.env.reset(**kwargs)
        
        # Force initial velocity commands to zero to ensure robot starts stopped
        if self.has_custom_commands:
            obs_dim = obs_tensor.shape[1]
            
            if obs_dim >= 48:
                vel_cmd_start_idx = 9
                vel_cmd_end_idx = 12
            else:
                vel_cmd_start_idx = 3
                vel_cmd_end_idx = 6
            
            if vel_cmd_end_idx <= obs_dim:
                # Force initial velocity to zero
                obs_tensor = obs_tensor.clone()
                obs_tensor[:, vel_cmd_start_idx:vel_cmd_end_idx] = 0.0
                print(f"[INFO] Reset: Forcing initial velocity commands to zero")
        
        return obs_tensor, info
    
    def step(self, actions):
        """Forward step call to environment and override velocity commands."""
        # Step environment
        obs_tensor, reward, done, info = self.env.step(actions)
        
        # Override velocity commands in returned observation
        if self.has_custom_commands:
            obs_dim = obs_tensor.shape[1]
            if obs_dim >= 48:
                vel_cmd_start_idx = 9
                vel_cmd_end_idx = 12
            else:
                vel_cmd_start_idx = 3
                vel_cmd_end_idx = 6
            if vel_cmd_end_idx <= obs_dim:
                if self.keyboard_controller:
                    vel_x, vel_y, vel_yaw = self.keyboard_controller.get_velocity()
                else:
                    vel_x = self.vel_x if self.vel_x is not None else 0.0
                    vel_y = self.vel_y if self.vel_y is not None else 0.0
                    vel_yaw = self.vel_yaw if self.vel_yaw is not None else 0.0
                obs_tensor = obs_tensor.clone()
                obs_tensor[:, vel_cmd_start_idx:vel_cmd_end_idx] = torch.tensor([vel_x, vel_y, vel_yaw], device=obs_tensor.device)
        
        return obs_tensor, reward, done, info
    
    def close(self):
        """Forward close call to environment."""
        return self.env.close()
    
    def __getattr__(self, name):
        """Forward all other attributes to the wrapped environment."""
        return getattr(self.env, name)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Initialize keyboard controller if enabled
    keyboard_controller = None
    if args_cli.keyboard_control:
        keyboard_controller = KeyboardController()

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # wrap for custom velocity commands AFTER RslRlVecEnvWrapper
    if any(v is not None for v in [args_cli.vel_x, args_cli.vel_y, args_cli.vel_yaw, args_cli.heading]) or keyboard_controller:
        env = CustomVelocityCommandWrapper(
            env, 
            vel_x=args_cli.vel_x, 
            vel_y=args_cli.vel_y, 
            vel_yaw=args_cli.vel_yaw, 
            heading=args_cli.heading,
            keyboard_controller=keyboard_controller
        )

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running() and (not keyboard_controller or keyboard_controller.running):
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # stop keyboard controller if it was running
    if keyboard_controller:
        keyboard_controller.stop()
        print("\n[INFO] Keyboard controller stopped.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
