
import gymnasium as gym
import gymnasium_robotics
import sys

# Register envs
gym.register_envs(gymnasium_robotics)

env_id = "AdroitHandPen-v1"
try:
    env = gym.make(env_id, render_mode="rgb_array") # Use rgb_array to avoid opening window but still init renderer elements if possible
    print(f"Env: {env}")
    print(f"Unwrapped: {env.unwrapped}")
    
    if hasattr(env.unwrapped, "mujoco_renderer"):
        print("Has mujoco_renderer")
        print(f"Renderer type: {type(env.unwrapped.mujoco_renderer)}")
        # In newer gymnasium, the viewer might be instantiated only on render
        # Let's try to get a frame to trigger init
        env.reset()
        env.render()
        if hasattr(env.unwrapped.mujoco_renderer, "viewer"):
             print(f"Renderer has viewer: {env.unwrapped.mujoco_renderer.viewer}")
             print(f"Viewer attributes: {dir(env.unwrapped.mujoco_renderer.viewer)}")
        else:
             print("Renderer has no viewer attribute yet (might be None)")
             
    if hasattr(env.unwrapped, "viewer"):
        print(f"Unwrapped has viewer: {env.unwrapped.viewer}")
        
except Exception as e:
    print(f"Error: {e}")
