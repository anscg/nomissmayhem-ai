"""
Training script with live visualization callback.
Watch the agent learn in real-time!

Optimized for Apple Silicon (M1/M2/M3/M4).
"""

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from env import NoMissMayhemEnv
from levels import get_level
from visualize import GameRenderer


def get_device(force_gpu: bool = False):
    """Get best available device
    
    Note: CPU is usually faster for MLP policies with parallel envs.
    """
    if force_gpu:
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
    return 'cpu'


class VisualTrainingCallback(BaseCallback):
    """
    Callback that renders training episodes in real-time.
    Shows every Nth episode to avoid slowing training too much.
    """
    
    def __init__(
        self,
        render_freq: int = 10,  # Render every N episodes
        fps: int = 60,
        scale: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.fps = fps
        self.scale = scale
        
        self.renderer = None
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.is_rendering = False
        self.last_obs = None
        self.last_action = None
    
    def _on_training_start(self) -> None:
        """Initialize renderer"""
        self.renderer = GameRenderer(fps=self.fps, scale=self.scale)
        self.episode_count = 0
        self.is_rendering = True  # Start by showing first episode
        print("\n🎮 Visual training mode enabled!")
        print("   SPACE: Pause | S: Step mode | N: Next frame | ESC: Stop visualization")
        print(f"   Showing every {self.render_freq} episodes\n")
    
    def _on_step(self) -> bool:
        """Called after each step"""
        if self.renderer is None:
            return True
        
        # Handle pygame events even when not rendering
        if not self.renderer.handle_events():
            print("\n🛑 Visualization stopped. Training continues in background...")
            self.renderer.close()
            self.renderer = None
            return True
        
        # Get the action that was just taken
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'get_attr'):
            try:
                # For VecEnv, get the actual env
                envs = self.model.env.envs
                if len(envs) > 0:
                    env = envs[0]
                    
                    # Get last action from the buffer
                    self.last_action = self.locals.get('actions', [None])[0]
                    
                    # Track reward
                    rewards = self.locals.get('rewards', [0])
                    self.current_episode_reward += rewards[0] if len(rewards) > 0 else 0
                    self.current_episode_steps += 1
                    
                    # Check for episode end
                    dones = self.locals.get('dones', [False])
                    if dones[0] if len(dones) > 0 else False:
                        self.episode_count += 1
                        
                        if self.verbose > 0:
                            print(f"Episode {self.episode_count}: "
                                  f"Reward={self.current_episode_reward:.2f}, "
                                  f"Steps={self.current_episode_steps}")
                        
                        # Decide if we should render next episode
                        self.is_rendering = (self.episode_count % self.render_freq == 0)
                        
                        # Reset tracking
                        self.current_episode_reward = 0
                        self.current_episode_steps = 0
                    
                    # Render if this is a rendering episode
                    if self.is_rendering and self.renderer:
                        self.renderer.render(
                            env,
                            action=self.last_action,
                            episode=self.episode_count,
                            step=self.current_episode_steps,
                            reward=rewards[0] if len(rewards) > 0 else 0,
                            total_reward=self.current_episode_reward,
                        )
                        
                        # Handle pause
                        while self.renderer.paused and self.renderer.running:
                            if not self.renderer.should_step:
                                self.renderer.handle_events()
                                self.renderer._draw_pause_overlay()
                                import pygame
                                pygame.display.flip()
                                self.renderer.clock.tick(30)
                            else:
                                break
                        
            except Exception as e:
                if self.verbose > 0:
                    print(f"Render error: {e}")
        
        return True
    
    def _on_training_end(self) -> None:
        """Cleanup"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None


def make_env(level_name: str, rank: int, seed: int = 0):
    """Create environment factory for parallel envs"""
    def _init():
        level_data, start_pos = get_level(level_name)
        env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def train_visual(
    level: str = 'simple',
    timesteps: int = 100000,
    render_freq: int = 10,
    fps: int = 60,
    scale: float = 1.0,
    save_path: str = './models/visual_model',
    verbose: int = 1,
    n_envs: int = 1,  # For visual, we need 1 env to render, but train faster with more
    fast_mode: bool = False,  # Skip rendering most of the time
    force_gpu: bool = False,
):
    """
    Train with live visualization.
    
    Args:
        level: Level to train on
        timesteps: Total training timesteps
        render_freq: Show every Nth episode
        fps: Rendering FPS
        scale: Display scale
        save_path: Where to save model
        verbose: Verbosity level
        n_envs: Number of parallel environments (1 for visual mode)
        fast_mode: If True, only render occasionally for speed
        force_gpu: Force GPU/MPS (not recommended for MLP)
    """
    device = get_device(force_gpu)
    print(f"🚀 Starting visual training on level '{level}'")
    print(f"   Device: {device}")
    print(f"   Timesteps: {timesteps}")
    print(f"   Parallel envs: {n_envs}")
    print(f"   Render frequency: every {render_freq} episodes\n")
    
    # Create environment(s)
    level_data, start_pos = get_level(level)
    
    if n_envs == 1:
        env = DummyVecEnv([lambda: NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)])
    else:
        # Use subprocess parallelism for speed
        env = SubprocVecEnv([make_env(level, i) for i in range(n_envs)])
    
    # Create model - optimized for M4
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=1024,  # Smaller = more frequent updates
        batch_size=256,  # Larger batch for M4
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        device=device,
        tensorboard_log="./logs/visual",
    )
    
    # Create callback
    viz_callback = VisualTrainingCallback(
        render_freq=render_freq,
        fps=fps,
        scale=scale,
        verbose=verbose,
    )
    
    try:
        # Train!
        model.learn(
            total_timesteps=timesteps,
            callback=viz_callback,
            progress_bar=True,
        )
        
        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"\n✅ Model saved to {save_path}")
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted. Saving current model...")
        model.save(save_path + "_interrupted")
        print(f"   Saved to {save_path}_interrupted")
    
    finally:
        env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with live visualization')
    parser.add_argument('--level', type=str, default='simple', 
                       help='Level: simple, arena, tutorial, level1, level2')
    parser.add_argument('--timesteps', type=int, default=500000, help='Training timesteps')
    parser.add_argument('--render-freq', type=int, default=20, 
                       help='Render every N episodes (higher = faster training)')
    parser.add_argument('--fps', type=int, default=60, help='Rendering FPS')
    parser.add_argument('--scale', type=float, default=1.0, help='Display scale')
    parser.add_argument('--save', type=str, default='./models/visual_model', help='Save path')
    parser.add_argument('--envs', type=int, default=1, 
                       help='Parallel envs (1 for visual, more for speed but no render)')
    
    args = parser.parse_args()
    
    train_visual(
        level=args.level,
        timesteps=args.timesteps,
        render_freq=args.render_freq,
        fps=args.fps,
        scale=args.scale,
        save_path=args.save,
        n_envs=args.envs,
    )
