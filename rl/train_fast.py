#!/usr/bin/env python3
"""
Fast training script optimized for Apple Silicon M-series chips.
Trains without visualization for maximum speed, then you can watch the result.

Usage:
    python train_fast.py --level simple --timesteps 1000000
    
Then watch:
    python visualize.py --model ./models/fast_model.zip
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env import NoMissMayhemEnv
from levels import get_level


def get_device(force_gpu: bool = False):
    """Get best available device
    
    Note: For MLP policies, CPU is often FASTER than GPU/MPS because:
    - Small networks don't benefit from GPU parallelism
    - Environment logic (Python) is the bottleneck, not NN
    - GPU transfer overhead slows things down
    """
    if force_gpu:
        if torch.backends.mps.is_available():
            print("🍎 Using MPS (Metal) - may be slower for MLP policies")
            return 'mps'
        elif torch.cuda.is_available():
            print("🎮 Using CUDA - may be slower for MLP policies")
            return 'cuda'
    
    print("💻 Using CPU (optimal for MLP policy + parallel envs)")
    return 'cpu'


def make_env(level_name: str, rank: int, seed: int = 0):
    """Create environment factory"""
    def _init():
        level_data, start_pos = get_level(level_name)
        env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos, max_steps=5000)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class ProgressCallback(BaseCallback):
    """Show training progress with win rate"""
    
    def __init__(self, check_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
    
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                if info.get('win'):
                    self.wins += 1
                else:
                    self.losses += 1
        
        if self.n_calls % self.check_freq == 0:
            total = self.wins + self.losses
            if total > 0:
                win_rate = self.wins / total * 100
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                print(f"   Step {self.n_calls:,}: Win rate {win_rate:.1f}% | "
                      f"Avg reward: {avg_reward:.1f} | Episodes: {total}")
        
        return True


def train_fast(
    level: str = 'simple',
    timesteps: int = 1_000_000,
    n_envs: int = 16,
    save_path: str = './models/fast_model',
    checkpoint_freq: int = 500000,  # Increased from 50000 - save less often
    seed: int = 42,
    force_gpu: bool = False,
):
    """
    Fast training without visualization.
    
    Optimized for M4 MacBook:
    - 16 parallel environments
    - CPU (fastest for MLP + parallel envs)
    - Large batch sizes
    - Frequent checkpoints
    """
    device = get_device(force_gpu)
    
    print(f"\n🚀 Fast Training Mode")
    print(f"   Level: {level}")
    print(f"   Timesteps: {timesteps:,}")
    print(f"   Parallel envs: {n_envs}")
    print(f"   Device: {device}")
    print(f"   Save path: {save_path}\n")
    
    # Create parallel environments
    env = SubprocVecEnv([make_env(level, i, seed) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create model optimized for speed
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,         # Steps per env before update
        batch_size=512,       # Large batch for M4
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,        # Entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,            # Quiet - we have our own progress
        device=device,
        seed=seed,
    )
    
    # Callbacks
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else './models', exist_ok=True)
    
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=os.path.dirname(save_path) or './models',
        name_prefix=os.path.basename(save_path),
    )
    
    progress_cb = ProgressCallback(check_freq=10000)
    
    try:
        print("Training started... (Ctrl+C to stop and save)\n")
        
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_cb, progress_cb],
            progress_bar=True,
        )
        
        # Save final model
        model.save(save_path)
        env.save(f"{save_path}_vecnorm.pkl")
        
        print(f"\n✅ Training complete!")
        print(f"   Model saved to: {save_path}.zip")
        print(f"\n   Watch it play: python visualize.py --model {save_path}.zip")
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted. Saving...")
        model.save(f"{save_path}_interrupted")
        env.save(f"{save_path}_interrupted_vecnorm.pkl")
        print(f"   Saved to: {save_path}_interrupted.zip")
        print(f"\n   Watch it play: python visualize.py --model {save_path}_interrupted.zip")
    
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast training for M4 Mac')
    parser.add_argument('--level', type=str, default='simple',
                       help='Level: simple, arena, tutorial, level1, level2')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    parser.add_argument('--envs', type=int, default=16,
                       help='Number of parallel environments')
    parser.add_argument('--save', type=str, default='./models/fast_model',
                       help='Save path (without .zip)')
    parser.add_argument('--checkpoint-freq', type=int, default=500000,
                       help='Save checkpoint every N steps (default: 500k)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--force-gpu', action='store_true',
                       help='Force GPU/MPS (not recommended for MLP)')
    
    args = parser.parse_args()
    
    train_fast(
        level=args.level,
        timesteps=args.timesteps,
        n_envs=args.envs,
        save_path=args.save,
        checkpoint_freq=args.checkpoint_freq,
        seed=args.seed,
        force_gpu=args.force_gpu,
    )
