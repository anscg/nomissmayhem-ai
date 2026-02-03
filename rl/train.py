"""
Training script for No Miss Mayhem RL agent.
Uses Stable-Baselines3 with PPO.

Optimized for Apple Silicon (M1/M2/M3/M4) with MPS acceleration.
"""

import os
import argparse
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env import NoMissMayhemEnv, DiscreteActionWrapper
from levels import get_level, ALL_LEVELS


class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.deaths = 0
    
    def _on_step(self) -> bool:
        # Log episode info when available
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                
                if info.get('win'):
                    self.wins += 1
                elif 'death_cause' in info:
                    self.deaths += 1
        
        # Log every 1000 steps
        if self.n_calls % 1000 == 0 and len(self.episode_rewards) > 0:
            self.logger.record('custom/mean_reward', np.mean(self.episode_rewards[-100:]))
            self.logger.record('custom/mean_length', np.mean(self.episode_lengths[-100:]))
            self.logger.record('custom/win_rate', self.wins / max(1, self.wins + self.deaths))
        
        return True


def make_env(level_name: str, rank: int, seed: int = 0, max_steps: int = 5000, randomize: bool = False):
    """Create environment factory"""
    def _init():
        # Each environment gets a different seed for randomization
        env_seed = seed + rank
        level_data, start_pos = get_level(level_name, randomize=randomize, seed=env_seed)
        env = NoMissMayhemEnv(
            level_data=level_data,
            start_pos=start_pos,
            max_steps=max_steps
        )
        env = Monitor(env)
        env.reset(seed=env_seed)
        return env
    set_random_seed(seed)
    return _init


def get_device(force_gpu: bool = False):
    """Get best available device (MPS for M-series Macs, CUDA, or CPU)
    
    Note: For MLP policies with parallel envs, CPU is often FASTER because:
    - Python environment logic is the bottleneck (GIL-bound)
    - Small MLP networks don't benefit from GPU parallelism
    - GPU transfer overhead adds latency
    - CPU allows better parallelization across environments
    """
    if force_gpu:
        if torch.backends.mps.is_available():
            print("🍎 Using MPS (Metal) acceleration on Apple Silicon")
            return 'mps'
        elif torch.cuda.is_available():
            print("🎮 Using CUDA acceleration")
            return 'cuda'
    
    print("💻 Using CPU (recommended for MLP + parallel envs)")
    return 'cpu'


def train(
    level: str = 'simple',
    algorithm: str = 'ppo',
    total_timesteps: int = 1_000_000,
    n_envs: int = 16,  # Increased for M4
    learning_rate: float = 3e-4,
    batch_size: int = 256,  # Larger batch for faster training
    n_steps: int = 1024,  # Smaller n_steps = more frequent updates
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    save_freq: int = 50000,
    eval_freq: int = 10000,
    seed: int = 42,
    device: str = 'auto',
    log_dir: str = './logs',
    model_dir: str = './models',
    resume: Optional[str] = None,
    randomize: bool = False,
):
    """
    Train the RL agent.
    
    Args:
        level: Level to train on ('simple', 'tutorial', 'level1', 'level2', 'arena')
        algorithm: 'ppo', 'sac', or 'dqn'
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        batch_size: Batch size for updates
        n_steps: Steps per environment per update (PPO)
        n_epochs: Epochs per update (PPO)
        gamma: Discount factor
        gae_lambda: GAE lambda (PPO)
        clip_range: Clipping range (PPO)
        ent_coef: Entropy coefficient
        save_freq: Checkpoint save frequency
        eval_freq: Evaluation frequency
        seed: Random seed
        device: 'auto', 'cuda', 'mps', or 'cpu'
        log_dir: Tensorboard log directory
        model_dir: Model save directory
        resume: Path to model to resume from
        randomize: Randomize enemy and door positions for better generalization
    """
    
    # Auto-detect best device
    if device == 'auto':
        device = get_device()
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_{level}_{timestamp}"
    
    log_path = os.path.join(log_dir, run_name)
    model_path = os.path.join(model_dir, run_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Training {algorithm.upper()} on level '{level}'")
    print(f"Logs: {log_path}")
    print(f"Models: {model_path}")
    
    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(level, i, seed, max_steps=5000, randomize=randomize) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(level, 0, seed, max_steps=5000, randomize=randomize)])
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Create evaluation environment (don't randomize for consistent evaluation)
    eval_env = DummyVecEnv([make_env(level, 0, seed + 1000, max_steps=5000, randomize=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Create or load model
    if resume:
        print(f"Resuming from {resume}")
        if algorithm == 'ppo':
            model = PPO.load(resume, env=env, device=device)
        elif algorithm == 'sac':
            model = SAC.load(resume, env=env, device=device)
        elif algorithm == 'dqn':
            model = DQN.load(resume, env=env, device=device)
    else:
        if algorithm == 'ppo':
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=1,
                tensorboard_log=log_path,
                device=device,
                seed=seed,
            )
        elif algorithm == 'sac':
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef='auto',
                verbose=1,
                tensorboard_log=log_path,
                device=device,
                seed=seed,
            )
        elif algorithm == 'dqn':
            # DQN needs discrete actions
            env = DiscreteActionWrapper(env)
            eval_env = DiscreteActionWrapper(eval_env)
            model = DQN(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                gamma=gamma,
                verbose=1,
                tensorboard_log=log_path,
                device=device,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_path,
        name_prefix='checkpoint',
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    tensorboard_callback = TensorboardCallback()
    
    callbacks = CallbackList([
        checkpoint_callback,
        eval_callback,
        tensorboard_callback,
    ])
    
    # Train
    print(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(model_path, 'final_model')
    model.save(final_path)
    env.save(os.path.join(model_path, 'vec_normalize.pkl'))
    
    print(f"Training complete! Model saved to {final_path}")
    
    return model, env


def evaluate(
    model_path: str,
    level: str = 'simple',
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
):
    """
    Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        level: Level to evaluate on
        n_episodes: Number of episodes to run
        render: Whether to render (not implemented yet)
        deterministic: Use deterministic actions
    """
    from stable_baselines3 import PPO
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    level_data, start_pos = get_level(level)
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    # Load normalization stats if available
    norm_path = os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl')
    if os.path.exists(norm_path):
        from stable_baselines3.common.vec_env import VecNormalize
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    # Run episodes
    wins = 0
    total_rewards = []
    total_steps = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if info.get('win'):
            wins += 1
            print(f"Episode {ep+1}: WIN! Reward={episode_reward:.2f}, Steps={steps}")
        else:
            cause = info.get('death_cause', 'unknown')
            print(f"Episode {ep+1}: LOSS ({cause}), Reward={episode_reward:.2f}, Steps={steps}")
    
    print(f"\n=== Results ===")
    print(f"Win rate: {wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Mean steps: {np.mean(total_steps):.0f} ± {np.std(total_steps):.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train No Miss Mayhem RL agent')
    parser.add_argument('--level', type=str, default='simple',
                        choices=list(ALL_LEVELS.keys()),
                        help='Level to train on')
    parser.add_argument('--algorithm', type=str, default='ppo',
                        choices=['ppo', 'sac', 'dqn'],
                        help='RL algorithm')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                        help='Total training timesteps')
    parser.add_argument('--n_envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume training from')
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to model to evaluate (skips training)')
    parser.add_argument('--randomize', action='store_true',
                        help='Randomize enemy and door positions for better generalization')
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args.eval, args.level)
    else:
        train(
            level=args.level,
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            randomize=args.randomize,
        )
