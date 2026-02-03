"""
Curriculum learning script - train agent progressively on harder levels.

Starts with simple scenarios and gradually increases difficulty.
This helps the agent learn fundamental skills before tackling complex levels.
"""

import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from env import NoMissMayhemEnv
from levels import get_level


def get_device():
    """CPU is fastest for MLP + parallel envs"""
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


class CurriculumCallback(BaseCallback):
    """Track performance and decide when to advance curriculum"""
    
    def __init__(self, check_freq: int = 10000, win_threshold: float = 0.3, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.win_threshold = win_threshold
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.ready_to_advance = False
    
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
            if total > 50:  # Need enough samples
                win_rate = self.wins / total
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                
                if self.verbose:
                    print(f"\n   📊 Step {self.n_calls:,}: Win rate {win_rate*100:.1f}% | "
                          f"Avg reward: {avg_reward:.1f} | Episodes: {total}")
                
                if win_rate >= self.win_threshold:
                    self.ready_to_advance = True
                    if self.verbose:
                        print(f"   ✅ Win rate threshold reached! Ready to advance curriculum.")
        
        return True
    
    def reset_stats(self):
        """Reset stats for new curriculum stage"""
        self.wins = 0
        self.losses = 0
        self.episode_rewards = []
        self.ready_to_advance = False


def train_curriculum(
    levels: list = ['simple', 'tutorial', 'level1'],
    timesteps_per_stage: int = 500_000,
    win_threshold: float = 0.25,
    n_envs: int = 16,
    save_path: str = './models/curriculum_model',
    seed: int = 42,
):
    """
    Train using curriculum learning.
    
    Args:
        levels: List of levels in order of difficulty
        timesteps_per_stage: Max timesteps per curriculum stage
        win_threshold: Win rate needed to advance (0-1)
        n_envs: Parallel environments
        save_path: Save path
        seed: Random seed
    """
    device = get_device()
    
    print(f"\n🎓 Curriculum Learning")
    print(f"   Stages: {' → '.join(levels)}")
    print(f"   Win threshold: {win_threshold*100}%")
    print(f"   Max timesteps/stage: {timesteps_per_stage:,}")
    print(f"   Parallel envs: {n_envs}")
    print(f"   Device: {device}\n")
    
    model = None
    
    for stage_idx, level in enumerate(levels):
        print(f"\n{'='*60}")
        print(f"📚 Stage {stage_idx + 1}/{len(levels)}: {level.upper()}")
        print(f"{'='*60}\n")
        
        # Create envs for this stage
        env = SubprocVecEnv([make_env(level, i, seed + stage_idx * 100) for i in range(n_envs)])
        
        # Normalize (or load previous normalization stats)
        if model is None:
            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        else:
            # Keep normalization stats from previous stage
            old_vec_norm = VecNormalize.load(f"{save_path}_stage{stage_idx-1}_vecnorm.pkl", env)
            env = old_vec_norm
            env.venv = env.venv  # Update underlying env
        
        # Create or continue model
        if model is None:
            # First stage - create new model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.02,  # Higher entropy for early exploration
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=0,
                device=device,
                seed=seed,
            )
        else:
            # Continue with same model but new environment
            model.set_env(env)
            # Reduce learning rate as we progress
            model.learning_rate = 3e-4 / (1 + stage_idx * 0.5)
            # Reduce entropy coefficient (less exploration needed)
            model.ent_coef = 0.02 / (1 + stage_idx)
        
        # Callback for this stage
        callback = CurriculumCallback(
            check_freq=10000,
            win_threshold=win_threshold,
            verbose=1
        )
        
        try:
            print(f"Training on '{level}'...\n")
            
            model.learn(
                total_timesteps=timesteps_per_stage,
                callback=callback,
                progress_bar=True,
                reset_num_timesteps=False,  # Keep total timesteps cumulative
            )
            
            # Save checkpoint for this stage
            stage_save_path = f"{save_path}_stage{stage_idx}_{level}"
            model.save(stage_save_path)
            env.save(f"{stage_save_path}_vecnorm.pkl")
            
            print(f"\n   💾 Saved checkpoint: {stage_save_path}.zip")
            
            # Check if ready to advance
            if callback.ready_to_advance:
                print(f"   ✅ Mastered '{level}'! Moving to next stage...")
            else:
                print(f"   ⚠️  Completed timesteps but didn't reach threshold.")
                print(f"      Continuing to next stage anyway...")
        
        except KeyboardInterrupt:
            print(f"\n\n⏸️  Training interrupted at stage {stage_idx + 1}")
            stage_save_path = f"{save_path}_stage{stage_idx}_{level}_interrupted"
            model.save(stage_save_path)
            env.save(f"{stage_save_path}_vecnorm.pkl")
            print(f"   💾 Saved: {stage_save_path}.zip")
            env.close()
            return
        
        env.close()
    
    # Save final model
    final_save_path = f"{save_path}_final"
    model.save(final_save_path)
    print(f"\n\n🎉 Curriculum complete!")
    print(f"   Final model: {final_save_path}.zip")
    print(f"\n   Watch it play: python visualize.py --model {final_save_path}.zip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curriculum learning')
    parser.add_argument('--levels', nargs='+', default=['simple', 'tutorial', 'level1'],
                       help='Levels in order of difficulty')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Max timesteps per stage')
    parser.add_argument('--win-threshold', type=float, default=0.25,
                       help='Win rate threshold to advance (0-1)')
    parser.add_argument('--envs', type=int, default=16,
                       help='Parallel environments')
    parser.add_argument('--save', type=str, default='./models/curriculum_model',
                       help='Save path prefix')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    train_curriculum(
        levels=args.levels,
        timesteps_per_stage=args.timesteps,
        win_threshold=args.win_threshold,
        n_envs=args.envs,
        save_path=args.save,
        seed=args.seed,
    )
