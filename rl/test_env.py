#!/usr/bin/env python3
"""
Quick test to verify the environment works correctly.
Run this before training to catch any bugs.
"""

import numpy as np
import time
from env import NoMissMayhemEnv
from levels import get_level, ALL_LEVELS


def test_basic():
    """Test basic environment functionality"""
    print("=" * 60)
    print("TEST 1: Basic Environment")
    print("=" * 60)
    
    level_data, start_pos = get_level('simple')
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    obs, _ = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
    print(f"✓ 10 random steps completed")
    print(f"  Final reward: {reward:.4f}")
    print()


def test_determinism():
    """Test that same actions produce same results"""
    print("=" * 60)
    print("TEST 2: Determinism")
    print("=" * 60)
    
    level_data, start_pos = get_level('simple')
    env1 = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    env2 = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    # Same seed should give same initial state
    assert np.allclose(obs1, obs2), "Initial observations differ!"
    print("✓ Same seed produces same initial state")
    
    # Same actions should give same results
    np.random.seed(123)
    actions = [env1.action_space.sample() for _ in range(50)]
    
    for i, action in enumerate(actions):
        obs1, r1, t1, tr1, _ = env1.step(action)
        obs2, r2, t2, tr2, _ = env2.step(action)
        
        if not np.allclose(obs1, obs2, rtol=1e-5):
            print(f"✗ Observations diverged at step {i}")
            diff_idx = np.where(~np.isclose(obs1, obs2, rtol=1e-5))[0]
            print(f"  Differing indices: {diff_idx[:10]}...")
            break
        
        if t1 or t2:
            break
    else:
        print("✓ 50 identical steps verified")
    print()


def test_speed():
    """Test environment speed"""
    print("=" * 60)
    print("TEST 3: Speed Benchmark")
    print("=" * 60)
    
    level_data, start_pos = get_level('arena')
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    n_steps = 10000
    obs, _ = env.reset()
    
    start = time.time()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
    
    elapsed = time.time() - start
    fps = n_steps / elapsed
    
    print(f"✓ {n_steps} steps in {elapsed:.2f}s")
    print(f"  Speed: {fps:.0f} FPS")
    print(f"  (Target: 1000+ FPS for efficient training)")
    print()


def test_combat():
    """Test combat mechanics"""
    print("=" * 60)
    print("TEST 4: Combat Mechanics")
    print("=" * 60)
    
    level_data, start_pos = get_level('arena')
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    obs, _ = env.reset()
    
    # Shoot toward enemies
    initial_enemies = len([e for e in env._get_current_room().enemies if e.is_active])
    print(f"  Initial enemies: {initial_enemies}")
    
    # Aim at first enemy and shoot
    enemy = env._get_current_room().enemies[0]
    aim_angle = np.arctan2(enemy.y - env.player.y, enemy.x - env.player.x)
    
    kills = 0
    damage_dealt = 0
    
    for _ in range(500):
        # Always shoot, move toward enemy
        action = np.array([0.5, 0.5, aim_angle, 1.0, 0.0])
        obs, reward, term, trunc, info = env.step(action)
        
        if term or trunc:
            break
        
        # Update aim
        active_enemies = [e for e in env._get_current_room().enemies if e.is_active]
        if active_enemies:
            enemy = active_enemies[0]
            aim_angle = np.arctan2(enemy.y - env.player.y, enemy.x - env.player.x)
    
    final_enemies = len([e for e in env._get_current_room().enemies if e.is_active])
    kills = initial_enemies - final_enemies
    
    print(f"  Enemies killed: {kills}")
    print(f"  Player health: {env.player.health}")
    print(f"  Coins collected: {env.player.money}")
    
    if kills > 0:
        print("✓ Combat system working")
    else:
        print("⚠ No kills - may need investigation")
    print()


def test_navigation():
    """Test room navigation"""
    print("=" * 60)
    print("TEST 5: Room Navigation")
    print("=" * 60)
    
    level_data, start_pos = get_level('simple')
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    obs, _ = env.reset()
    initial_room = tuple(env.room_pos)
    print(f"  Starting room: {initial_room}")
    
    # Move down and shoot door
    for step in range(300):
        # Move down and shoot door
        action = np.array([0, 1, np.pi/2, 1.0, 0.0])  # Move down, aim down, shoot
        obs, reward, term, trunc, info = env.step(action)
        
        if tuple(env.room_pos) != initial_room:
            print(f"✓ Room transition at step {step}")
            print(f"  New room: {tuple(env.room_pos)}")
            break
        
        if term or trunc:
            print(f"  Episode ended: term={term}, trunc={trunc}")
            break
    else:
        print("⚠ No room transition in 300 steps")
    print()


def test_all_levels():
    """Test all level definitions"""
    print("=" * 60)
    print("TEST 6: All Levels")
    print("=" * 60)
    
    for level_name in ALL_LEVELS.keys():
        try:
            level_data, start_pos = get_level(level_name)
            env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
            obs, _ = env.reset()
            
            # Take a few random steps
            for _ in range(100):
                action = env.action_space.sample()
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
            
            print(f"✓ {level_name}: OK")
        except Exception as e:
            print(f"✗ {level_name}: {e}")
    print()


def test_observation_values():
    """Test that observation values are reasonable"""
    print("=" * 60)
    print("TEST 7: Observation Values")
    print("=" * 60)
    
    level_data, start_pos = get_level('arena')
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    obs_min = np.inf * np.ones(env.observation_space.shape)
    obs_max = -np.inf * np.ones(env.observation_space.shape)
    
    for _ in range(100):
        obs, _ = env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            
            obs_min = np.minimum(obs_min, obs)
            obs_max = np.maximum(obs_max, obs)
            
            if term or trunc:
                break
    
    print(f"  Observation range: [{obs_min.min():.3f}, {obs_max.max():.3f}]")
    
    # Check for NaN or Inf
    if np.any(np.isnan(obs_min)) or np.any(np.isnan(obs_max)):
        print("✗ NaN values detected!")
    elif np.any(np.isinf(obs_min)) or np.any(np.isinf(obs_max)):
        print("✗ Inf values detected!")
    else:
        print("✓ No NaN or Inf values")
    
    # Check normalization
    if obs_max.max() > 10 or obs_min.min() < -10:
        print("⚠ Values outside [-10, 10] - consider normalizing")
    else:
        print("✓ Values in reasonable range")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("NO MISS MAYHEM - ENVIRONMENT TESTS")
    print("=" * 60 + "\n")
    
    test_basic()
    test_determinism()
    test_speed()
    test_combat()
    test_navigation()
    test_all_levels()
    test_observation_values()
    
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
