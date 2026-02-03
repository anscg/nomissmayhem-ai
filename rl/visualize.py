"""
Visual renderer for No Miss Mayhem RL environment.
Uses pygame to display the game at normal FPS during training/evaluation.
"""

import pygame
import numpy as np
import math
import time
from typing import Optional

# Colors matching the JS game
COLORS = {
    'background': (17, 17, 17),
    'player': (68, 136, 255),
    'player_invuln': (255, 68, 68),
    'player_no_dash': (133, 177, 255),
    'enemy_regular': (255, 0, 0),
    'enemy_attacker': (255, 0, 255),
    'enemy_shielded': (0, 255, 0),
    'enemy_laser': (156, 25, 255),
    'enemy_key': (245, 197, 66),
    'projectile': (255, 255, 255),
    'projectile_enemy': (255, 100, 100),
    'laser': (255, 255, 255),
    'coin_bronze': (205, 127, 50),
    'coin_silver': (192, 192, 192),
    'coin_gold': (255, 215, 0),
    'key': (255, 215, 0),
    'health': (255, 0, 0),
    'door_closed': (255, 0, 0),
    'door_open': (0, 255, 0),
    'door_key': (255, 215, 0),
    'shield': (150, 200, 255),
    'text': (255, 255, 255),
    'health_bar_bg': (68, 68, 68),
    'health_bar': (51, 255, 51),
    'health_bar_warn': (255, 255, 51),
    'health_bar_danger': (255, 51, 51),
}


class GameRenderer:
    """Pygame-based renderer for the RL environment"""
    
    def __init__(self, width: int = 600, height: int = 600, scale: float = 1.0, fps: int = 60):
        """
        Initialize the renderer.
        
        Args:
            width: Canvas width (should match CANVAS_WIDTH)
            height: Canvas height (should match CANVAS_HEIGHT)
            scale: Scale factor for display (e.g., 1.5 for larger window)
            fps: Target frames per second
        """
        self.width = width
        self.height = height
        self.scale = scale
        self.fps = fps
        
        self.display_width = int(width * scale)
        self.display_height = int(height * scale)
        
        pygame.init()
        pygame.display.set_caption("No Miss Mayhem - RL Training")
        
        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
        self.surface = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        self.running = True
        self.paused = False
        self.step_mode = False  # Step one frame at a time
        self.should_step = False
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if window closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_s:
                    self.step_mode = not self.step_mode
                    self.paused = self.step_mode
                elif event.key == pygame.K_n and self.step_mode:
                    self.should_step = True
        return True
    
    def render(self, env, action: Optional[np.ndarray] = None, 
               episode: int = 0, step: int = 0, reward: float = 0,
               total_reward: float = 0):
        """
        Render the current environment state.
        
        Args:
            env: NoMissMayhemEnv instance
            action: Current action being taken
            episode: Episode number
            step: Step number
            reward: Current step reward
            total_reward: Total episode reward
        """
        if not self.handle_events():
            return False
        
        if self.paused and not self.should_step:
            # Still render but don't advance
            self._draw_pause_overlay()
            pygame.display.flip()
            self.clock.tick(30)
            return True
        
        self.should_step = False
        
        # Clear surface
        self.surface.fill(COLORS['background'])
        
        # Get current room
        room = env._get_current_room()
        player = env.player
        
        # Draw room background (simple grid)
        self._draw_room_grid()
        
        # Draw doors
        self._draw_doors(room)
        
        # Draw lasers (behind everything)
        self._draw_lasers(room, env.game_time_ms)
        
        # Draw collectibles
        self._draw_collectibles(room)
        
        # Draw enemies
        self._draw_enemies(room)
        
        # Draw projectiles
        self._draw_projectiles(room)
        
        # Draw player
        self._draw_player(player, action)
        
        # Draw UI
        self._draw_ui(env, episode, step, reward, total_reward, action)
        
        # Scale and display
        if self.scale != 1.0:
            scaled = pygame.transform.scale(self.surface, (self.display_width, self.display_height))
            self.screen.blit(scaled, (0, 0))
        else:
            self.screen.blit(self.surface, (0, 0))
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        
        return True
    
    def _draw_room_grid(self):
        """Draw subtle grid lines"""
        grid_color = (30, 30, 30)
        for x in range(0, self.width, 50):
            pygame.draw.line(self.surface, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, 50):
            pygame.draw.line(self.surface, grid_color, (0, y), (self.width, y))
    
    def _draw_doors(self, room):
        """Draw doors on all four sides"""
        door_positions = {
            'up': (260, 0, 80, 20),
            'down': (260, 580, 80, 20),
            'left': (0, 260, 20, 80),
            'right': (580, 260, 20, 80),
        }
        
        for direction, (x, y, w, h) in door_positions.items():
            door = room.doors[direction]
            
            if door.type == 'wall':
                continue  # Don't draw walls
            
            # Background (red = locked)
            pygame.draw.rect(self.surface, COLORS['door_closed'], (x, y, w, h))
            
            # Progress fill
            if door.open:
                fill_color = COLORS['door_open']
                progress = 1.0
            elif door.type == 'key':
                fill_color = COLORS['door_key']
                progress = 0.0
            else:
                fill_color = COLORS['door_open']
                progress = door.get_progress()
            
            if progress > 0:
                if w > h:  # Horizontal door
                    fill_w = int(w * progress)
                    pygame.draw.rect(self.surface, fill_color, (x, y, fill_w, h))
                else:  # Vertical door
                    fill_h = int(h * progress)
                    pygame.draw.rect(self.surface, fill_color, (x, y, w, fill_h))
    
    def _draw_player(self, player, action: Optional[np.ndarray] = None):
        """Draw the player"""
        # Determine color
        if player.is_invulnerable:
            # Flash between normal and red
            if int(time.time() * 10) % 2 == 0:
                color = COLORS['player_invuln']
            else:
                color = COLORS['player'] if player.can_dash else COLORS['player_no_dash']
        elif not player.can_dash:
            color = COLORS['player_no_dash']
        else:
            color = COLORS['player']
        
        # Draw player circle
        pygame.draw.circle(self.surface, color, (int(player.x), int(player.y)), int(player.radius))
        
        # Draw aim direction
        if action is not None and len(action) >= 6:
            # New format: [move_x, move_y, aim_x, aim_y, shoot, dash]
            aim_x = float(action[2]) if hasattr(action[2], '__float__') else action[2]
            aim_y = float(action[3]) if hasattr(action[3], '__float__') else action[3]
            aim_angle = math.atan2(aim_y, aim_x)
            end_x = player.x + math.cos(aim_angle) * 40
            end_y = player.y + math.sin(aim_angle) * 40
            pygame.draw.line(self.surface, COLORS['text'], 
                           (int(player.x), int(player.y)), 
                           (int(end_x), int(end_y)), 2)
            
            # Draw backward line if double
            if player.double:
                end_x2 = player.x + math.cos(aim_angle + math.pi) * 40
                end_y2 = player.y + math.sin(aim_angle + math.pi) * 40
                pygame.draw.line(self.surface, COLORS['text'],
                               (int(player.x), int(player.y)),
                               (int(end_x2), int(end_y2)), 2)
        elif action is not None and len(action) >= 3:
            # Old format: [move_x, move_y, aim_angle, shoot, dash]
            aim_angle = float(action[2]) if hasattr(action[2], '__float__') else action[2]
            end_x = player.x + math.cos(aim_angle) * 40
            end_y = player.y + math.sin(aim_angle) * 40
            pygame.draw.line(self.surface, COLORS['text'], 
                           (int(player.x), int(player.y)), 
                           (int(end_x), int(end_y)), 2)
            
            # Draw backward line if double
            if player.double:
                end_x2 = player.x + math.cos(aim_angle + math.pi) * 40
                end_y2 = player.y + math.sin(aim_angle + math.pi) * 40
                pygame.draw.line(self.surface, COLORS['text'],
                               (int(player.x), int(player.y)),
                               (int(end_x2), int(end_y2)), 2)
    
    def _draw_enemies(self, room):
        """Draw all enemies"""
        for enemy in room.enemies:
            if not enemy.is_active:
                continue
            
            # Determine color
            if enemy.has_key:
                color = COLORS['enemy_key']
            elif enemy.type == 'regular':
                color = COLORS['enemy_regular']
            elif enemy.type == 'attacker':
                color = COLORS['enemy_attacker']
            elif enemy.type == 'shielded':
                color = COLORS['enemy_shielded']
            elif enemy.type == 'laser':
                color = COLORS['enemy_laser']
            else:
                color = COLORS['enemy_regular']
            
            # Draw enemy
            pygame.draw.circle(self.surface, color, 
                             (int(enemy.x), int(enemy.y)), int(enemy.radius))
            
            # Draw shield for shielded enemies
            if enemy.type == 'shielded' and hasattr(enemy, 'shield_angle'):
                shield_radius = enemy.radius + 10
                start_angle = enemy.shield_angle - enemy.shield_arc / 2
                end_angle = enemy.shield_angle + enemy.shield_arc / 2
                
                # Draw arc
                rect = pygame.Rect(
                    int(enemy.x - shield_radius),
                    int(enemy.y - shield_radius),
                    int(shield_radius * 2),
                    int(shield_radius * 2)
                )
                pygame.draw.arc(self.surface, COLORS['shield'], rect, 
                              -end_angle, -start_angle, 4)
            
            # Draw health bar
            bar_width = 40
            bar_height = 4
            bar_x = enemy.x - bar_width / 2
            bar_y = enemy.y - enemy.radius - 10
            
            # Background
            pygame.draw.rect(self.surface, COLORS['door_closed'],
                           (int(bar_x), int(bar_y), bar_width, bar_height))
            # Health fill
            health_pct = enemy.health / enemy.max_health
            pygame.draw.rect(self.surface, COLORS['door_open'],
                           (int(bar_x), int(bar_y), int(bar_width * health_pct), bar_height))
    
    def _draw_projectiles(self, room):
        """Draw all projectiles"""
        for proj in room.projectiles:
            if proj.is_enemy:
                color = COLORS['projectile_enemy']
            else:
                # Color based on bounces (gets redder)
                bounce_factor = min(proj.bounces / 10, 1.0)
                r = 255
                g = int(255 * (1 - bounce_factor * 0.6))
                b = int(255 * (1 - bounce_factor * 0.6))
                color = (r, g, b)
            
            pygame.draw.circle(self.surface, color,
                             (int(proj.x), int(proj.y)), int(proj.radius))
    
    def _draw_lasers(self, room, game_time_ms: float):
        """Draw laser beams"""
        for laser in room.lasers:
            if laser.is_expired(game_time_ms):
                continue
            
            # Calculate alpha based on time remaining
            elapsed = game_time_ms - laser.fire_time
            if elapsed < laser.delay:
                # Charging - draw warning line
                alpha = 100
            else:
                # Active - draw full laser
                remaining = laser.remain_time - elapsed
                alpha = int(255 * (remaining / laser.remain_time))
            
            # Draw laser line
            end_x = laser.x + math.cos(laser.angle) * 1000
            end_y = laser.y + math.sin(laser.angle) * 1000
            
            # Create a surface for the laser with alpha
            if elapsed >= laser.delay:
                pygame.draw.line(self.surface, (255, 255, 255),
                               (int(laser.x), int(laser.y)),
                               (int(end_x), int(end_y)), 8)
            else:
                # Warning line (thinner, dimmer)
                pygame.draw.line(self.surface, (255, 100, 100),
                               (int(laser.x), int(laser.y)),
                               (int(end_x), int(end_y)), 2)
    
    def _draw_collectibles(self, room):
        """Draw coins, keys, and health pickups"""
        for item in room.collectibles:
            if 'coin' in item.type:
                if 'bronze' in item.type:
                    color = COLORS['coin_bronze']
                elif 'silver' in item.type:
                    color = COLORS['coin_silver']
                else:
                    color = COLORS['coin_gold']
            elif item.type == 'key':
                color = COLORS['key']
            elif item.type == 'health':
                color = COLORS['health']
            else:
                color = COLORS['text']
            
            pygame.draw.circle(self.surface, color,
                             (int(item.x), int(item.y)), int(item.radius))
            pygame.draw.circle(self.surface, (0, 0, 0),
                             (int(item.x), int(item.y)), int(item.radius), 1)
    
    def _draw_ui(self, env, episode: int, step: int, reward: float, 
                 total_reward: float, action: Optional[np.ndarray] = None):
        """Draw UI overlay"""
        player = env.player
        
        # Health bar
        bar_width = 150
        bar_height = 15
        bar_x = 10
        bar_y = 10
        
        pygame.draw.rect(self.surface, COLORS['health_bar_bg'],
                        (bar_x, bar_y, bar_width, bar_height))
        
        health_pct = player.health / player.max_health
        if health_pct > 0.6:
            color = COLORS['health_bar']
        elif health_pct > 0.3:
            color = COLORS['health_bar_warn']
        else:
            color = COLORS['health_bar_danger']
        
        pygame.draw.rect(self.surface, color,
                        (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        
        # Text info
        y_offset = 35
        texts = [
            f"Episode: {episode}  Step: {step}",
            f"Reward: {reward:.2f}  Total: {total_reward:.2f}",
            f"HP: {player.health}/{player.max_health}  Coins: {player.money}",
            f"Room: {env.room_pos}  Keys: {len(player.keys)}",
        ]
        
        if action is not None and len(action) >= 6:
            # New action format: [move_x, move_y, aim_x, aim_y, shoot, dash]
            a0 = float(action[0]) if hasattr(action[0], '__float__') else action[0]
            a1 = float(action[1]) if hasattr(action[1], '__float__') else action[1]
            aim_x = float(action[2]) if hasattr(action[2], '__float__') else action[2]
            aim_y = float(action[3]) if hasattr(action[3], '__float__') else action[3]
            a4 = float(action[4]) if hasattr(action[4], '__float__') else action[4]
            a5 = float(action[5]) if hasattr(action[5], '__float__') else action[5]
            
            # Convert aim_x, aim_y to angle for display
            aim_angle_deg = math.degrees(math.atan2(aim_y, aim_x))
            
            action_str = f"Move: ({a0:.2f}, {a1:.2f}) Aim: {aim_angle_deg:.0f}°"
            action_str2 = f"Shoot: {a4:.2f} ({'Y' if a4 > 0.5 else 'N'})  Dash: {a5:.2f} ({'Y' if a5 > 0.5 else 'N'})"
            texts.append(action_str)
            texts.append(action_str2)
        elif action is not None and len(action) >= 5:
            # Old action format: [move_x, move_y, aim_angle, shoot, dash]
            a0 = float(action[0]) if hasattr(action[0], '__float__') else action[0]
            a1 = float(action[1]) if hasattr(action[1], '__float__') else action[1]
            a2 = float(action[2]) if hasattr(action[2], '__float__') else action[2]
            a3 = float(action[3]) if hasattr(action[3], '__float__') else action[3]
            a4 = float(action[4]) if hasattr(action[4], '__float__') else action[4]
            
            action_str = f"Move: ({a0:.2f}, {a1:.2f}) Aim: {math.degrees(a2):.0f}°"
            action_str2 = f"Shoot: {a3:.2f} ({'Y' if a3 > 0.5 else 'N'})  Dash: {a4:.2f} ({'Y' if a4 > 0.5 else 'N'})"
            texts.append(action_str)
            texts.append(action_str2)
        
        for i, text in enumerate(texts):
            text_surface = self.small_font.render(text, True, COLORS['text'])
            self.surface.blit(text_surface, (10, y_offset + i * 18))
        
        # Controls hint
        hint = "SPACE: Pause  S: Step mode  N: Next step  ESC: Quit"
        hint_surface = self.small_font.render(hint, True, (100, 100, 100))
        self.surface.blit(hint_surface, (10, self.height - 20))
    
    def _draw_pause_overlay(self):
        """Draw pause indicator"""
        text = "PAUSED" if not self.step_mode else "STEP MODE (N = next)"
        text_surface = self.font.render(text, True, COLORS['text'])
        text_rect = text_surface.get_rect(center=(self.display_width // 2, self.display_height // 2))
        
        # Dark overlay
        overlay = pygame.Surface((self.display_width, self.display_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surface, text_rect)
    
    def close(self):
        """Clean up pygame"""
        pygame.quit()


def watch_training(
    model_path: Optional[str] = None,
    level: str = 'simple',
    n_episodes: int = 10,
    fps: int = 60,
    scale: float = 1.0,
    deterministic: bool = False,  # Default to stochastic to match training
):
    """
    Watch a trained model play, or watch random actions.
    
    Args:
        model_path: Path to trained model (None for random)
        level: Level to play
        n_episodes: Number of episodes
        fps: Target FPS
        scale: Display scale
        deterministic: Use deterministic actions (False = stochastic like training)
    """
    from env import NoMissMayhemEnv
    from levels import get_level
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import os
    
    # Load model if provided
    model = None
    vec_normalize = None
    if model_path:
        from stable_baselines3 import PPO, SAC, DQN
        # Try loading as different model types
        try:
            model = PPO.load(model_path)
            print(f"Loaded PPO model from {model_path}")
        except:
            try:
                model = SAC.load(model_path)
                print(f"Loaded SAC model from {model_path}")
            except:
                model = DQN.load(model_path)
                print(f"Loaded DQN model from {model_path}")
        
        # Try to load VecNormalize stats
        vecnorm_path = model_path.replace('.zip', '_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            print(f"Loading normalization stats from {vecnorm_path}")
        else:
            print(f"⚠️  Warning: No normalization stats found at {vecnorm_path}")
            print(f"   Model may perform poorly without correct observation normalization!")
        
        print(f"Mode: {'deterministic' if deterministic else 'stochastic (like training)'}")
    else:
        print("No model provided - using random actions")
    
    # Create environment
    level_data, start_pos = get_level(level)
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    # Wrap in VecEnv and load normalization if available
    if model_path:
        vecnorm_path = model_path.replace('.zip', '_vecnorm.pkl')
        if os.path.exists(vecnorm_path):
            # Wrap in DummyVecEnv first
            vec_env = DummyVecEnv([lambda: env])
            # Load VecNormalize with saved stats
            env = VecNormalize.load(vecnorm_path, vec_env)
            env.training = False  # Don't update stats during evaluation
            env.norm_reward = False  # Don't normalize rewards for display
            print("✓ Using normalized observations (as in training)")
    
    # Create renderer
    renderer = GameRenderer(fps=fps, scale=scale)
    
    try:
        for episode in range(n_episodes):
            obs = env.reset()
            # VecEnv returns just obs, not (obs, info)
            if isinstance(obs, tuple):
                obs = obs[0]
            
            done = False
            step = 0
            total_reward = 0
            
            print(f"\n=== Episode {episode + 1} ===")
            
            while not done and renderer.running:
                # Get action
                if model:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    # Add Gaussian noise to aim_x and aim_y to smooth out quantized aiming
                    # Indices 2 and 3 are aim_x and aim_y
                    action = action.copy()
                    if len(action) >= 4:
                        noise = np.random.randn(2) * 0.15  # Small noise
                        action[2] = np.clip(action[2] + noise[0], -1, 1)
                        action[3] = np.clip(action[3] + noise[1], -1, 1)
                else:
                    action = env.action_space.sample()
                
                # Flatten action if needed for display (save original for env.step)
                action_for_display = action.flatten() if hasattr(action, 'flatten') else action
                
                # Step environment - VecEnv handles action shape internally
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # VecEnv returns 4 values
                    obs, reward, done, info = step_result
                    done = done[0] if hasattr(done, '__len__') else done
                
                # Handle array rewards from VecEnv
                if hasattr(reward, '__len__'):
                    reward = float(reward[0])
                
                total_reward += reward
                step += 1
                
                # Get the actual env for rendering (unwrap VecNormalize if present)
                render_env = env.venv.envs[0] if hasattr(env, 'venv') else env
                
                # Render with flattened action for display
                if not renderer.render(render_env, action_for_display, episode + 1, step, reward, total_reward):
                    break
                
                # Handle pause
                while renderer.paused and not renderer.should_step and renderer.running:
                    if not renderer.handle_events():
                        break
                    renderer._draw_pause_overlay()
                    pygame.display.flip()
                    renderer.clock.tick(30)
            
            if not renderer.running:
                break
            
            # Episode summary
            # Extract info from array if needed
            if hasattr(info, '__len__') and not isinstance(info, dict):
                info = info[0] if len(info) > 0 else {}
            
            result = "WIN!" if info.get('win') else f"LOSS ({info.get('death_cause', 'timeout')})"
            print(f"Result: {result}, Steps: {step}, Reward: {total_reward:.2f}")
    
    finally:
        renderer.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch RL agent play')
    parser.add_argument('--model', type=str, default=None, help='Path to model (None for random)')
    parser.add_argument('--level', type=str, default='simple', help='Level to play')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS')
    parser.add_argument('--scale', type=float, default=1.0, help='Display scale')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions (default now)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions (mean policy)')
    
    args = parser.parse_args()
    
    # Default to stochastic unless explicitly set deterministic
    use_deterministic = args.deterministic and not args.stochastic
    
    watch_training(
        model_path=args.model,
        level=args.level,
        n_episodes=args.episodes,
        fps=args.fps,
        scale=args.scale,
        deterministic=use_deterministic,
    )
