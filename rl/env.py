"""
No Miss Mayhem - Python RL Environment
Exact replica of the JavaScript game physics
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import copy
import math

# =============================================================================
# CONSTANTS (exactly matching constants.js)
# =============================================================================

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 600

PLAYER_RADIUS = 20
PLAYER_SPEED = 1.0  # Acceleration per frame
PLAYER_MAX_SPEED = 5.0
PLAYER_FRICTION = 0.96
PLAYER_INVULNERABLE_TIME = 500  # ms
PLAYER_DASH_COOLDOWN = 2000  # ms
PLAYER_DASH_DISTANCE = 100  # Not actually used in JS (dash just sets isDashing)
PLAYER_DASH_DURATION = 200  # ms
PLAYER_MAX_HEALTH = 80
PLAYER_DAMAGE_PER_HIT = 10
PLAYER_SHOOT_COOLDOWN = 300  # ms

PROJECTILE_SPEED = 6.0
PROJECTILE_MAX_BOUNCES = 10
PROJECTILE_RADIUS = 5

ENEMY_PROJECTILE_RADIUS = 10

COIN_RADIUS = 8
KEY_RADIUS = 20
HEALTH_RADIUS = 10

# Door zones (from Rooms.js checkDoorCollision)
DOOR_X_MIN = 258
DOOR_X_MAX = 342
DOOR_Y_MIN = 258
DOOR_Y_MAX = 342
DOOR_EDGE_THRESHOLD = 22

# Player transition zones (from Game.js checkRooms)
PLAYER_DOOR_X_MIN = 250
PLAYER_DOOR_X_MAX = 350
PLAYER_DOOR_Y_MIN = 250
PLAYER_DOOR_Y_MAX = 350
PLAYER_EDGE_UP = 25
PLAYER_EDGE_DOWN = 575
PLAYER_EDGE_LEFT = 25
PLAYER_EDGE_RIGHT = 575

# Shop card hitboxes (from Store.js)
CARD_LEFT_X_MIN = 148
CARD_LEFT_X_MAX = 222
CARD_RIGHT_X_MIN = 348
CARD_RIGHT_X_MAX = 492
CARD_Y_MIN = 33
CARD_Y_MAX = 237

# Frame time (60 FPS)
FRAME_TIME_MS = 1000 / 60  # ~16.67ms


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_circle_collision(x1: float, y1: float, r1: float, 
                           x2: float, y2: float, r2: float) -> bool:
    """Check collision between two circles (matches utils.js checkCollision)"""
    dx = x1 - x2
    dy = y1 - y2
    distance = math.sqrt(dx * dx + dy * dy)
    return distance < r1 + r2


def check_laser_collision(player_x: float, player_y: float, player_radius: float,
                          laser_x: float, laser_y: float, laser_angle: float) -> bool:
    """Check if player intersects laser beam (matches utils.js checkLaserCollision)"""
    # Laser direction
    dir_x = math.cos(laser_angle)
    dir_y = math.sin(laser_angle)
    
    # Vector from laser origin to player
    to_player_x = player_x - laser_x
    to_player_y = player_y - laser_y
    
    # Projection length
    projection = to_player_x * dir_x + to_player_y * dir_y
    
    if projection < 0:
        return False  # Player is behind laser
    
    # Closest point on laser line
    closest_x = laser_x + projection * dir_x
    closest_y = laser_y + projection * dir_y
    
    # Distance from player to closest point
    dist = math.sqrt((closest_x - player_x) ** 2 + (closest_y - player_y) ** 2)
    
    # JS uses hardcoded threshold of 4 (not player radius!)
    return dist < 4


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# =============================================================================
# DATA CLASSES
# =============================================================================

class Player:
    def __init__(self):
        self.reset()
    
    def reset(self, x: float = CANVAS_WIDTH / 2, y: float = CANVAS_HEIGHT / 2):
        self.x = x
        self.y = y
        self.radius = PLAYER_RADIUS
        self.dx = 0.0
        self.dy = 0.0
        self.health = PLAYER_MAX_HEALTH
        self.max_health = PLAYER_MAX_HEALTH
        self.is_invulnerable = False
        self.invulnerable_timer = 0.0
        self.can_dash = True
        self.is_dashing = False
        self.dash_timer = 0.0
        self.dash_cooldown_timer = 0.0
        self.shoot_cooldown = PLAYER_SHOOT_COOLDOWN
        self.last_shot_timer = PLAYER_SHOOT_COOLDOWN  # Can shoot immediately
        self.money = 0
        self.keys: List[str] = []
        self.double = False
        self.spread = False
    
    def update(self, move_x: float, move_y: float, aim_angle: float, 
               do_dash: bool, dt_ms: float) -> None:
        """Update player physics (matches Player.js update method)"""
        
        # Normalize diagonal movement
        if move_x != 0 and move_y != 0:
            length = math.sqrt(move_x * move_x + move_y * move_y)
            move_x /= length
            move_y /= length
        
        # Apply acceleration
        self.dx += move_x * PLAYER_SPEED
        self.dy += move_y * PLAYER_SPEED
        
        # Handle dash
        if do_dash and self.can_dash:
            self.can_dash = False
            self.is_dashing = True
            self.dash_timer = PLAYER_DASH_DURATION
            self.dash_cooldown_timer = PLAYER_DASH_COOLDOWN
        
        # Cap speed (unless dashing)
        current_speed = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        if current_speed > PLAYER_MAX_SPEED and not self.is_dashing:
            ratio = PLAYER_MAX_SPEED / current_speed
            self.dx *= ratio
            self.dy *= ratio
        
        # Apply friction
        self.dx *= PLAYER_FRICTION
        self.dy *= PLAYER_FRICTION
        
        # Update position
        self.x += self.dx
        self.y += self.dy
        
        # Clamp to canvas bounds
        self.x = max(self.radius, min(CANVAS_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(CANVAS_HEIGHT - self.radius, self.y))
        
        # Zero out tiny velocities (matches JS: Math.abs < 0.01)
        if abs(self.dx) < 0.01:
            self.dx = 0
        if abs(self.dy) < 0.01:
            self.dy = 0
        
        # Update timers
        self.last_shot_timer += dt_ms
        
        if self.is_invulnerable:
            self.invulnerable_timer -= dt_ms
            if self.invulnerable_timer <= 0:
                self.is_invulnerable = False
        
        if self.is_dashing:
            self.dash_timer -= dt_ms
            if self.dash_timer <= 0:
                self.is_dashing = False
        
        if not self.can_dash:
            self.dash_cooldown_timer -= dt_ms
            if self.dash_cooldown_timer <= 0:
                self.can_dash = True
    
    def take_damage(self) -> bool:
        """Handle player taking damage. Returns True if player died."""
        if self.is_invulnerable:
            return False
        
        self.health -= PLAYER_DAMAGE_PER_HIT
        
        if self.health <= 0:
            return True  # Dead
        
        self.is_invulnerable = True
        self.invulnerable_timer = PLAYER_INVULNERABLE_TIME
        return False
    
    def can_shoot(self) -> bool:
        return self.last_shot_timer >= self.shoot_cooldown
    
    def shoot(self) -> None:
        self.last_shot_timer = 0
    
    def add_powerup(self, powerup: str) -> None:
        """Add powerup (matches Player.js addPowerup)"""
        if powerup == "extraballs":
            self.shoot_cooldown = 150
        elif powerup == "extrahealth":
            self.max_health += 10
            self.health = self.max_health
        elif powerup == "super":
            self.shoot_cooldown = 0
        elif powerup == "double":
            self.double = True
        elif powerup == "spread":
            self.spread = True


class Projectile:
    def __init__(self, x: float, y: float, angle: float, 
                 spawn_offset: float, radius: float, is_enemy: bool = False):
        """
        Create projectile (matches Projectile.js constructor)
        spawn_offset: distance from origin to spawn (playerRadius in JS)
        """
        self.x = x + spawn_offset * math.cos(angle)
        self.y = y + spawn_offset * math.sin(angle)
        self.dx = PROJECTILE_SPEED * math.cos(angle)
        self.dy = PROJECTILE_SPEED * math.sin(angle)
        self.radius = radius
        self.bounces = 0
        self.is_enemy = is_enemy
        self.can_hurt = False
        self.age_ms = 0.0
    
    def update(self, dt_ms: float) -> bool:
        """Update projectile. Returns True if should be removed."""
        self.age_ms += dt_ms
        
        # Can hurt after 100ms (matches JS)
        if not self.can_hurt and self.age_ms > 100:
            self.can_hurt = True
        
        # Move
        self.x += self.dx
        self.y += self.dy
        
        # Bounce off walls (matches Projectile.js update)
        if self.x - self.radius < 0:
            self.x = self.radius
            self.dx = abs(self.dx)
            self.bounces += 1
        if self.x + self.radius > CANVAS_WIDTH:
            self.x = CANVAS_WIDTH - self.radius
            self.dx = -abs(self.dx)
            self.bounces += 1
        if self.y - self.radius < 0:
            self.y = self.radius
            self.dy = abs(self.dy)
            self.bounces += 1
        if self.y + self.radius > CANVAS_HEIGHT:
            self.y = CANVAS_HEIGHT - self.radius
            self.dy = -abs(self.dy)
            self.bounces += 1
        
        return self.bounces >= PROJECTILE_MAX_BOUNCES


class Enemy:
    def __init__(self, enemy_type: str, x: float, y: float, 
                 enemy_id: str = "", has_key: bool = False, 
                 healing: bool = False, radius: float = 20, 
                 health: float = 100):
        self.type = enemy_type
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = 1.7
        self.health = health
        self.max_health = health
        self.is_active = True
        self.has_key = has_key
        self.healing = healing
        self.id = enemy_id
        
        # Attack timing
        self.attack_cooldown = 1900 + np.random.random() * 200  # ms
        self.last_attack = 0.0
        
        # Type-specific properties (matches Enemy.js)
        if enemy_type == 'regular':
            self.min_distance = self.radius * 1.5
            self.coin_type = 'bronze'
            self.coin_value = 1
        elif enemy_type == 'attacker':
            self.min_distance = self.radius * 12
            self.coin_type = 'silver'
            self.coin_value = 2
        elif enemy_type == 'shielded':
            self.min_distance = self.radius * 10
            self.coin_type = 'gold'
            self.coin_value = 5
            self.health = 60
            self.max_health = 60
            self.shield_angle = 0.0
            self.shield_arc = math.pi / 4  # 45 degrees (not 90 as comment says)
        elif enemy_type == 'laser':
            self.min_distance = self.radius * 12
            self.coin_type = 'gold'
            self.coin_value = 5
            self.health = 150
            self.max_health = 150
            self.laser_cooldown = 2500  # ms
            self.laser_charge_time = 300  # ms
            self.laser_duration = 2000  # ms
            self.fired = False
    
    def update(self, player: Player, game_time_ms: float, dt_ms: float) -> Optional[Any]:
        """
        Update enemy. Returns projectile/laser if attacking, else None.
        Matches Enemy.js update methods.
        """
        if not self.is_active:
            return None
        
        # Calculate distance and angle to player
        dx = player.x - self.x
        dy = player.y - self.y
        distance = math.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx)
        
        # Movement (all types use same base movement)
        if distance > self.min_distance:
            self.x += math.cos(angle) * self.speed
            self.y += math.sin(angle) * self.speed
        elif distance < self.min_distance - 5:
            self.x -= math.cos(angle) * self.speed
            self.y -= math.sin(angle) * self.speed
        
        # Clamp to bounds
        self.x = max(self.radius, min(CANVAS_WIDTH - self.radius, self.x))
        self.y = max(self.radius, min(CANVAS_HEIGHT - self.radius, self.y))
        
        # Type-specific behavior
        if self.type == 'regular':
            return None  # No ranged attack
        
        elif self.type == 'attacker':
            if game_time_ms - self.last_attack >= self.attack_cooldown:
                self.last_attack = game_time_ms
                return self._create_projectile(player)
        
        elif self.type == 'shielded':
            # Update shield to face player
            self.shield_angle = angle
            
            if game_time_ms - self.last_attack >= self.attack_cooldown:
                self.last_attack = game_time_ms
                return self._create_projectile(player)
        
        elif self.type == 'laser':
            if not self.fired:
                self.fired = True
                self.last_attack = game_time_ms - self.laser_cooldown
            
            if game_time_ms - self.last_attack >= self.laser_cooldown:
                self.last_attack = game_time_ms
                return {
                    'type': 'laser',
                    'x': self.x,
                    'y': self.y,
                    'angle': angle,
                    'fire_time': game_time_ms,
                    'remain_time': self.laser_duration,
                    'delay': self.laser_charge_time,
                }
        
        return None
    
    def _create_projectile(self, player: Player) -> Dict:
        """Create enemy projectile"""
        angle = math.atan2(player.y - self.y, player.x - self.x)
        return {
            'type': 'projectile',
            'x': self.x,
            'y': self.y,
            'angle': angle,
        }
    
    def check_shield_block(self, proj_x: float, proj_y: float) -> bool:
        """Check if projectile is blocked by shield (shielded enemies only)"""
        if self.type != 'shielded':
            return False
        
        angle = math.atan2(proj_y - self.y, proj_x - self.x)
        angle_diff = abs(normalize_angle(angle - self.shield_angle))
        
        return angle_diff <= self.shield_arc / 2
    
    def take_damage(self, damage: float) -> bool:
        """Take damage. Returns True if died."""
        self.health -= damage
        if self.health <= 0:
            self.is_active = False
            return True
        return False


class Laser:
    def __init__(self, x: float, y: float, angle: float, 
                 fire_time: float, remain_time: float, delay: float):
        self.x = x
        self.y = y
        self.angle = angle
        self.fire_time = fire_time
        self.remain_time = remain_time
        self.delay = delay
    
    def is_active(self, game_time_ms: float) -> bool:
        """Check if laser is currently damaging"""
        elapsed = game_time_ms - self.fire_time
        return elapsed >= self.delay
    
    def is_expired(self, game_time_ms: float) -> bool:
        """Check if laser should be removed"""
        elapsed = game_time_ms - self.fire_time
        return elapsed >= self.remain_time


class Collectible:
    def __init__(self, ctype: str, x: float, y: float, value: int = 0, cid: str = ""):
        self.type = ctype  # 'coin_bronze', 'coin_silver', 'coin_gold', 'key', 'health'
        self.x = x
        self.y = y
        self.value = value
        self.id = cid
        
        if 'coin' in ctype:
            self.radius = COIN_RADIUS
        elif ctype == 'key':
            self.radius = KEY_RADIUS
        elif ctype == 'health':
            self.radius = HEALTH_RADIUS
        else:
            self.radius = 10


class Door:
    def __init__(self, door_type: str, openreq: int):
        """
        door_type: 'wall', 'door', 'key'
        openreq: shots required to open (0 = already open for 'door' type)
        """
        self.type = door_type
        self.openreq = openreq
        self.shotcount = 0
        
        # Determine initial open state (matches createTravel in Rooms.js)
        if door_type == 'wall':
            self.open = False
        elif door_type == 'key':
            self.open = False  # Key doors start closed
        elif door_type == 'door':
            self.open = (openreq == 0)
        else:
            self.open = False
    
    def can_pass(self, has_key: bool) -> bool:
        """Check if player can pass through this door"""
        if self.type == 'wall':
            return False
        if self.open:
            return True
        if self.type == 'key' and has_key:
            return True
        if self.type == 'door' and self.shotcount >= self.openreq:
            return True
        return False
    
    def get_progress(self) -> float:
        """Get door unlock progress (0-1)"""
        if self.type == 'wall':
            return 0.0
        if self.open:
            return 1.0
        if self.openreq == 0:
            return 1.0
        return min(self.shotcount / self.openreq, 1.0)


class Room:
    def __init__(self, room_type: str, travel: Dict[str, Tuple[str, int]],
                 enemies: List[Dict], powerups: List[Tuple] = None):
        """
        room_type: 'regular', 'shop', 'win', 'nullTile'
        travel: {'up': ('door', 3), 'down': ('wall', 0), ...}
        enemies: [{'type': 'regular', 'x': 100, 'y': 200, ...}, ...]
        powerups: [('double', 5, 'Double Barrel', 'desc'), ...]
        """
        self.type = room_type
        self.doors = {
            'up': Door(travel['up'][0], travel['up'][1]),
            'down': Door(travel['down'][0], travel['down'][1]),
            'left': Door(travel['left'][0], travel['left'][1]),
            'right': Door(travel['right'][0], travel['right'][1]),
        }
        self.enemy_templates = enemies
        self.powerups = powerups or []
        self.bought = [False, False]
        self.visited = False
        
        # Runtime state (populated on room entry)
        self.enemies: List[Enemy] = []
        self.projectiles: List[Projectile] = []
        self.lasers: List[Laser] = []
        self.collectibles: List[Collectible] = []
    
    def reset_enemies(self):
        """Spawn enemies from templates"""
        self.enemies = []
        for e in self.enemy_templates:
            enemy = Enemy(
                enemy_type=e['type'],
                x=e['x'],
                y=e['y'],
                enemy_id=e.get('id', ''),
                has_key=e.get('has_key', False),
                healing=e.get('healing', False),
                radius=e.get('radius', 20),
                health=e.get('health', 100),
            )
            self.enemies.append(enemy)
        
        self.projectiles = []
        self.lasers = []
        self.collectibles = []


# =============================================================================
# MAIN ENVIRONMENT
# =============================================================================

class NoMissMayhemEnv(gym.Env):
    """
    Gymnasium environment for No Miss Mayhem.
    Exactly replicates JavaScript game physics.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    # Observation space sizes
    MAX_ENEMIES = 8
    MAX_PROJECTILES = 30
    MAX_LASERS = 4
    MAX_COLLECTIBLES = 10
    
    def __init__(self, level_data: List[List[Dict]] = None, 
                 start_pos: Tuple[int, int] = None,
                 render_mode: str = None,
                 max_steps: int = 10000):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Level data (will be set by set_level or use default)
        self.level_data = level_data
        self.start_pos = start_pos
        
        # Action space: [move_x, move_y, aim_x, aim_y, shoot, dash]
        # move_x, move_y: -1 to 1 (movement direction)
        # aim_x, aim_y: -1 to 1 (aiming direction, converted to angle)
        # shoot, dash: 0 to 1 (threshold at 0.5)
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space (calculated below)
        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Game state
        self.player: Optional[Player] = None
        self.rooms: Optional[List[List[Room]]] = None
        self.room_pos: List[int] = [0, 0]
        self.game_time_ms: float = 0.0
        self.step_count: int = 0
        self.total_reward: float = 0.0
    
    def _calculate_obs_size(self) -> int:
        """Calculate observation vector size"""
        size = 0
        size += 12  # Player state
        size += 4 * 3  # 4 doors × 3 values
        size += self.MAX_ENEMIES * 8  # enemies
        size += self.MAX_PROJECTILES * 5  # projectiles
        size += self.MAX_LASERS * 4  # lasers
        size += self.MAX_COLLECTIBLES * 4  # collectibles
        size += 4  # room info
        return size
    
    def set_level(self, level_data: List[List[Dict]], start_pos: Tuple[int, int]):
        """Set level data (call before reset)"""
        self.level_data = level_data
        self.start_pos = start_pos
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        if self.level_data is None:
            raise ValueError("Level data not set. Call set_level() first or pass level_data to constructor.")
        
        # Reset player
        self.player = Player()
        
        # Deep copy level data to create rooms
        self.rooms = []
        for row in self.level_data:
            room_row = []
            for room_data in row:
                room = Room(
                    room_type=room_data['type'],
                    travel=room_data['travel'],
                    enemies=copy.deepcopy(room_data.get('enemies', [])),
                    powerups=room_data.get('powerups', []),
                )
                room_row.append(room)
            self.rooms.append(room_row)
        
        # Set starting position
        self.room_pos = list(self.start_pos)
        
        # Initialize current room
        current_room = self._get_current_room()
        current_room.visited = True
        current_room.reset_enemies()
        
        # Reset timers
        self.game_time_ms = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        
        return self._get_obs(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one game step (1/60 second).
        
        action: [move_x, move_y, aim_x, aim_y, shoot, dash]
        """
        move_x = float(np.clip(action[0], -1, 1))
        move_y = float(np.clip(action[1], -1, 1))
        
        # Convert aim_x, aim_y to angle (more NN-friendly than raw angle)
        aim_x = float(np.clip(action[2], -1, 1))
        aim_y = float(np.clip(action[3], -1, 1))
        aim_angle = math.atan2(aim_y, aim_x)  # Returns [-pi, pi]
        if aim_angle < 0:
            aim_angle += 2 * math.pi  # Convert to [0, 2pi]
        
        do_shoot = float(action[4]) > 0.5
        do_dash = float(action[5]) > 0.5
        
        dt_ms = FRAME_TIME_MS
        self.game_time_ms += dt_ms
        self.step_count += 1
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        room = self._get_current_room()
        prev_health = self.player.health
        prev_money = self.player.money
        prev_enemy_count = len([e for e in room.enemies if e.is_active])
        prev_projectile_count = len(room.projectiles)
        
        # Track shooting accuracy
        shot_this_step = False
        active_enemies = [e for e in room.enemies if e.is_active]
        
        # 1. Update player
        self.player.update(move_x, move_y, aim_angle, do_dash, dt_ms)
        
        # 2. Handle shooting
        if do_shoot and self.player.can_shoot():
            self.player.shoot()
            shot_this_step = True
            
            # SHOOTING ACCURACY PENALTY/REWARD
            if len(active_enemies) == 0:
                # Shooting when no enemies = wasteful
                reward -= 0.5
            else:
                # Check if aiming toward nearest enemy
                nearest_enemy = min(active_enemies,
                                  key=lambda e: math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2))
                angle_to_enemy = math.atan2(nearest_enemy.y - self.player.y, 
                                           nearest_enemy.x - self.player.x)
                # Normalize angle difference to [-pi, pi]
                angle_diff = ((aim_angle - angle_to_enemy + math.pi) % (2 * math.pi)) - math.pi
                aim_accuracy = 1.0 - min(abs(angle_diff) / math.pi, 1.0)  # 1.0 = perfect, 0.0 = opposite
                
                # Scale reward linearly with aim accuracy to encourage precision
                reward += aim_accuracy * 0.5  # Up to +0.5 for perfect aim
                
                if aim_accuracy < 0.3:  # Very poor aim
                    reward -= 0.3
            
            # Main projectile
            proj = Projectile(
                self.player.x, self.player.y, aim_angle,
                spawn_offset=35, radius=PROJECTILE_RADIUS, is_enemy=False
            )
            room.projectiles.append(proj)
            
            # Double powerup (shoots backward)
            if self.player.double:
                proj2 = Projectile(
                    self.player.x, self.player.y, aim_angle + math.pi,
                    spawn_offset=35, radius=PROJECTILE_RADIUS, is_enemy=False
                )
                room.projectiles.append(proj2)
            
            # Spread powerup (shoots ±10 degrees)
            if self.player.spread:
                proj3 = Projectile(
                    self.player.x, self.player.y, aim_angle + math.pi/18,
                    spawn_offset=35, radius=PROJECTILE_RADIUS, is_enemy=False
                )
                proj4 = Projectile(
                    self.player.x, self.player.y, aim_angle - math.pi/18,
                    spawn_offset=35, radius=PROJECTILE_RADIUS, is_enemy=False
                )
                room.projectiles.append(proj3)
                room.projectiles.append(proj4)
        
        # 3. Update enemies
        for enemy in room.enemies:
            if not enemy.is_active:
                continue
            
            result = enemy.update(self.player, self.game_time_ms, dt_ms)
            
            if result:
                if result['type'] == 'projectile':
                    proj = Projectile(
                        result['x'], result['y'], result['angle'],
                        spawn_offset=enemy.radius, radius=ENEMY_PROJECTILE_RADIUS,
                        is_enemy=True
                    )
                    room.projectiles.append(proj)
                elif result['type'] == 'laser':
                    laser = Laser(
                        result['x'], result['y'], result['angle'],
                        result['fire_time'], result['remain_time'], result['delay']
                    )
                    room.lasers.append(laser)
            
            # Enemy-player collision
            if check_circle_collision(
                enemy.x, enemy.y, enemy.radius,
                self.player.x, self.player.y, self.player.radius
            ):
                if self.player.take_damage():
                    terminated = True
                    reward -= 100
                    info['death_cause'] = 'enemy_collision'
        
        # 4. Update and check lasers
        lasers_to_remove = []
        for i, laser in enumerate(room.lasers):
            if laser.is_expired(self.game_time_ms):
                lasers_to_remove.append(i)
            elif laser.is_active(self.game_time_ms):
                if check_laser_collision(
                    self.player.x, self.player.y, self.player.radius,
                    laser.x, laser.y, laser.angle
                ):
                    if self.player.take_damage():
                        terminated = True
                        reward -= 100
                        info['death_cause'] = 'laser'
        
        for i in reversed(lasers_to_remove):
            room.lasers.pop(i)
        
        # 5. Update projectiles
        projs_to_remove = []
        for i, proj in enumerate(room.projectiles):
            should_remove = proj.update(dt_ms)
            
            if should_remove:
                projs_to_remove.append(i)
                continue
            
            # Check player collision
            if check_circle_collision(
                proj.x, proj.y, proj.radius,
                self.player.x, self.player.y, self.player.radius
            ):
                projs_to_remove.append(i)
                if self.player.take_damage():
                    terminated = True
                    reward -= 100
                    info['death_cause'] = 'projectile'
                continue
            
            # Check enemy collision (player projectiles only)
            if not proj.is_enemy:
                for enemy in room.enemies:
                    if not enemy.is_active:
                        continue
                    
                    # Shielded enemy - check shield first
                    if enemy.type == 'shielded':
                        # Expand hitbox for shield check
                        if check_circle_collision(
                            proj.x, proj.y, proj.radius,
                            enemy.x, enemy.y, enemy.radius + 5
                        ):
                            if enemy.check_shield_block(proj.x, proj.y):
                                # Reflect projectile
                                collision_angle = math.atan2(
                                    proj.y - enemy.y, proj.x - enemy.x
                                )
                                reflection_angle = 2 * enemy.shield_angle - collision_angle
                                proj.dx = PROJECTILE_SPEED * math.cos(reflection_angle)
                                proj.dy = PROJECTILE_SPEED * math.sin(reflection_angle)
                                # Move away to prevent re-collision
                                proj.x = enemy.x + (enemy.radius + proj.radius + 1) * math.cos(collision_angle)
                                proj.y = enemy.y + (enemy.radius + proj.radius + 1) * math.sin(collision_angle)
                                proj.bounces += 1
                                continue
                    
                    # Normal collision
                    if check_circle_collision(
                        proj.x, proj.y, proj.radius,
                        enemy.x, enemy.y, enemy.radius
                    ):
                        if i not in projs_to_remove:
                            projs_to_remove.append(i)
                        
                        # REWARD FOR HITTING ENEMY
                        reward += 2.0  # Good shot!
                        
                        if enemy.take_damage(20):
                            # Enemy died - spawn drops
                            room.collectibles.append(Collectible(
                                f'coin_{enemy.coin_type}', enemy.x, enemy.y, enemy.coin_value
                            ))
                            if enemy.has_key:
                                room.collectibles.append(Collectible(
                                    'key', enemy.x, enemy.y, 0, enemy.id
                                ))
                            if enemy.healing:
                                room.collectibles.append(Collectible(
                                    'health', enemy.x, enemy.y
                                ))
                        break
            
            # Check door collision (player projectiles only)
            if not proj.is_enemy and i not in projs_to_remove:
                door_hit = self._check_door_collision(proj.x, proj.y)
                if door_hit:
                    door = room.doors[door_hit]
                    if not door.open and door.type == 'door':
                        prev_progress = door.get_progress()
                        door.shotcount += 1
                        new_progress = door.get_progress()
                        projs_to_remove.append(i)
                        
                        # Reward for door progress - scales with progress
                        reward += 2.0  # Base reward for hitting door
                        
                        # Bonus if door just opened
                        if new_progress >= 1.0 and prev_progress < 1.0:
                            door.open = True
                            reward += 15  # Big bonus for opening door!
            
            # Check shop card collision
            if not proj.is_enemy and room.type == 'shop' and i not in projs_to_remove:
                card_hit = self._check_card_collision(proj.x, proj.y)
                if card_hit is not None and not room.bought[card_hit]:
                    if len(room.powerups) > card_hit:
                        powerup = room.powerups[card_hit]
                        cost = powerup[1]
                        if self.player.money >= cost:
                            self.player.money -= cost
                            self.player.add_powerup(powerup[0])
                            room.bought[card_hit] = True
                            projs_to_remove.append(i)
                            reward += 10  # Reward for buying powerup
        
        # Remove projectiles (in reverse order)
        for i in sorted(set(projs_to_remove), reverse=True):
            if i < len(room.projectiles):
                room.projectiles.pop(i)
        
        # 6. Collect items
        collectibles_to_remove = []
        for i, item in enumerate(room.collectibles):
            if check_circle_collision(
                self.player.x, self.player.y, self.player.radius,
                item.x, item.y, item.radius
            ):
                collectibles_to_remove.append(i)
                if 'coin' in item.type:
                    self.player.money += item.value
                    reward += item.value * 0.2  # Small coin reward
                elif item.type == 'key':
                    self.player.keys.append(item.id if item.id else 'key1')
                    reward += 50  # Increased from 20 - keys are important!
                elif item.type == 'health':
                    self.player.health = self.player.max_health
                    reward += 10  # Increased from 5
        
        for i in reversed(collectibles_to_remove):
            room.collectibles.pop(i)
        
        # 7. Check room transition
        transition = self._check_room_transition()
        if transition:
            new_room = self._get_current_room()
            if new_room.type == 'win':
                terminated = True
                # Time bonus: faster = more reward
                time_bonus = max(0, 1000 - self.game_time_ms / 100)
                reward += 1000 + time_bonus  # Increased from 500
                info['win'] = True
            else:
                if not new_room.visited:
                    new_room.visited = True
                    new_room.reset_enemies()
                    reward += 30  # Increased from 15 - reward for exploring
        
        # 8. Calculate rewards
        # DENSE REWARD SHAPING for better learning
        
        # Base survival (very small)
        reward += 0.001
        
        # Damage penalty (scaled)
        damage_taken = prev_health - self.player.health
        if damage_taken > 0:
            reward -= damage_taken * 2  # Increased penalty
        
        # Enemy kill reward (big!)
        current_enemy_count = len([e for e in room.enemies if e.is_active])
        enemies_killed = prev_enemy_count - current_enemy_count
        if enemies_killed > 0:
            reward += enemies_killed * 10  # Increased from 5
        
        # DENSE SHAPING: Reward approaching/aiming at enemies
        if len(room.enemies) > 0:
            active_enemies = [e for e in room.enemies if e.is_active]
            if active_enemies:
                # Find closest enemy
                closest_enemy = min(active_enemies, 
                                  key=lambda e: math.sqrt((e.x - self.player.x)**2 + (e.y - self.player.y)**2))
                dist_to_enemy = math.sqrt((closest_enemy.x - self.player.x)**2 + (closest_enemy.y - self.player.y)**2)
                
                # Reward being at good distance (not too close, not too far)
                ideal_distance = 150  # Sweet spot for combat
                distance_reward = -abs(dist_to_enemy - ideal_distance) / 1000  # Small shaping
                reward += distance_reward
        else:
            # No enemies - focus on doors!
            door_positions = {
                'up': (300, 10),
                'down': (300, 590),
                'left': (10, 300),
                'right': (590, 300),
            }
            
            # Find doors that need attention (closed/unlocking)
            doors_need_shooting = []
            doors_ready_to_enter = []
            
            for direction, door in room.doors.items():
                if door.type == 'wall':
                    continue
                if door.open:
                    doors_ready_to_enter.append(direction)
                elif door.type == 'door':
                    # Door needs shooting
                    doors_need_shooting.append(direction)
                elif door.type == 'key' and 'key1' in self.player.keys:
                    # Has key, can enter
                    doors_ready_to_enter.append(direction)
            
            # Priority 1: Shoot closed doors
            if doors_need_shooting:
                closest_locked = min(doors_need_shooting,
                                   key=lambda d: math.sqrt((door_positions[d][0] - self.player.x)**2 +
                                                          (door_positions[d][1] - self.player.y)**2))
                door_pos = door_positions[closest_locked]
                dist_to_door = math.sqrt((door_pos[0] - self.player.x)**2 + (door_pos[1] - self.player.y)**2)
                
                # Reward approaching closed door (stronger than before)
                approach_reward = (400 - dist_to_door) / 80  # Up to +5 for being close
                reward += approach_reward
                
                # Extra reward if aiming at the door
                angle_to_door = math.atan2(door_pos[1] - self.player.y, door_pos[0] - self.player.x)
                angle_diff = abs(((aim_angle - angle_to_door + math.pi) % (2 * math.pi)) - math.pi)
                if angle_diff < 0.5:  # Aiming roughly at door
                    reward += 0.5
            
            # Priority 2: Enter open/unlocked doors
            elif doors_ready_to_enter:
                closest_door = min(doors_ready_to_enter,
                                 key=lambda d: math.sqrt((door_positions[d][0] - self.player.x)**2 +
                                                        (door_positions[d][1] - self.player.y)**2))
                door_pos = door_positions[closest_door]
                dist_to_door = math.sqrt((door_pos[0] - self.player.x)**2 + (door_pos[1] - self.player.y)**2)
                
                # VERY strong reward for approaching ready door
                reward += (300 - dist_to_door) / 50  # Up to +6 for being at door
        
        # Coin reward
        coins_gained = self.player.money - prev_money
        reward += coins_gained * 0.5
        
        # Penalty for camping in corner (anti-exploit)
        corner_dist = min(
            self.player.x + self.player.y,  # Top-left
            (CANVAS_WIDTH - self.player.x) + self.player.y,  # Top-right
            self.player.x + (CANVAS_HEIGHT - self.player.y),  # Bottom-left
            (CANVAS_WIDTH - self.player.x) + (CANVAS_HEIGHT - self.player.y)  # Bottom-right
        )
        if corner_dist < 100 and len([e for e in room.enemies if e.is_active]) > 0:
            reward -= 0.1  # Penalty for corner camping with enemies alive
        
        # Check truncation
        if self.step_count >= self.max_steps:
            truncated = True
        
        self.total_reward += reward
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_current_room(self) -> Room:
        return self.rooms[self.room_pos[0]][self.room_pos[1]]
    
    def _check_door_collision(self, x: float, y: float) -> Optional[str]:
        """Check if projectile hit a door zone (matches Rooms.js checkDoorCollision)"""
        if DOOR_X_MIN < x < DOOR_X_MAX:
            if y < DOOR_EDGE_THRESHOLD:
                return 'up'
            if y > CANVAS_HEIGHT - DOOR_EDGE_THRESHOLD:
                return 'down'
        if DOOR_Y_MIN < y < DOOR_Y_MAX:
            if x < DOOR_EDGE_THRESHOLD:
                return 'left'
            if x > CANVAS_WIDTH - DOOR_EDGE_THRESHOLD:
                return 'right'
        return None
    
    def _check_card_collision(self, x: float, y: float) -> Optional[int]:
        """Check if projectile hit a shop card (matches Store.js)"""
        if CARD_Y_MIN < y < CARD_Y_MAX:
            if CARD_LEFT_X_MIN < x < CARD_LEFT_X_MAX:
                return 0
            if CARD_RIGHT_X_MIN < x < CARD_RIGHT_X_MAX:
                return 1
        return None
    
    def _check_room_transition(self) -> bool:
        """Check and handle room transitions (matches Game.js checkRooms)"""
        room = self._get_current_room()
        has_key = 'key1' in self.player.keys
        
        transitioned = False
        
        # Up
        if (self.player.y < PLAYER_EDGE_UP and 
            PLAYER_DOOR_X_MIN < self.player.x < PLAYER_DOOR_X_MAX):
            door = room.doors['up']
            if door.can_pass(has_key):
                if door.type == 'key' and has_key and not door.open:
                    self.player.keys.remove('key1')
                door.open = True
                self.room_pos[0] -= 1
                self.player.y = 560
                self.player.x = 300
                self._get_current_room().doors['down'].open = True
                transitioned = True
        
        # Down
        elif (self.player.y > PLAYER_EDGE_DOWN and 
              PLAYER_DOOR_X_MIN < self.player.x < PLAYER_DOOR_X_MAX):
            door = room.doors['down']
            if door.can_pass(has_key):
                if door.type == 'key' and has_key and not door.open:
                    self.player.keys.remove('key1')
                door.open = True
                self.room_pos[0] += 1
                self.player.y = 40
                self.player.x = 300
                self._get_current_room().doors['up'].open = True
                transitioned = True
        
        # Right
        elif (self.player.x > PLAYER_EDGE_RIGHT and 
              PLAYER_DOOR_Y_MIN < self.player.y < PLAYER_DOOR_Y_MAX):
            door = room.doors['right']
            if door.can_pass(has_key):
                if door.type == 'key' and has_key and not door.open:
                    self.player.keys.remove('key1')
                door.open = True
                self.room_pos[1] += 1
                self.player.x = 40
                self.player.y = 300
                self._get_current_room().doors['left'].open = True
                transitioned = True
        
        # Left
        elif (self.player.x < PLAYER_EDGE_LEFT and 
              PLAYER_DOOR_Y_MIN < self.player.y < PLAYER_DOOR_Y_MAX):
            door = room.doors['left']
            if door.can_pass(has_key):
                if door.type == 'key' and has_key and not door.open:
                    self.player.keys.remove('key1')
                door.open = True
                self.room_pos[1] -= 1
                self.player.x = 560
                self.player.y = 300
                self._get_current_room().doors['right'].open = True
                transitioned = True
        
        return transitioned
    
    def _get_obs(self) -> np.ndarray:
        """Build observation vector"""
        room = self._get_current_room()
        has_key = 'key1' in self.player.keys
        
        obs = []
        
        # Player state (12 values)
        obs.extend([
            self.player.x / CANVAS_WIDTH,
            self.player.y / CANVAS_HEIGHT,
            self.player.dx / PLAYER_MAX_SPEED,
            self.player.dy / PLAYER_MAX_SPEED,
            self.player.health / self.player.max_health,
            float(self.player.is_invulnerable),
            float(self.player.can_dash),
            float(self.player.is_dashing),
            min(self.player.money / 20, 1.0),
            float(has_key),
            float(self.player.double),
            float(self.player.spread),
        ])
        
        # Doors (4 × 3 = 12 values)
        for direction in ['up', 'down', 'left', 'right']:
            door = room.doors[direction]
            type_val = {'wall': 0, 'door': 0.5, 'key': 1.0}.get(door.type, 0)
            obs.extend([
                type_val,
                float(door.open or door.can_pass(has_key)),
                door.get_progress(),
            ])
        
        # Enemies (MAX_ENEMIES × 8)
        active_enemies = [e for e in room.enemies if e.is_active]
        for i in range(self.MAX_ENEMIES):
            if i < len(active_enemies):
                e = active_enemies[i]
                type_onehot = [0, 0, 0, 0]
                type_idx = ['regular', 'attacker', 'shielded', 'laser'].index(e.type)
                type_onehot[type_idx] = 1
                obs.extend([
                    e.x / CANVAS_WIDTH,
                    e.y / CANVAS_HEIGHT,
                    *type_onehot,
                    e.health / e.max_health,
                    e.radius / 40,
                ])
            else:
                obs.extend([0, 0, 0, 0, 0, 0, 0, 0])
        
        # Projectiles (MAX_PROJECTILES × 5)
        for i in range(self.MAX_PROJECTILES):
            if i < len(room.projectiles):
                p = room.projectiles[i]
                obs.extend([
                    p.x / CANVAS_WIDTH,
                    p.y / CANVAS_HEIGHT,
                    p.dx / PROJECTILE_SPEED,
                    p.dy / PROJECTILE_SPEED,
                    float(p.is_enemy),
                ])
            else:
                obs.extend([0, 0, 0, 0, 0])
        
        # Lasers (MAX_LASERS × 4)
        active_lasers = [l for l in room.lasers if not l.is_expired(self.game_time_ms)]
        for i in range(self.MAX_LASERS):
            if i < len(active_lasers):
                l = active_lasers[i]
                obs.extend([
                    l.x / CANVAS_WIDTH,
                    l.y / CANVAS_HEIGHT,
                    l.angle / (2 * math.pi),
                    float(l.is_active(self.game_time_ms)),
                ])
            else:
                obs.extend([0, 0, 0, 0])
        
        # Collectibles (MAX_COLLECTIBLES × 4)
        for i in range(self.MAX_COLLECTIBLES):
            if i < len(room.collectibles):
                c = room.collectibles[i]
                type_val = {
                    'coin_bronze': 0.2, 'coin_silver': 0.4, 'coin_gold': 0.6,
                    'key': 0.8, 'health': 1.0
                }.get(c.type, 0)
                obs.extend([
                    c.x / CANVAS_WIDTH,
                    c.y / CANVAS_HEIGHT,
                    type_val,
                    c.value / 5,
                ])
            else:
                obs.extend([0, 0, 0, 0])
        
        # Room info (4 values)
        obs.extend([
            self.room_pos[0] / 10,
            self.room_pos[1] / 10,
            float(room.type == 'shop'),
            self.game_time_ms / 60000,  # Normalized time (1 min)
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def render(self):
        """Render environment (for debugging)"""
        if self.render_mode == 'human':
            # Could implement pygame rendering here
            pass
        return None
    
    def close(self):
        pass


# =============================================================================
# DISCRETE ACTION WRAPPER (optional, for DQN)
# =============================================================================

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Convert continuous actions to discrete for DQN training.
    """
    
    # 9 movement directions × 8 aim directions × 2 shoot × 2 dash = 288 actions
    MOVE_DIRS = [
        (0, 0),    # Stay
        (0, -1),   # Up
        (0, 1),    # Down
        (-1, 0),   # Left
        (1, 0),    # Right
        (-1, -1),  # Up-Left
        (1, -1),   # Up-Right
        (-1, 1),   # Down-Left
        (1, 1),    # Down-Right
    ]
    
    AIM_ANGLES = [i * math.pi / 4 for i in range(8)]  # 8 directions
    
    def __init__(self, env):
        super().__init__(env)
        n_moves = len(self.MOVE_DIRS)
        n_aims = len(self.AIM_ANGLES)
        self.action_space = spaces.Discrete(n_moves * n_aims * 2 * 2)
    
    def action(self, action: int) -> np.ndarray:
        n_moves = len(self.MOVE_DIRS)
        n_aims = len(self.AIM_ANGLES)
        
        dash = action % 2
        action //= 2
        shoot = action % 2
        action //= 2
        aim_idx = action % n_aims
        action //= n_aims
        move_idx = action % n_moves
        
        move_x, move_y = self.MOVE_DIRS[move_idx]
        aim_angle = self.AIM_ANGLES[aim_idx]
        
        return np.array([move_x, move_y, aim_angle, shoot, dash], dtype=np.float32)


if __name__ == "__main__":
    # Quick test
    from levels import TUTORIAL_LEVEL, TUTORIAL_START
    
    env = NoMissMayhemEnv()
    env.set_level(TUTORIAL_LEVEL, TUTORIAL_START)
    
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Random actions test
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            print(f"Episode ended at step {i}: reward={reward}, info={info}")
            break
    
    print("Environment test passed!")
