"""
Level data for No Miss Mayhem RL environment.
Exactly matches the JavaScript level definitions.
"""

import random
import copy
import math
from typing import Optional

# =============================================================================
# TUTORIAL LEVEL (from tutorial.js)
# =============================================================================

def _null_tile():
    return {
        'type': 'nullTile',
        'travel': {
            'up': ('wall', 0),
            'down': ('wall', 0),
            'left': ('wall', 0),
            'right': ('wall', 0),
        },
        'enemies': [],
    }

TUTORIAL_LEVEL = [
    # Row 0
    [
        _null_tile(),
        _null_tile(),
        _null_tile(),
        _null_tile(),
        # t40 - Starting room
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 0),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [],
        },
    ],
    # Row 1
    [
        _null_tile(),
        _null_tile(),
        # b21 - Boss room with key
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 0),
                'left': ('wall', 4),
                'right': ('key', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 300, 'y': 300, 'id': 'key1', 'has_key': True, 'healing': False, 'radius': 40, 'health': 160},
                {'type': 'attacker', 'x': 400, 'y': 400},
            ],
        },
        # Win room
        {
            'type': 'win',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [],
        },
        # t41
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 0),
                'down': ('door', 3),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 100},
            ],
        },
    ],
    # Row 2
    [
        # t02
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 6),
                'left': ('wall', 0),
                'right': ('door', 4),
            },
            'enemies': [
                {'type': 'attacker', 'x': 250, 'y': 500, 'healing': True},
                {'type': 'attacker', 'x': 400, 'y': 400},
            ],
        },
        # t12
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 6),
                'left': ('door', 4),
                'right': ('door', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 300, 'y': 300},
                {'type': 'attacker', 'x': 250, 'y': 500, 'healing': True},
            ],
        },
        # t22
        {
            'type': 'regular',
            'travel': {
                'up': ('key', 0),
                'down': ('wall', 0),
                'left': ('door', 4),
                'right': ('door', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 100},
                {'type': 'regular', 'x': 400, 'y': 400},
            ],
        },
        # s32 - Shop
        {
            'type': 'shop',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('door', 0),
                'right': ('door', 0),
            },
            'enemies': [],
            'powerups': [
                ('extraballs', 2, 'Extra Balls', 'Increases fire rate!'),
                ('extrahealth', 1, 'Extra Health', 'Increases max health!'),
            ],
        },
        # t42
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 0),
                'down': ('wall', 0),
                'left': ('door', 4),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 100},
                {'type': 'attacker', 'x': 400, 'y': 400},
            ],
        },
    ],
    # Row 3
    [
        # t03
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('door', 4),
            },
            'enemies': [
                {'type': 'regular', 'x': 300, 'y': 400},
                {'type': 'regular', 'x': 500, 'y': 200},
                {'type': 'regular', 'x': 400, 'y': 100, 'id': 'key1', 'has_key': True, 'healing': True},
            ],
        },
        # t13
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('wall', 0),
                'left': ('door', 5),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 300, 'y': 300},
                {'type': 'attacker', 'x': 400, 'y': 400},
                {'type': 'attacker', 'x': 200, 'y': 100},
            ],
        },
        _null_tile(),
        _null_tile(),
        _null_tile(),
    ],
]

TUTORIAL_START = (0, 4)  # Row 0, Column 4


# =============================================================================
# LEVEL 1 (from levelone.js)
# =============================================================================

LEVEL1_DATA = [
    # Row 0
    [
        # d00
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 6),
                'left': ('wall', 0),
                'right': ('door', 7),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 100},
                {'type': 'attacker', 'x': 400, 'y': 400},
                {'type': 'shielded', 'x': 150, 'y': 500, 'id': 'key1', 'has_key': True, 'healing': True},
            ],
        },
        # d10
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 5),
                'left': ('door', 6),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'shielded', 'x': 150, 'y': 400},
                {'type': 'regular', 'x': 100, 'y': 300},
                {'type': 'regular', 'x': 200, 'y': 500, 'healing': True},
            ],
        },
        _null_tile(),
        _null_tile(),
        _null_tile(),
    ],
    # Row 1
    [
        # d01
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 6),
                'down': ('door', 6),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 100},
                {'type': 'attacker', 'x': 100, 'y': 400, 'healing': True},
                {'type': 'attacker', 'x': 300, 'y': 500, 'healing': True},
            ],
        },
        # d11
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 6),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('door', 6),
            },
            'enemies': [
                {'type': 'attacker', 'x': 100, 'y': 500},
                {'type': 'attacker', 'x': 140, 'y': 100},
                {'type': 'attacker', 'x': 250, 'y': 560},
            ],
        },
        # d21
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 6),
                'left': ('door', 4),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'shielded', 'x': 150, 'y': 240},
                {'type': 'attacker', 'x': 200, 'y': 450, 'healing': True},
                {'type': 'attacker', 'x': 160, 'y': 550},
            ],
        },
        _null_tile(),
        # Win room
        {
            'type': 'win',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [],
        },
    ],
    # Row 2
    [
        # d02
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 6),
                'down': ('door', 6),
                'left': ('wall', 0),
                'right': ('door', 7),
            },
            'enemies': [
                {'type': 'shielded', 'x': 150, 'y': 140},
                {'type': 'regular', 'x': 200, 'y': 400},
                {'type': 'attacker', 'x': 100, 'y': 500},
            ],
        },
        # s12 - Shop
        {
            'type': 'shop',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('door', 0),
                'right': ('door', 0),
            },
            'enemies': [],
            'powerups': [
                ('double', 5, 'Double Barrel', 'Second gun!'),
                ('extrahealth', 3, 'Extra Health', 'More health!'),
            ],
        },
        # d22
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('door', 6),
                'left': ('door', 4),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'attacker', 'x': 200, 'y': 300, 'healing': True},
                {'type': 'attacker', 'x': 250, 'y': 550},
                {'type': 'attacker', 'x': 130, 'y': 550},
            ],
        },
        _null_tile(),
        # b42 - Boss room
        {
            'type': 'regular',
            'travel': {
                'up': ('key', 0),
                'down': ('door', 0),
                'left': ('wall', 4),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'shielded', 'x': 300, 'y': 300, 'id': 'key1', 'has_key': True, 'healing': False, 'radius': 40, 'health': 250},
                {'type': 'regular', 'x': 250, 'y': 500, 'healing': True},
                {'type': 'regular', 'x': 450, 'y': 400, 'healing': True},
                {'type': 'regular', 'x': 400, 'y': 100},
            ],
        },
    ],
    # Row 3
    [
        # d03
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 6),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('door', 7),
            },
            'enemies': [
                {'type': 'attacker', 'x': 100, 'y': 400},
                {'type': 'regular', 'x': 350, 'y': 500},
                {'type': 'regular', 'x': 100, 'y': 550, 'healing': True},
            ],
        },
        # d13
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('door', 4),
                'right': ('door', 6),
            },
            'enemies': [
                {'type': 'shielded', 'x': 150, 'y': 280},
                {'type': 'regular', 'x': 200, 'y': 500},
                {'type': 'regular', 'x': 150, 'y': 550},
            ],
        },
        # d23
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('door', 6),
                'left': ('door', 5),
                'right': ('key', 0),
            },
            'enemies': [
                {'type': 'shielded', 'x': 200, 'y': 300, 'healing': True},
            ],
        },
        # s33 - Shop
        {
            'type': 'shop',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('door', 0),
                'right': ('door', 0),
            },
            'enemies': [],
            'powerups': [
                ('super', 8, 'Super Shot', 'No cooldown!'),
                ('extrahealth', 5, 'Extra Health', 'More health!'),
            ],
        },
        # d43
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 6),
                'down': ('wall', 6),
                'left': ('door', 5),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'shielded', 'x': 200, 'y': 300, 'healing': True},
            ],
        },
    ],
    # Row 4
    [
        _null_tile(),
        _null_tile(),
        # d24
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 3),
                'down': ('wall', 6),
                'left': ('wall', 5),
                'right': ('wall', 0),
            },
            'enemies': [],
        },
        _null_tile(),
        _null_tile(),
    ],
]

LEVEL1_START = (4, 2)  # Row 4, Column 2


# =============================================================================
# LEVEL 2 (from leveltwo.js) - Has laser enemies
# =============================================================================

LEVEL2_DATA = [
    # Row 0
    [
        # d00 - Boss with key
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('key', 0),
                'left': ('wall', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 250, 'y': 300, 'id': 'key1', 'has_key': True, 'healing': True, 'radius': 40, 'health': 250},
                {'type': 'laser', 'x': 200, 'y': 500, 'healing': True},
                {'type': 'regular', 'x': 100, 'y': 300, 'healing': True},
            ],
        },
        # d10
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('wall', 0),
                'left': ('door', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'regular', 'x': 100, 'y': 500, 'healing': True},
                {'type': 'shielded', 'x': 200, 'y': 300},
            ],
        },
        # d20
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('wall', 0),
                'left': ('door', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'regular', 'x': 200, 'y': 350},
                {'type': 'regular', 'x': 250, 'y': 300},
                {'type': 'attacker', 'x': 400, 'y': 150},
            ],
        },
        # s30 - Shop
        {
            'type': 'shop',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('door', 0),
                'right': ('door', 0),
            },
            'enemies': [],
            'powerups': [
                ('double', 13, 'Double Barrel', 'Second gun!'),
                ('super', 15, 'Super Shot', 'No cooldown!'),
            ],
        },
        # d40
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('door', 6),
                'left': ('door', 5),
                'right': ('wall', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 100, 'y': 350, 'healing': True},
                {'type': 'laser', 'x': 210, 'y': 500},
            ],
        },
        _null_tile(),
    ],
    # Row 1
    [
        # d01
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 3),
                'down': ('door', 0),
                'left': ('wall', 5),
                'right': ('wall', 5),
            },
            'enemies': [],
        },
        # d11
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('wall', 6),
                'left': ('wall', 5),
                'right': ('door', 3),
            },
            'enemies': [],
        },
        # d21
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('wall', 6),
                'left': ('door', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 250, 'y': 100, 'healing': True},
            ],
        },
        # s31 - Shop
        {
            'type': 'shop',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('door', 0),
                'right': ('door', 1),
            },
            'enemies': [],
            'powerups': [
                ('extrahealth', 5, 'Extra Health', 'More health!'),
                ('spread', 5, 'Spread Shot', 'Triple gun!'),
            ],
        },
        # d41
        {
            'type': 'regular',
            'travel': {
                'up': ('key', 3),
                'down': ('door', 6),
                'left': ('door', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 200, 'y': 100},
                {'type': 'regular', 'x': 250, 'y': 300},
                {'type': 'regular', 'x': 500, 'y': 100, 'healing': True},
            ],
        },
        # d51
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 4),
                'down': ('door', 6),
                'left': ('door', 5),
                'right': ('wall', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 300, 'y': 500},
                {'type': 'laser', 'x': 350, 'y': 300},
                {'type': 'regular', 'x': 350, 'y': 100, 'healing': True},
            ],
        },
    ],
    # Row 2
    [
        # Win room
        {
            'type': 'win',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [],
        },
        _null_tile(),
        _null_tile(),
        # d32
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 3),
                'down': ('door', 6),
                'left': ('wall', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 300, 'y': 300},
                {'type': 'regular', 'x': 350, 'y': 100, 'healing': True},
                {'type': 'attacker', 'x': 100, 'y': 300},
            ],
        },
        # d42
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 3),
                'down': ('wall', 6),
                'left': ('door', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'shielded', 'x': 500, 'y': 300},
                {'type': 'regular', 'x': 300, 'y': 300},
                {'type': 'regular', 'x': 100, 'y': 300, 'healing': True},
            ],
        },
        # d52
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 4),
                'down': ('door', 6),
                'left': ('door', 5),
                'right': ('wall', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 300, 'y': 300},
                {'type': 'regular', 'x': 350, 'y': 100},
            ],
        },
    ],
    # Row 3
    [
        _null_tile(),
        _null_tile(),
        _null_tile(),
        # d33
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 3),
                'down': ('door', 6),
                'left': ('wall', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'shielded', 'x': 100, 'y': 300},
                {'type': 'regular', 'x': 350, 'y': 100},
                {'type': 'regular', 'x': 100, 'y': 350},
            ],
        },
        _null_tile(),
        # d53
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('door', 6),
                'left': ('wall', 5),
                'right': ('wall', 5),
            },
            'enemies': [
                {'type': 'shielded', 'x': 100, 'y': 300},
                {'type': 'regular', 'x': 350, 'y': 100},
                {'type': 'regular', 'x': 100, 'y': 350, 'healing': True},
            ],
        },
    ],
    # Row 4
    [
        _null_tile(),
        _null_tile(),
        _null_tile(),
        # d34
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('wall', 6),
                'left': ('wall', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'shielded', 'x': 550, 'y': 100},
                {'type': 'regular', 'x': 120, 'y': 350, 'healing': True},
            ],
        },
        # d44
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 5),
                'down': ('door', 6),
                'left': ('door', 5),
                'right': ('door', 5),
            },
            'enemies': [
                {'type': 'laser', 'x': 550, 'y': 100},
                {'type': 'laser', 'x': 120, 'y': 350},
                {'type': 'laser', 'x': 420, 'y': 350},
            ],
        },
        # d54
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('wall', 6),
                'left': ('door', 5),
                'right': ('wall', 5),
            },
            'enemies': [
                {'type': 'shielded', 'x': 450, 'y': 150},
                {'type': 'attacker', 'x': 250, 'y': 400},
            ],
        },
    ],
    # Row 5
    [
        _null_tile(),
        _null_tile(),
        _null_tile(),
        _null_tile(),
        # d45 - Final boss room
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 5),
                'down': ('wall', 6),
                'left': ('wall', 5),
                'right': ('wall', 5),
            },
            'enemies': [
                {'type': 'regular', 'x': 550, 'y': 100},
                {'type': 'regular', 'x': 120, 'y': 350},
                {'type': 'attacker', 'x': 120, 'y': 350},
                {'type': 'laser', 'x': 420, 'y': 350, 'id': 'key1', 'has_key': True, 'healing': True},
            ],
        },
        _null_tile(),
    ],
]

LEVEL2_START = (1, 1)  # Row 1, Column 1


# =============================================================================
# SIMPLE TRAINING LEVEL (easier for initial training)
# =============================================================================

SIMPLE_LEVEL = [
    # Row 0
    [
        # Start room - just one enemy
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('door', 2),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 400, 'y': 200},
            ],
        },
    ],
    # Row 1
    [
        # Second room - two enemies
        {
            'type': 'regular',
            'travel': {
                'up': ('door', 0),
                'down': ('door', 3),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 150},
                {'type': 'regular', 'x': 450, 'y': 450},
            ],
        },
    ],
    # Row 2
    [
        # Win room
        {
            'type': 'win',
            'travel': {
                'up': ('door', 0),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [],
        },
    ],
]

SIMPLE_START = (0, 0)


# =============================================================================
# SINGLE ROOM ARENA (for training combat)
# =============================================================================

ARENA_LEVEL = [
    [
        {
            'type': 'regular',
            'travel': {
                'up': ('wall', 0),
                'down': ('wall', 0),
                'left': ('wall', 0),
                'right': ('wall', 0),
            },
            'enemies': [
                {'type': 'regular', 'x': 150, 'y': 150},
                {'type': 'regular', 'x': 450, 'y': 150},
                {'type': 'attacker', 'x': 300, 'y': 450},
            ],
        },
    ],
]

ARENA_START = (0, 0)


# =============================================================================
# ALL LEVELS
# =============================================================================

ALL_LEVELS = {
    'tutorial': (TUTORIAL_LEVEL, TUTORIAL_START),
    'level1': (LEVEL1_DATA, LEVEL1_START),
    'level2': (LEVEL2_DATA, LEVEL2_START),
    'simple': (SIMPLE_LEVEL, SIMPLE_START),
    'arena': (ARENA_LEVEL, ARENA_START),
}


def randomize_level(level_data: list, seed: Optional[int] = None):
    """
    Randomize enemy positions and door configurations in a level.
    
    Args:
        level_data: Level template (list of rows)
        seed: Random seed for reproducibility
    
    Returns:
        Copy of level with randomized positions
    """
    if seed is not None:
        random.seed(seed)
    
    random_level = copy.deepcopy(level_data)
    
    # Safe spawn area bounds (avoid walls and player spawn)
    MIN_POS = 50
    MAX_POS = 550
    SAFE_RADIUS_FROM_PLAYER = 120  # Don't spawn too close to center
    
    for row in random_level:
        for tile in row:
            if tile.get('type') == 'nullTile':
                continue
                
            # Randomize enemy positions
            for enemy in tile.get('enemies', []):
                # Keep some enemies with key at fixed position (has_key=True)
                if enemy.get('has_key'):
                    continue
                
                # Generate random position
                for _ in range(10):  # Try up to 10 times to find good position
                    new_x = random.randint(MIN_POS, MAX_POS)
                    new_y = random.randint(MIN_POS, MAX_POS)
                    
                    # Check distance from player spawn (assume center at 300, 300)
                    dist_to_center = math.sqrt((new_x - 300)**2 + (new_y - 300)**2)
                    if dist_to_center >= SAFE_RADIUS_FROM_PLAYER:
                        enemy['x'] = new_x
                        enemy['y'] = new_y
                        break
            
            # Randomize door positions (slight offset, keep on correct edge)
            travel = tile.get('travel', {})
            for direction, (door_type, _) in travel.items():
                if door_type == 'wall':
                    continue
                    
                # Add small random offset to door center position
                # (this is visual only, collision detection uses fixed zones)
                # Note: actual door collision zones are fixed in env.py constants
                
    return random_level


def get_level(name: str, randomize: bool = False, seed: Optional[int] = None):
    """
    Get level data by name.
    
    Args:
        name: Level name
        randomize: Whether to randomize enemy/door positions
        seed: Random seed for reproducibility (only used if randomize=True)
    
    Returns:
        Tuple of (level_data, start_pos)
    """
    if name not in ALL_LEVELS:
        raise ValueError(f"Unknown level: {name}. Available: {list(ALL_LEVELS.keys())}")
    
    level_data, start_pos = ALL_LEVELS[name]
    
    if randomize:
        level_data = randomize_level(level_data, seed)
    
    return level_data, start_pos
