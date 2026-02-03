# No Miss Mayhem - RL Training

Train a reinforcement learning agent to play No Miss Mayhem!

## Setup

```bash
cd rl
pip install -r requirements.txt
```

## Training

### Quick Start (Simple Level)
```bash
python train.py --level simple --timesteps 500000
```

### Full Training (Tutorial Level)
```bash
python train.py --level tutorial --timesteps 2000000 --n_envs 16
```

### Training Options
```bash
python train.py --help

# Examples:
python train.py --level arena --timesteps 100000      # Combat training
python train.py --level level1 --timesteps 5000000    # Full level 1
python train.py --algorithm sac --level simple        # Use SAC instead of PPO
python train.py --resume ./models/checkpoint_500000   # Resume training
python train.py --randomize --ent_coef 0.05           # Randomize positions + higher entropy
```

## Curriculum Learning (Recommended)

For best results, train progressively:

```bash
# Stage 1: Learn basic combat (arena)
python train.py --level arena --timesteps 500000

# Stage 2: Learn navigation (simple)
python train.py --level simple --timesteps 1000000

# Stage 3: Full tutorial
python train.py --level tutorial --timesteps 3000000

# Stage 4: Level 1
python train.py --level level1 --timesteps 5000000
```

## Evaluation

```bash
python train.py --eval ./models/ppo_simple_*/best_model.zip --level simple
```

## Export to JavaScript

After training, export to ONNX for browser use:

```bash
python export_onnx.py ./models/ppo_simple_*/best_model.zip --output ../public/ai/model.onnx --verify
```

This creates:
- `model.onnx` - The neural network
- `model_normalization.json` - Observation normalization stats
- `model_inference.js` - JavaScript inference code

## Using in the Browser

1. Install ONNX Runtime Web in your project:
```bash
npm install onnxruntime-web
```

2. Copy the exported files to your `public/ai/` folder

3. In your Game.js:
```javascript
import { AIAgent, runAIStep } from './ai/model_inference.js';

// In Game class:
async initAI() {
    this.aiAgent = new AIAgent();
    await this.aiAgent.load('./ai/model.onnx');
    this.useAI = true;
}

// In update() or gameLoop():
if (this.useAI) {
    await runAIStep(this, this.aiAgent);
}
```

## Files

- `env.py` - Gymnasium environment (exact JS physics replica)
- `levels.py` - Level data matching JS definitions
- `train.py` - Training script with PPO/SAC/DQN
- `export_onnx.py` - Export to ONNX + generate JS code
- `requirements.txt` - Python dependencies

## Observation Space (326 values)

| Component | Size | Description |
|-----------|------|-------------|
| Player | 12 | x, y, dx, dy, health, invuln, dash, dashing, money, key, double, spread |
| Doors | 12 | 4 doors × (type, open, progress) |
| Enemies | 64 | 8 enemies × (x, y, type[4], health, radius) |
| Projectiles | 150 | 30 projectiles × (x, y, dx, dy, is_enemy) |
| Lasers | 16 | 4 lasers × (x, y, angle, active) |
| Collectibles | 40 | 10 items × (x, y, type, value) |
| Room | 4 | row, col, is_shop, time |

## Action Space (5 continuous values)

| Action | Range | Description |
|--------|-------|-------------|
| move_x | [-1, 1] | Left/right input |
| move_y | [-1, 1] | Up/down input |
| aim_angle | [0, 2π] | Aim direction |
| shoot | [0, 1] | Fire weapon (>0.5 = shoot) |
| dash | [0, 1] | Dash ability (>0.5 = dash) |

## Reward Function

- **+0.01** per frame survived
- **+5** per enemy killed
- **+0.5** per coin collected
- **+20** for collecting a key
- **+5** for health pickup
- **+0.5** for shooting a door (progress)
- **+10** for buying powerup
- **+15** for entering new room
- **+500-1000** for winning (time bonus)
- **-10** per damage taken
- **-100** for death

## Tips

1. **Start simple**: Train on `arena` first to learn combat
2. **Use many envs**: `--n_envs 16` or higher speeds training significantly
3. **Monitor tensorboard**: `tensorboard --logdir ./logs`
4. **Patience**: Complex levels need millions of steps
5. **Curriculum**: Progressive difficulty works best
6. **Randomize for generalization**: Use `--randomize` to randomize enemy positions, preventing overfitting to fixed spawn points
7. **Higher entropy**: Use `--ent_coef 0.05` for better exploration and more diverse aiming
