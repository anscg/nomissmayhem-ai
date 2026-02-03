"""
Export trained model to ONNX format for use in JavaScript/browser.
Also provides a JS inference template.
"""

import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import NoMissMayhemEnv
from levels import get_level


def export_to_onnx(
    model_path: str,
    output_path: str = 'model.onnx',
    level: str = 'simple',
    opset_version: int = 11,
):
    """
    Export SB3 model to ONNX format.
    
    Args:
        model_path: Path to saved .zip model
        output_path: Output ONNX file path
        level: Level name (for observation shape)
        opset_version: ONNX opset version
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Get observation shape
    level_data, start_pos = get_level(level)
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    obs, _ = env.reset()
    obs_shape = obs.shape
    print(f"Observation shape: {obs_shape}")
    
    # Check for normalization stats
    norm_path = os.path.join(os.path.dirname(model_path), 'vec_normalize.pkl')
    norm_stats = None
    if os.path.exists(norm_path):
        print(f"Loading normalization stats from {norm_path}...")
        dummy_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(norm_path, dummy_env)
        norm_stats = {
            'obs_mean': vec_env.obs_rms.mean.tolist(),
            'obs_var': vec_env.obs_rms.var.tolist(),
            'clip_obs': vec_env.clip_obs,
        }
        print("Normalization stats loaded!")
    
    # Extract the policy network
    policy = model.policy
    policy.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, *obs_shape)
    
    # Export to ONNX
    print(f"Exporting to {output_path}...")
    
    # We need to export just the actor (policy) part for inference
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
        
        def forward(self, obs):
            # Get deterministic action
            features = self.policy.extract_features(obs)
            if self.policy.share_features_extractor:
                latent_pi = self.policy.mlp_extractor.forward_actor(features)
            else:
                pi_features = self.policy.pi_features_extractor(obs)
                latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
            
            mean_actions = self.policy.action_net(latent_pi)
            return mean_actions
    
    wrapped_policy = PolicyWrapper(policy)
    wrapped_policy.eval()
    
    torch.onnx.export(
        wrapped_policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # Save normalization stats as JSON
    if norm_stats:
        import json
        norm_output = output_path.replace('.onnx', '_normalization.json')
        with open(norm_output, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Normalization stats saved to {norm_output}")
    
    # Generate JS inference code
    js_output = output_path.replace('.onnx', '_inference.js')
    generate_js_inference(js_output, obs_shape[0], norm_stats)
    print(f"JS inference code generated at {js_output}")
    
    return output_path


def generate_js_inference(output_path: str, obs_size: int, norm_stats: dict = None):
    """Generate JavaScript code for running inference"""
    
    norm_code = ""
    if norm_stats:
        norm_code = f"""
// Normalization statistics from training
const OBS_MEAN = {norm_stats['obs_mean']};
const OBS_VAR = {norm_stats['obs_var']};
const CLIP_OBS = {norm_stats['clip_obs']};

function normalizeObs(obs) {{
    const normalized = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) {{
        const std = Math.sqrt(OBS_VAR[i] + 1e-8);
        normalized[i] = Math.max(-CLIP_OBS, Math.min(CLIP_OBS, (obs[i] - OBS_MEAN[i]) / std));
    }}
    return normalized;
}}
"""
    else:
        norm_code = """
// No normalization needed (model trained without VecNormalize)
function normalizeObs(obs) {
    return obs;
}
"""

    js_code = f'''/**
 * No Miss Mayhem - AI Agent Inference
 * 
 * This file provides the interface to run the trained RL model in the browser.
 * Requires ONNX Runtime Web: https://www.npmjs.com/package/onnxruntime-web
 * 
 * Usage:
 *   npm install onnxruntime-web
 *   
 *   import {{ AIAgent }} from './model_inference.js';
 *   const agent = new AIAgent();
 *   await agent.load('model.onnx');
 *   const action = await agent.getAction(gameState);
 */

import * as ort from 'onnxruntime-web';

// Constants matching Python environment
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 600;
const PLAYER_MAX_SPEED = 5.0;
const PROJECTILE_SPEED = 6.0;
const MAX_ENEMIES = 8;
const MAX_PROJECTILES = 30;
const MAX_LASERS = 4;
const MAX_COLLECTIBLES = 10;
const OBS_SIZE = {obs_size};
{norm_code}

/**
 * Convert game state to observation vector
 * This MUST match the _get_obs() method in env.py EXACTLY
 */
function gameStateToObs(game) {{
    const player = game.player;
    const room = game.getCurrentRoom();
    const hasKey = player.keys.includes('key1');
    
    const obs = new Float32Array(OBS_SIZE);
    let idx = 0;
    
    // Player state (12 values)
    obs[idx++] = player.x / CANVAS_WIDTH;
    obs[idx++] = player.y / CANVAS_HEIGHT;
    obs[idx++] = player.dx / PLAYER_MAX_SPEED;
    obs[idx++] = player.dy / PLAYER_MAX_SPEED;
    obs[idx++] = player.health / player.healthBar.maxHealth;
    obs[idx++] = player.isInvulnerable ? 1 : 0;
    obs[idx++] = player.canDash ? 1 : 0;
    obs[idx++] = player.isDashing ? 1 : 0;
    obs[idx++] = Math.min(player.money / 20, 1.0);
    obs[idx++] = hasKey ? 1 : 0;
    obs[idx++] = player.double ? 1 : 0;
    obs[idx++] = player.spread ? 1 : 0;
    
    // Doors (4 × 3 = 12 values)
    const directions = ['up', 'down', 'left', 'right'];
    for (const dir of directions) {{
        const door = room.travel[dir];
        const typeMap = {{ 'wall': 0, 'door': 0.5, 'key': 1.0 }};
        const typeVal = typeMap[door.type] || 0;
        const isOpen = door.open || (door.type === 'door' && door.shotcount >= door.openreq);
        const progress = door.openreq > 0 ? Math.min(door.shotcount / door.openreq, 1.0) : 1.0;
        
        obs[idx++] = typeVal;
        obs[idx++] = isOpen ? 1 : 0;
        obs[idx++] = progress;
    }}
    
    // Enemies (MAX_ENEMIES × 8)
    const activeEnemies = room.enemies.filter(e => e.isActive);
    for (let i = 0; i < MAX_ENEMIES; i++) {{
        if (i < activeEnemies.length) {{
            const e = activeEnemies[i];
            obs[idx++] = e.x / CANVAS_WIDTH;
            obs[idx++] = e.y / CANVAS_HEIGHT;
            
            // Type one-hot: [regular, attacker, shielded, laser]
            const types = ['regular', 'attacker', 'shielded', 'laser'];
            for (const t of types) {{
                obs[idx++] = e.type === t ? 1 : 0;
            }}
            
            obs[idx++] = e.health / e.maxHealth;
            obs[idx++] = e.radius / 40;
        }} else {{
            // Padding
            for (let j = 0; j < 8; j++) obs[idx++] = 0;
        }}
    }}
    
    // Projectiles (MAX_PROJECTILES × 5)
    const projs = room.projectiles || [];
    for (let i = 0; i < MAX_PROJECTILES; i++) {{
        if (i < projs.length) {{
            const p = projs[i];
            obs[idx++] = p.x / CANVAS_WIDTH;
            obs[idx++] = p.y / CANVAS_HEIGHT;
            obs[idx++] = p.dx / PROJECTILE_SPEED;
            obs[idx++] = p.dy / PROJECTILE_SPEED;
            obs[idx++] = p.isEnemyProjectile ? 1 : 0;
        }} else {{
            for (let j = 0; j < 5; j++) obs[idx++] = 0;
        }}
    }}
    
    // Lasers (MAX_LASERS × 4)
    const lasers = room.lasers || [];
    const gameTime = Date.now();
    for (let i = 0; i < MAX_LASERS; i++) {{
        if (i < lasers.length) {{
            const l = lasers[i];
            const elapsed = gameTime - l.fireTime;
            obs[idx++] = l.x / CANVAS_WIDTH;
            obs[idx++] = l.y / CANVAS_HEIGHT;
            obs[idx++] = l.angle / (2 * Math.PI);
            obs[idx++] = elapsed >= l.delay ? 1 : 0;
        }} else {{
            for (let j = 0; j < 4; j++) obs[idx++] = 0;
        }}
    }}
    
    // Collectibles (MAX_COLLECTIBLES × 4)
    const collectibles = [
        ...(room.coins || []).map(c => ({{ ...c, ctype: 'coin_' + c.type }})),
        ...(room.keys || []).map(k => ({{ ...k, ctype: 'key', value: 0 }})),
        ...(room.health || []).map(h => ({{ ...h, ctype: 'health', value: 0 }})),
    ];
    for (let i = 0; i < MAX_COLLECTIBLES; i++) {{
        if (i < collectibles.length) {{
            const c = collectibles[i];
            const typeMap = {{
                'coin_bronze': 0.2, 'coin_silver': 0.4, 'coin_gold': 0.6,
                'key': 0.8, 'health': 1.0
            }};
            obs[idx++] = c.x / CANVAS_WIDTH;
            obs[idx++] = c.y / CANVAS_HEIGHT;
            obs[idx++] = typeMap[c.ctype] || 0;
            obs[idx++] = (c.value || 0) / 5;
        }} else {{
            for (let j = 0; j < 4; j++) obs[idx++] = 0;
        }}
    }}
    
    // Room info (4 values)
    obs[idx++] = game.roomPosition[0] / 10;
    obs[idx++] = game.roomPosition[1] / 10;
    obs[idx++] = room.type === 'shop' ? 1 : 0;
    obs[idx++] = game.elapsedTime / 60000;
    
    return obs;
}}

/**
 * AI Agent class for running inference
 */
export class AIAgent {{
    constructor() {{
        this.session = null;
        this.isLoaded = false;
    }}
    
    /**
     * Load the ONNX model
     * @param {{string}} modelPath - Path to the .onnx file
     */
    async load(modelPath) {{
        try {{
            this.session = await ort.InferenceSession.create(modelPath);
            this.isLoaded = true;
            console.log('AI Agent loaded successfully!');
        }} catch (error) {{
            console.error('Failed to load AI model:', error);
            throw error;
        }}
    }}
    
    /**
     * Get action from the model
     * @param {{object}} game - The Game instance
     * @returns {{object}} Action object with keys, mouse position, and dash
     */
    async getAction(game) {{
        if (!this.isLoaded) {{
            throw new Error('Model not loaded. Call load() first.');
        }}
        
        // Convert game state to observation
        const obsRaw = gameStateToObs(game);
        const obs = normalizeObs(obsRaw);
        
        // Create tensor
        const inputTensor = new ort.Tensor('float32', obs, [1, OBS_SIZE]);
        
        // Run inference
        const results = await this.session.run({{ observation: inputTensor }});
        const actionData = results.action.data;
        
        // Parse action: [move_x, move_y, aim_angle, shoot, dash]
        const moveX = Math.max(-1, Math.min(1, actionData[0]));
        const moveY = Math.max(-1, Math.min(1, actionData[1]));
        const aimAngle = actionData[2] % (2 * Math.PI);
        const shoot = actionData[3] > 0.5;
        const dash = actionData[4] > 0.5;
        
        // Convert to game inputs
        return {{
            keys: {{
                w: moveY < -0.3,
                s: moveY > 0.3,
                a: moveX < -0.3,
                d: moveX > 0.3,
                shift: dash,
            }},
            mouseX: game.player.x + Math.cos(aimAngle) * 100,
            mouseY: game.player.y + Math.sin(aimAngle) * 100,
            isMouseDown: shoot,
        }};
    }}
}}

/**
 * Integration helper - call this in your game loop
 */
export async function runAIStep(game, agent) {{
    const action = await agent.getAction(game);
    
    // Apply to game state
    game.keys = action.keys;
    game.mouseX = action.mouseX;
    game.mouseY = action.mouseY;
    game.isMouseDown = action.isMouseDown;
}}

// Export for use
export {{ gameStateToObs, normalizeObs }};
'''
    
    with open(output_path, 'w') as f:
        f.write(js_code)


def verify_export(onnx_path: str, model_path: str, level: str = 'simple', n_tests: int = 10):
    """Verify ONNX export produces same outputs as original model"""
    import onnxruntime as ort
    
    print(f"\nVerifying ONNX export...")
    
    # Load original model
    model = PPO.load(model_path)
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)
    
    # Create environment
    level_data, start_pos = get_level(level)
    env = NoMissMayhemEnv(level_data=level_data, start_pos=start_pos)
    
    max_diff = 0
    for i in range(n_tests):
        obs, _ = env.reset()
        
        # Get PyTorch output
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            torch_action, _, _ = model.policy(obs_tensor, deterministic=True)
            torch_action = torch_action.numpy()[0]
        
        # Get ONNX output
        ort_inputs = {'observation': obs.reshape(1, -1).astype(np.float32)}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_action = ort_outputs[0][0]
        
        diff = np.abs(torch_action - onnx_action).max()
        max_diff = max(max_diff, diff)
        
        if diff > 0.01:
            print(f"  Test {i}: diff={diff:.6f} (may need attention)")
    
    print(f"Max difference: {max_diff:.6f}")
    if max_diff < 0.01:
        print("✓ ONNX export verified successfully!")
    else:
        print("⚠ Large differences detected, check export")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('model_path', type=str, help='Path to trained model (.zip)')
    parser.add_argument('--output', type=str, default='model.onnx', help='Output ONNX path')
    parser.add_argument('--level', type=str, default='simple', help='Level name for obs shape')
    parser.add_argument('--verify', action='store_true', help='Verify export')
    
    args = parser.parse_args()
    
    onnx_path = export_to_onnx(args.model_path, args.output, args.level)
    
    if args.verify:
        verify_export(onnx_path, args.model_path, args.level)
