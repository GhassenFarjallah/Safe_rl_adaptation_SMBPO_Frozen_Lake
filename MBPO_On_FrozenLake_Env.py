# -*- coding: utf-8 -*-
"""
Created on Fri May  2 15:40:37 2025
@author: ghass
"""
import os, random, time
from collections import deque
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# Workaround for OpenMP conflict on some systems
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# =============================================================================
# 0. Replay Buffer Implementation
# =============================================================================
class ReplayBuffer:
    """
    Cyclic buffer to store and sample experience tuples.
    Each entry is a tuple: (state, action, next_state, reward, done_flag).
    """
    def __init__(self, capacity):
        # Use deque to automatically discard old experiences
        self.buffer = deque(maxlen=capacity)

    def append(self, *exp):
        """Add a new experience to buffer"""
        self.buffer.append(exp)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences.
        Returns separate arrays for states, actions, next_states, rewards, dones.
        """
        batch = random.sample(self.buffer, batch_size)
        return list(map(np.array, zip(*batch)))

    def __len__(self):
        """Current size of the buffer"""
        return len(self.buffer)

# =============================================================================
# 1. Custom Environment: Non-absorbing-hole FrozenLake
# =============================================================================
class NonAbsorbingHoleFrozenLake(FrozenLakeEnv):
    """
    Extends FrozenLake to keep holes non-terminating and inject custom transitions.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Dimensions and action/state counts
        self.nrow, self.ncol = self.desc.shape
        self.nS, self.nA = self.nrow * self.ncol, self.action_space.n
        # Keep original map for reference
        self.orig_desc = self.desc.copy()
        # Override transitions on hole tiles
        self._fix_hole_transitions()

    def _calc_transition(self, r, c, a):
        """
        Deterministic move: given row, col, and action index, compute next state.
        Returns next_state_index, reward, done_flag (only on goal).
        """
        # Map action to position delta
        if a == 0:  # left
            nc, nr = max(c-1, 0), r
        elif a == 1:  # down
            nr, nc = min(r+1, self.nrow-1), c
        elif a == 2:  # right
            nc, nr = min(c+1, self.ncol-1), r
        else:  # up
            nr, nc = max(r-1, 0), c
        ns = nr * self.ncol + nc
        tile = self.desc[nr, nc]
        # Reward only when reaching goal tile
        return ns, float(tile == b'G'), (tile == b'G')

    def _calc_slip(self, r, c, a):
        """
        Slippery dynamics: 80% intended action, 10% perpendicular each side.
        Returns list of tuples (prob, next_state, reward, done).
        """
        return [
            (0.8, *self._calc_transition(r, c, a)),
            (0.1, *self._calc_transition(r, c, (a-1) % 4)),
            (0.1, *self._calc_transition(r, c, (a+1) % 4)),
        ]

    def _fix_hole_transitions(self):
        """
        For each hole cell, override all transitions to use slip model,
        but ensure landing on hole does NOT terminate the episode.
        """
        for s in range(self.nS):
            r, c = divmod(s, self.ncol)
            if self.orig_desc[r, c] == b'H':
                for a in range(self.nA):
                    self.P[s][a] = self._calc_slip(r, c, a)

    def step(self, a):
        """
        Override step to ignore hole-terminals: holes are non-absorbing.
        Returns obs, reward, done, info.
        """
        result = super().step(a)
        # Support Gym versions returning (obs,rew,done,info) or with truncation
        if len(result) == 5:
            obs, rew, done_flag, truncated, info = result
            done = done_flag or truncated
        else:
            obs, rew, done, info = result
        # If landing on hole, never done
        r, c = divmod(obs, self.ncol)
        if self.orig_desc[r, c] == b'H':
            done = False
        return obs, rew, done, info

# =============================================================================
# 2. Map Generation and Perturbation Helpers
# =============================================================================
np.random.seed(0)

def generate_custom_map(size=15, holes=10):
    """
    Create a size×size map with 'S' start, 'G' goal, and randomly placed holes.
    Returns array of encoded bytes for FrozenLake.
    """
    desc = np.full((size, size), 'F')
    desc[0,0], desc[-1,-1] = 'S', 'G'
    h = 0
    while h < holes:
        r, c = np.random.randint(size, size=2)
        if desc[r,c] == 'F':
            desc[r,c] = 'H'
            h += 1
    # Encode characters to bytes
    return np.array([[x.encode() for x in row] for row in desc])


def perturb_map(desc, shift=1, p=0.4):
    """
    Randomly move a fraction p of holes by up to 'shift' cells,
    creating a noisy version of the map for evaluation.
    """
    noisy = desc.copy()
    sz = desc.shape[0]
    for r, c in zip(*np.where(desc == b'H')):
        if np.random.rand() < p:
            dr, dc = np.random.randint(-shift, shift+1, 2)
            r2, c2 = np.clip([r+dr, c+dc], 0, sz-1)
            if noisy[r2, c2] == b'F':
                noisy[r, c], noisy[r2, c2] = b'F', b'H'
    return noisy

# =============================================================================
# 3. Utility Functions for State Encoding & Reward Shaping
# =============================================================================

def one_hot(s, N):
    """
    Convert integer state or action index to one-hot vector.
    """
    vec = np.zeros(N, dtype=np.float32)
    vec[int(s)] = 1.0
    return vec


def to_tensor(x):
    """
    Convert numpy array to PyTorch float tensor.
    """
    return torch.tensor(np.array(x), dtype=torch.float32)


def manhattan(a, b):
    """
    Compute Manhattan distance between two (row,col) tuples.
    """
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def precompute(env):
    """
    Identify goal and hole positions for reward shaping.
    """
    goal = None
    holes = []
    for r in range(env.nrow):
        for c in range(env.ncol):
            if env.orig_desc[r,c] == b'G':
                goal = (r,c)
            elif env.orig_desc[r,c] == b'H':
                holes.append((r,c))
    env._goal = goal
    env._holes = holes
    # Total distance for normalization
    env._tot = sum(manhattan(goal, h) for h in holes)


def shaped_reward(ns, env):
    """
    Provide a dense penalty when stepping on a hole, scaled by distance to goal.
    No shaping on regular frozen cells; positive reward on reaching G.
    """
    r, c = divmod(ns, env.ncol)
    if not hasattr(env, '_goal'):
        precompute(env)

    tile = env.orig_desc[r, c]
    if tile == b'H':
        # Normalize Manhattan distance to goal
        MAX_DIST = (env.nrow-1) + (env.ncol-1)
        d = manhattan((r, c), env._goal)
        return -PENALTY_C * (d / MAX_DIST)
    return 1.0 if tile == b'G' else 0.5

# =============================================================================
# 4. Training Hyperparameters
# =============================================================================
NUM_EP            = 300    # Number of training episodes
MAX_STEPS         = 150    # Max steps per episode
BATCH_SIZE        = 64     # Batch size for updates
GAMMA             = 0.99   # Discount factor
HORIZON           = 10     # Model rollout horizon
PENALTY_C         = 3.0    # Scaling factor for cost shaping
COST_LIMIT        = 3.0    # Allowed average cost threshold
LR_CRITIC         = LR_POLICY = LR_ALPHA = 3e-4  # Learning rates
TAU               = 0.005  # Soft update rate for target networks
ENTROPY_TGT       = -np.log(4) * 0.98  # Target entropy
N_POLICY_UPDATES  = 10     # Policy updates per episode

# =============================================================================
# 5. Environment Initialization
# =============================================================================
true_map  = generate_custom_map()
noisy_map = perturb_map(true_map)
train_env = NonAbsorbingHoleFrozenLake(desc=true_map, is_slippery=True)
test_env  = NonAbsorbingHoleFrozenLake(desc=noisy_map, is_slippery=True)
S_DIM, A_DIM = train_env.nS, train_env.nA

# =============================================================================
# 6. Network Definitions and Optimizers
# =============================================================================

class MLP(nn.Module):
    """
    Simple feedforward network for policy or cost prediction.
    """
    def __init__(self, in_dim, out_dim, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class QNet(nn.Module):
    """
    Critic network estimating Q-values for all actions given state.
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,a_dim)
        )
    def forward(self, x):
        return self.net(x)

class Dyn(nn.Module):
    """
    Dynamics model: predicts next-state one-hot distribution from state and action.
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim+a_dim,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,s_dim)
        )
    def forward(self, s, a):
        # Concatenate state and action vectors
        return self.net(torch.cat([s,a], 1))

# Instantiate policy, critics, cost models, and their target copies
policy   = MLP(S_DIM, A_DIM)
q1, q2   = QNet(S_DIM,A_DIM), QNet(S_DIM,A_DIM)
q1_t     = QNet(S_DIM,A_DIM); q1_t.load_state_dict(q1.state_dict())
q2_t     = QNet(S_DIM,A_DIM); q2_t.load_state_dict(q2.state_dict())
cost    = MLP(S_DIM,1)
cost_t  = MLP(S_DIM,1); cost_t.load_state_dict(cost.state_dict())

# Optimizers for each component
q1_opt   = optim.Adam(q1.parameters(), LR_CRITIC)
q2_opt   = optim.Adam(q2.parameters(), LR_CRITIC)
c_opt    = optim.Adam(cost.parameters(), 1e-3)
p_opt    = optim.Adam(policy.parameters(), LR_POLICY)

# Automatic entropy tuning
log_alpha = torch.tensor(0.0, requires_grad=True)
alpha_opt = optim.Adam([log_alpha], LR_ALPHA)

# Lagrange multiplier for cost penalty
lambda_var = torch.tensor(1.0, requires_grad=True)
lam_opt    = optim.Adam([lambda_var], 1e-3)

# Ensemble of dynamics models
ensemble = [Dyn(S_DIM,A_DIM) for _ in range(7)]
dyn_opts  = [optim.Adam(m.parameters(), 1e-3) for m in ensemble]

# Replay buffer initialization
D = ReplayBuffer(100_000)

# =============================================================================
# 7. SAC-style Update Helpers
# =============================================================================

def soft(target_net, source_net):
    """
    Soft-update target parameters toward source network.
    θ_target ← τ θ_source + (1-τ) θ_target
    """
    for tp, sp in zip(target_net.parameters(), source_net.parameters()):
        tp.data.copy_(TAU*sp.data + (1-TAU)*tp.data)

# Q-function update: clipped double Q-learning
def update_q(batch):
    s,a,ns,r,d = batch
    # Convert to tensors
    S  = to_tensor([one_hot(int(x), S_DIM) for x in s])
    S2 = to_tensor([one_hot(int(x), S_DIM) for x in ns])
    A  = torch.tensor(a, dtype=torch.long)
    R  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    Dn = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        # Next-state action distribution
        logits_next = policy(S2)
        p_next       = F.softmax(logits_next,1)
        logp_next    = F.log_softmax(logits_next,1)
        # Compute value target using target critics
        v_next = (p_next * (torch.min(q1_t(S2), q2_t(S2)) - torch.exp(log_alpha)*logp_next)).sum(1,keepdim=True)
        # TD target
        y      = R + GAMMA*(1 - Dn)*v_next

    # Critic loss and optimizer step
    for q, q_opt in [(q1,q1_opt),(q2,q2_opt)]:
        q_pred = q(S).gather(1, A.view(-1,1))
        loss   = F.mse_loss(q_pred, y)
        q_opt.zero_grad()
        loss.backward()
        q_opt.step()

# Cost critic update: learn cost-to-go and adjust λ
def update_cost(batch):
    s,a,ns,r,d = batch
    S  = to_tensor([one_hot(int(x), S_DIM) for x in s])
    S2 = to_tensor([one_hot(int(x), S_DIM) for x in ns])
    R  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    Dn = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        # target includes absolute raw cost plus discounted next cost
        y = R.abs() + GAMMA * cost_t(S2) * (1 - Dn)
    loss = F.mse_loss(cost(S), y)
    c_opt.zero_grad(); loss.backward(); c_opt.step()

    # Dual update for Lagrange multiplier
    lam_loss = -lambda_var * (cost(S).mean().detach() - COST_LIMIT)
    lam_opt.zero_grad(); lam_loss.backward(); lam_opt.step()
    lambda_var.data.clamp_(0.0)

# Policy and entropy coefficient update
def update_policy_alpha(batch):
    s = batch[0]
    S = to_tensor([one_hot(int(x), S_DIM) for x in s])
    logits = policy(S)
    p      = F.softmax(logits,1)
    logp   = F.log_softmax(logits,1)
    q_min  = torch.min(q1(S), q2(S))
    c_pred = cost(S)

    # Advantage with cost penalty
    adv = q_min - lambda_var.detach()*c_pred
    loss_pi = (p * (torch.exp(log_alpha)*logp - adv)).sum(1).mean()
    p_opt.zero_grad(); loss_pi.backward(); p_opt.step()

    # Adjust entropy coefficient to meet entropy target
    ent   = -(p*logp).sum(1).mean().detach()
    a_loss = torch.exp(log_alpha) * (-ent + ENTROPY_TGT)
    alpha_opt.zero_grad(); a_loss.backward(); alpha_opt.step()

# Imagined rollout using one random dynamics model
def simulate_rollout(s0, horizon=HORIZON):
    """
    Generate a synthetic trajectory from state s0 using the learned dynamics ensemble.
    Returns list of (s, a, ns, shaped_reward, done_flag).
    """
    path = []
    s = s0
    for _ in range(horizon):
        # Sample action from current policy
        S_tensor = to_tensor([one_hot(s, S_DIM)])
        logits   = policy(S_tensor)
        a        = F.softmax(logits,1).multinomial(1).item()
        A_tensor = to_tensor([one_hot(a, A_DIM)])
        # Predict next-state distribution and pick argmax
        dyn = random.choice(ensemble)
        with torch.no_grad():
            ns_pred = dyn(S_tensor, A_tensor).squeeze()
        ns = int(ns_pred.argmax().item())
        # Compute shaped reward and done flag
        r_shaped  = shaped_reward(ns, train_env)
        done_flag = (train_env.orig_desc.flatten()[ns] == b'G')
        path.append((s, a, ns, r_shaped, done_flag))
        s = ns
        if done_flag:
            break
    return path

# =============================================================================
# 8. Training Loop with Live Visualization
# =============================================================================
# Enable interactive plotting
plt.figure(figsize=(8, 8))
plt.ion()

def update_visualization(env, positions, episode):
    """
    Draw the grid, agent path, and current position onto a matplotlib plot.
    Called intermittently during training for live feedback.
    """
    plt.clf()
    desc = env.orig_desc
    nrow, ncol = env.nrow, env.ncol
    color_map = {b'S':'green', b'F':'white', b'H':'black', b'G':'gold'}
    # Draw each cell with label
    for r in range(nrow):
        for c in range(ncol):
            plt.text(c, r, desc[r,c].decode(), ha='center', va='center', fontsize=8, fontweight='bold')
            plt.gca().add_patch(plt.Rectangle((c-0.5,r-0.5),1,1,fill=True,edgecolor='gray',facecolor=color_map[desc[r,c]],alpha=0.3))
    # Plot path and current position
    if positions:
        path_r, path_c = zip(*positions)
        plt.plot(path_c, path_r, 'r-', linewidth=1)
        plt.scatter(path_c, path_r, c='red', s=30)
        last_r, last_c = positions[-1]
        plt.scatter(last_c, last_r, marker='o', s=100, edgecolor='red', facecolor='none', linewidth=2)
    plt.title(f"Training Episode {episode}")
    plt.xticks([]); plt.yticks([])
    plt.xlim(-0.5, ncol-0.5); plt.ylim(nrow-0.5, -0.5)
    plt.draw(); plt.pause(0.1)

# Histories for plotting at end
eps, R_hist, C_hist, L_hist = [], [], [], []

# Main training loop
for ep in range(1, NUM_EP+1):
    s, _ = train_env.reset()
    ep_r = ep_c = 0.0
    positions = []
    for t in range(MAX_STEPS):
        # Record position for visualization
        r, c = divmod(s, train_env.ncol)
        positions.append((r,c))
        # Select action by sampling policy distribution
        logits = policy(to_tensor(one_hot(s, S_DIM)).unsqueeze(0))
        a = F.softmax(logits,1).multinomial(1).item()
        # Step environment
        ns, _, done, _ = train_env.step(a)
        r_sh = shaped_reward(ns, train_env)
        # Store shaped experience
        D.append(s, a, ns, r_sh, done)
        ep_r += r_sh
        ep_c += max(0.0, -r_sh)
        s = ns
        # Live update every 10 episodes
        if t%5==0 and ep%10==0:
            update_visualization(train_env, positions, ep)
        if done:
            break
    # Final update for this episode
    if ep%10==0:
        update_visualization(train_env, positions, ep)
        plt.pause(0.5)
    # Train dynamics ensemble on real transitions
    for _ in range(20):
        if len(D)<BATCH_SIZE: break
        bs, ba, bns, _, _ = D.sample(BATCH_SIZE)
        S = to_tensor([one_hot(si, S_DIM) for si in bs])
        A = to_tensor([one_hot(ai, A_DIM) for ai in ba])
        NS= to_tensor([one_hot(nsi,S_DIM) for nsi in bns])
        for m,opt in zip(ensemble, dyn_opts):
            opt.zero_grad(); loss=F.mse_loss(m(S,A),NS); loss.backward(); opt.step()
    # Generate imagined rollouts to augment buffer
    bD=[]
    if len(D)>=BATCH_SIZE:
        for idx in np.random.choice(len(D), BATCH_SIZE, replace=False):
            s0 = D.buffer[idx][0]
            bD.extend(simulate_rollout(s0))
    mix = list(D.buffer) + bD
    # Multiple policy updates per episode
    for _ in range(N_POLICY_UPDATES):
        batch = random.sample(mix, BATCH_SIZE)
        batch = list(map(np.array, zip(*batch)))
        update_q(batch); update_cost(batch); update_policy_alpha(batch)
    # Soft updates for all targets
    soft(q1_t, q1); soft(q2_t, q2); soft(cost_t, cost)
    # Record stats
    eps.append(ep); R_hist.append(ep_r); C_hist.append(ep_c/(t+1)); L_hist.append(lambda_var.item())
    if ep%10==0:
        print(f"Ep{ep}: R={ep_r:.1f}, C={C_hist[-1]:.2f}, λ={L_hist[-1]:.2f}, α={torch.exp(log_alpha).item():.3f}")

# Disable interactive mode and show final plots
plt.ioff()
plt.figure();
plt.plot(eps, R_hist); plt.title('Reward vs Episode')
plt.figure(); plt.plot(eps, C_hist); plt.axhline(COST_LIMIT, ls='--'); plt.title('Cost vs Episode')
plt.figure(); plt.plot(eps, L_hist); plt.title('Lambda vs Episode')
plt.show()
