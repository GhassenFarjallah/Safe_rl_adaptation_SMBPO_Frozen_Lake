# -*- coding: utf-8 -*-
"""
Created on Thu May  1 20:33:42 2025

@author: ghass
"""


import gym, random, matplotlib.pyplot as plt
import numpy as np, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP conflict
# ────────────────────────────────
# 0. Replay buffer
# ────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def append(self, *exp):
        self.buffer.append(exp)
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(map(np.array, zip(*batch)))
    def __len__(self):
        return len(self.buffer)



# ────────────────────────────────
# 1. FrozenLake 15×15 non-absorbing
# ────────────────────────────────
class NonAbsorbingHoleFrozenLake(FrozenLakeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nrow, self.ncol = self.desc.shape
        self.nS, self.nA = self.nrow * self.ncol, self.action_space.n
        self.orig_desc = self.desc.copy()
        self._fix_hole_transitions()

    def _calc_transition(self, r, c, a):
        if a == 0:
            nc, nr = max(c-1, 0), r
        elif a == 1:
            nr, nc = min(r+1, self.nrow-1), c
        elif a == 2:
            nc, nr = min(c+1, self.ncol-1), r
        else:
            nr, nc = max(r-1, 0), c
        ns = nr * self.ncol + nc
        ltr = self.desc[nr, nc]
        return ns, float(ltr == b'G'), (ltr == b'G')

    def _calc_slip(self, r, c, a):
        return [
            (0.8, *self._calc_transition(r, c, a)),
            (0.1, *self._calc_transition(r, c, (a-1) % 4)),
            (0.1, *self._calc_transition(r, c, (a+1) % 4)),
        ]

    def _fix_hole_transitions(self):
        for s in range(self.nS):
            r, c = divmod(s, self.ncol)
            if self.orig_desc[r, c] == b'H':
                for a in range(self.nA):
                    self.P[s][a] = self._calc_slip(r, c, a)

    def step(self, a):
        out = super().step(a)
        if len(out) == 5:
            obs, rew, done_flag, truncated, info = out
            done = done_flag or truncated
        else:
            obs, rew, done, info = out
        r, c = divmod(obs, self.ncol)
        if self.orig_desc[r, c] == b'H':
            done = False
        return obs, rew, done, info

# ────────────────────────────────
# 2. Map helpers
# ────────────────────────────────
np.random.seed(0)

def generate_custom_map(size=15, holes=10):
    desc = np.full((size, size), 'F')
    desc[0,0], desc[-1,-1] = 'S','G'
    h = 0
    while h < holes:
        r, c = np.random.randint(size, size=2)
        if desc[r,c] == 'F':
            desc[r,c] = 'H'
            h += 1
    return np.array([[x.encode() for x in row] for row in desc])


def perturb_map(desc, shift=1, p=0.4):
    noisy = desc.copy()
    sz = desc.shape[0]
    for r, c in zip(*np.where(desc == b'H')):
        if np.random.rand() < p:
            dr, dc = np.random.randint(-shift, shift+1, 2)
            r2, c2 = np.clip([r+dr, c+dc], 0, sz-1)
            if noisy[r2, c2] == b'F':
                noisy[r, c], noisy[r2, c2] = b'F', b'H'
    return noisy



# ─────────────────────────────

# ────────────────────────────────
# 2. Map helpers
# ────────────────────────────────
np.random.seed(0)


# ────────────────────────────────
# 3. Utils
# ────────────────────────────────

def one_hot(s, N):
    return np.eye(N, dtype=np.float32)[int(s)]

def to_tensor(x):
    return torch.tensor(np.array(x), dtype=torch.float32)

def manhattan(a,b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def precompute(env):
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
    env._tot = sum(manhattan(goal,h) for h in holes)

def shaped_reward(ns, env):
    r, c = divmod(ns, env.ncol)
    if not hasattr(env, '_goal'):
        precompute(env)
    
    tile = env.orig_desc[r,c]
    if tile == b'H':
        MAX_DIST = (env.nrow-1)+(env.ncol-1)
        d = manhattan((r,c), env._goal)
        return -PENALTY_C * (d / MAX_DIST)
    return 1.0 if tile == b'G' else 0.5

# ────────────────────────────────
# 4. Hyper-params
# ────────────────────────────────
NUM_EP            = 300
MAX_STEPS         = 150
BATCH_SIZE        = 64
GAMMA             = 0.99
HORIZON           = 10
PENALTY_C         = 3.0
COST_LIMIT        = 3.0
LR_CRITIC         = LR_POLICY = LR_ALPHA = 3e-4
TAU               = 0.005
ENTROPY_TGT       = -np.log(4) * 0.98
N_POLICY_UPDATES  = 10

# ────────────────────────────────
# 5. Env init
# ────────────────────────────────
true_map  = generate_custom_map()
train_env = NonAbsorbingHoleFrozenLake(desc=true_map, is_slippery=True)
S_DIM, A_DIM = train_env.nS, train_env.nA

# ────────────────────────────────
# 6. Networks & optimizers
# ────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class QNet(nn.Module):
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
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim+a_dim,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(),
            nn.Linear(64,s_dim)
        )
    def forward(self, s, a):
        return self.net(torch.cat([s,a], 1))

policy   = MLP(S_DIM, A_DIM)
q1, q2   = QNet(S_DIM,A_DIM), QNet(S_DIM,A_DIM)
q1_t     = QNet(S_DIM,A_DIM); q1_t.load_state_dict(q1.state_dict())
q2_t     = QNet(S_DIM,A_DIM); q2_t.load_state_dict(q2.state_dict())
q1_opt   = optim.Adam(q1.parameters(), LR_CRITIC)
q2_opt   = optim.Adam(q2.parameters(), LR_CRITIC)
p_opt    = optim.Adam(policy.parameters(), LR_POLICY)

cost    = MLP(S_DIM,1)
cost_t  = MLP(S_DIM,1); cost_t.load_state_dict(cost.state_dict())
c_opt   = optim.Adam(cost.parameters(), 1e-3)

log_alpha = torch.tensor(0.0, requires_grad=True)
alpha_opt = optim.Adam([log_alpha], LR_ALPHA)

lambda_var = torch.tensor(1.0, requires_grad=True)
lam_opt     = optim.Adam([lambda_var], 1e-3)

ensemble = [Dyn(S_DIM,A_DIM) for _ in range(7)]
dyn_opts  = [optim.Adam(m.parameters(), 1e-3) for m in ensemble]

D = ReplayBuffer(100_000)

# ────────────────────────────────
# 7. SAC-lite helpers (with dtype fixes)
# ────────────────────────────────
def soft(tar, src):
    for tp, sp in zip(tar.parameters(), src.parameters()):
        tp.data.copy_(TAU*sp.data + (1-TAU)*tp.data)

def update_q(batch):
    s,a,ns,r,d = batch
    S  = to_tensor([one_hot(int(x), S_DIM) for x in s])
    S2 = to_tensor([one_hot(int(x), S_DIM) for x in ns])
    A  = torch.tensor(a, dtype=torch.long)
    R  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    Dn = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        logits_next = policy(S2)
        p_next       = F.softmax(logits_next,1)
        logp_next    = F.log_softmax(logits_next,1)
        v_next = (p_next * (torch.min(q1_t(S2), q2_t(S2)) - torch.exp(log_alpha)*logp_next)).sum(1,keepdim=True)
        y      = R + GAMMA*(1 - Dn)*v_next

    for q, q_opt in [(q1,q1_opt),(q2,q2_opt)]:
        q_pred = q(S).gather(1, A.view(-1,1))
        loss   = F.mse_loss(q_pred, y)
        q_opt.zero_grad()
        loss.backward()
        q_opt.step()

def update_cost(batch):
    s,a,ns,r,d = batch
    S  = to_tensor([one_hot(int(x), S_DIM) for x in s])
    S2 = to_tensor([one_hot(int(x), S_DIM) for x in ns])
    R  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
    Dn = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        y = R.abs() + GAMMA * cost_t(S2) * (1 - Dn)

    loss = F.mse_loss(cost(S), y)
    c_opt.zero_grad()
    loss.backward()
    c_opt.step()

    lam_loss = -lambda_var * (cost(S).mean().detach() - COST_LIMIT)
    lam_opt.zero_grad()
    lam_loss.backward()
    lam_opt.step()
    lambda_var.data.clamp_(0.0)

def update_policy_alpha(batch):
    s = batch[0]
    S = to_tensor([one_hot(int(x), S_DIM) for x in s])
    logits = policy(S)
    p      = F.softmax(logits,1)
    logp   = F.log_softmax(logits,1)
    q_min  = torch.min(q1(S), q2(S))
    c_pred = cost(S)

    adv = q_min - lambda_var.detach()*c_pred
    loss_pi = (p * (torch.exp(log_alpha)*logp - adv)).sum(1).mean()
    p_opt.zero_grad()
    loss_pi.backward()
    p_opt.step()

    ent   = -(p*logp).sum(1).mean().detach()
    a_loss = torch.exp(log_alpha) * (-ent + ENTROPY_TGT)
    alpha_opt.zero_grad()
    a_loss.backward()
    alpha_opt.step()
def simulate_rollout(s0, horizon=HORIZON):
    """Génère une trajectoire imaginée à partir de s0 via un modèle aléatoire."""
    path = []
    s = s0
    for _ in range(horizon):
        # 1) échantillonner l'action
        S_tensor = to_tensor([one_hot(s, S_DIM)])
        logits   = policy(S_tensor)
        a        = F.softmax(logits, 1).multinomial(1).item()
        # 2) one-hot de l’action pour le modèle
        A_tensor = to_tensor([one_hot(a, A_DIM)])
        # 3) choisir un modèle dyn de l’ensemble
        dyn = random.choice(ensemble)
        with torch.no_grad():
            ns_pred_vec = dyn(S_tensor, A_tensor).squeeze()
        ns = int(ns_pred_vec.argmax().item())
        # 4) calculer reward et done
        r_shaped   = shaped_reward(ns, train_env)
        done_flag  = (train_env.orig_desc.flatten()[ns] == b'G')
        # 5) stocker et avancer
        path.append((s, a, ns, r_shaped, done_flag))
        s = ns
        if done_flag:
            break
    return path

####drawing map####
plt.figure(figsize=(8, 8))  # Create figure for live updates
plt.ion()  # Enable interactive mode

def update_visualization(env, positions, episode):
    """Update the visualization plot with current agent path"""
    plt.clf()
    desc = env.orig_desc
    nrow, ncol = env.nrow, env.ncol
    
    # Create background grid
    color_map = {
        b'S': 'green',
        b'F': 'white',
        b'H': 'black',
        b'G': 'gold'
    }
    
    # Draw grid cells
    for r in range(nrow):
        for c in range(ncol):
            plt.text(c, r, desc[r, c].decode(), 
                    ha='center', va='center', 
                    color=color_map.get(desc[r, c], 'white'),
                    fontsize=8,
                    fontweight='bold')
            plt.gca().add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                            fill=True, 
                                            edgecolor='gray',
                                            facecolor=color_map.get(desc[r, c], 'white'),
                                            alpha=0.3))
    
    # Draw agent path
    if len(positions) > 1:
        path_r = [pos[0] for pos in positions]
        path_c = [pos[1] for pos in positions]
        plt.plot(path_c, path_r, 'r-', linewidth=1)
        plt.scatter(path_c, path_r, c='red', s=30)
    
    # Add current position marker
    if positions:
        last_r, last_c = positions[-1]
        plt.scatter(last_c, last_r, marker='o', s=100, 
                   edgecolor='red', facecolor='none', linewidth=2)
    
    plt.title(f"Training Episode {episode}\n"
             f"Current Position: ({last_c}, {last_r})" if positions else f"Training Episode {episode}")
    plt.xticks([]); plt.yticks([])
    plt.xlim(-0.5, ncol-0.5); plt.ylim(nrow-0.5, -0.5)
    plt.draw()
    plt.pause(0.1)

# ────────────────────────────────
# 8. Training loop
# ────────────────────────────────
eps, R_hist, C_hist, L_hist = [], [], [], []

for ep in range(1, NUM_EP+1):
    s, _ = train_env.reset()
    ep_r = ep_c = 0.0
    positions = []  # Store agent positions for visualization
    
    for t in range(MAX_STEPS):
        # Record current position
        r, c = divmod(s, train_env.ncol)
        positions.append((r, c))
        
        # Policy action selection
        logits = policy(to_tensor(one_hot(s, S_DIM)).unsqueeze(0))
        a = F.softmax(logits, 1).multinomial(1).item()
        
        # Environment step
        ns, r_raw, done, _ = train_env.step(a)
        shaped = shaped_reward(ns, train_env)
        
        # Store experience
        D.append(s, a, ns, shaped, done)
        ep_r += shaped
        ep_c += max(0.0, -shaped)
        s = ns
        
        # Update visualization every 5 steps
        if t % 5 == 0 and ep % 10 == 0:
            update_visualization(train_env, positions, ep)
            
        if done:
            break
    
    # Final visualization update for the episode
    if ep % 10 == 0:
        update_visualization(train_env, positions, ep)
        plt.pause(0.5)
    # (optional) train dynamics ensemble...
    for _ in range(20):
        if len(D) < BATCH_SIZE:
            break
        bs, ba, bns, _, _ = D.sample(BATCH_SIZE)
        S   = to_tensor([one_hot(si, S_DIM) for si in bs])
        A   = to_tensor([one_hot(ai, A_DIM) for ai in ba])
        NS  = to_tensor([one_hot(nsi, S_DIM) for nsi in bns])
        for dyn, opt in zip(ensemble, dyn_opts):
            opt.zero_grad()
            pred = dyn(S, A)
            loss = F.mse_loss(pred, NS)
            loss.backward()
            opt.step()

    # simulate-rollout via le modèle pour enrichir le buffer
    bD = []
    if len(D) >= BATCH_SIZE:
        idxs = np.random.choice(len(D), BATCH_SIZE, replace=False)
        for idx in idxs:
            s0 = D.buffer[idx][0]    # état initial du sample
            bD.extend(simulate_rollout(s0))

    mix = list(D.buffer) + bD

    for _ in range(N_POLICY_UPDATES):
        batch = random.sample(mix, BATCH_SIZE)
        batch = list(map(np.array, zip(*batch)))
        update_q(batch)
        update_cost(batch)
        update_policy_alpha(batch)

    soft(q1_t, q1)
    soft(q2_t, q2)
    soft(cost_t, cost)

    eps.append(ep)
    R_hist.append(ep_r)
    C_hist.append(ep_c / (t+1))
    L_hist.append(lambda_var.item())

    if ep % 10 == 0:
        print(f"Ep{ep}: R={ep_r:.1f}, C={C_hist[-1]:.2f}, λ={L_hist[-1]:.2f}, α={torch.exp(log_alpha).item():.3f}")
plt.ioff()
plt.show()
# ────────────────────────────────
# 9. Evaluation & plots
# ────────────────────────────────
plt.figure(); plt.plot(eps, R_hist); plt.title('Reward vs Episode')
plt.figure(); plt.plot(eps, C_hist); plt.axhline(COST_LIMIT, ls='--'); plt.title('Cost vs Episode')
plt.figure(); plt.plot(eps, L_hist); plt.title('Lambda vs Episode')
plt.show()



# ─────────────────────────────────────────────
# FINAL EVALUATION on the noisy map (test_env)
# ─────────────────────────────────────────────
# ────────────────────────────────
# 10. Test dynamique avec perturbations
# ────────────────────────────────
class DynamicHoleWrapper(gym.Wrapper):
    def __init__(self, base_env, true_map, perturb_prob=0.3, shift=1):
        super().__init__(base_env)
        self.true_map = true_map
        self.perturb_prob = perturb_prob
        self.shift = shift
        self.current_desc = true_map.copy()

    def _perturb_step(self):
        """Applique une perturbation aléatoire à chaque étape avec probabilité"""
        if np.random.rand() < self.perturb_prob:
            self.current_desc = perturb_map(self.current_desc, shift=self.shift, p=0.2)
            self.update_env_map()

    def update_env_map(self):
        """Met à jour la carte de l'environnement"""
        self.env.desc = self.current_desc.copy()
        self.env._fix_hole_transitions()

    def reset(self):
        self.current_desc = self.true_map.copy()
        self.update_env_map()
        return super().reset()

    def step(self, action):
        self._perturb_step()
        return super().step(action)

#environnement de test perturbé (c'est à dire position avec probabilité d'erreur de présence dans une position)
test_env = DynamicHoleWrapper(
    NonAbsorbingHoleFrozenLake(desc=true_map, is_slippery=True),
    true_map=true_map,
    perturb_prob=0.2,
    shift=1
)
S_DIM, A_DIM = train_env.nS, train_env.nA
eps, R_hist, C_hist, L_hist = [], [], [], []
test_metrics = {'eps': [], 'reward': [], 'cost': [], 'success': [], 'lambda': []}

#evaluation politique pour l'utiliser à chaque itération
def evaluate_policy(env, policy, num_episodes=300, max_steps=MAX_STEPS):
    success_rate = 0
    total_cost = 0
    total_reward = 0

    test_R_hist, test_C_hist, test_S_hist, test_L_hist = [], [], [], []
    for _ in range(num_episodes):
        s = env.reset()
        ep_reward = 0
        ep_cost = 0
        done = False
        steps = 0


        while not done and steps < max_steps:
            with torch.no_grad():
                logits = policy(to_tensor(one_hot(s, S_DIM)).unsqueeze(0))
                a = F.softmax(logits, 1).multinomial(1).item()

            ns, r, done, _ = env.step(a)
            shaped = shaped_reward(ns, env)

            ep_reward += shaped
            ep_cost += max(0.0, -shaped)
            s = ns
            steps += 1

            if env.orig_desc.flatten()[s] == b'G':
                success_rate += 1
                break

    avg_reward = total_reward / num_episodes
    avg_cost = total_cost / num_episodes
    success_rate = success_rate / num_episodes * 100

    return avg_reward, avg_cost, success_rate

for ep in range(1, NUM_EP+1):


    # Évaluation périodique
    if ep % 10 == 0:
        test_reward, test_cost, test_success = evaluate_policy(test_env, policy,num_episodes=500)

        # Enregistrement des métriques
        test_metrics['eps'].append(ep)
        test_metrics['reward'].append(test_reward)
        test_metrics['cost'].append(test_cost)
        test_metrics['success'].append(test_success)
        test_metrics['lambda'].append(lambda_var.item())

        print(f"Ep{ep}: Test R={test_reward:.1f}, C={test_cost:.2f}, Success={test_success:.1f}%")

# ────────────────────────────────
# 9. Visualisation (version corrigée)
# ────────────────────────────────
plt.figure(figsize=(14, 10))

# Reward
plt.subplot(221)
plt.plot(eps, R_hist, label='Train')
plt.plot(test_metrics['eps'], test_metrics['reward'], 'o-', color='orange', label='Test')
plt.title('Reward Comparison')
plt.ylabel('Average Reward')
plt.legend()

# Cost
plt.subplot(222)
plt.plot(eps, C_hist, label='Train')
plt.plot(test_metrics['eps'], test_metrics['cost'], 'o-', color='orange', label='Test')
plt.axhline(COST_LIMIT, color='r', linestyle='--', label='Cost Limit')
plt.title('Cost Comparison')
plt.ylabel('Average Cost')
plt.legend()

# Lambda
plt.subplot(223)
plt.plot(eps, L_hist, label='Train Lambda')
plt.plot(test_metrics['eps'], test_metrics['lambda'], 'o-', color='green', label='Test Lambda')
plt.title('Lambda Value Evolution')
plt.ylabel('Lambda Value')
plt.legend()

# Success Rate
plt.subplot(224)
plt.plot(test_metrics['eps'], test_metrics['success'], 's-', color='purple')
plt.ylim(0, 100)
plt.title('Test Environment Success Rate')
plt.ylabel('Success Rate (%)')
plt.xlabel('Episodes')

plt.tight_layout()
plt.show()
