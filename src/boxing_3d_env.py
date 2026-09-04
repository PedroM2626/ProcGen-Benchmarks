import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple


class Boxing3DEnvState(NamedTuple):
    # Posições no ringue [-2.5, 2.5] x [-2.5, 2.5] x [0, 2.0]
    p1_pos: jnp.ndarray        # (3,) [x, y, z] centro de massa
    p1_rot: float              # Ângulo de tronco / orientação (radianos)
    p1_head: jnp.ndarray       # (3,) Coordenada da cabeça
    p1_glove_l: jnp.ndarray    # (3,) Luva esquerda
    p1_glove_r: jnp.ndarray    # (3,) Luva direita
    p1_guard: float            # Nível da guarda [0, 1]
    p1_hp: float               # Saúde [0, 100]
    p1_stamina: float          # Energia [0, 100]
    
    p2_pos: jnp.ndarray        # Oponente
    p2_rot: float
    p2_head: jnp.ndarray
    p2_glove_l: jnp.ndarray
    p2_glove_r: jnp.ndarray
    p2_guard: float
    p2_hp: float
    p2_stamina: float
    
    step_count: int


class HumanoidBoxing3DEnv:
    """
    Ambiente de Boxe 3D Humanoide Contínuo em JAX Puro.
    Dois lutadores em um ringue com física contínua de membros, socos, esquivas e guarda.
    Ações contínuas de 8 dimensões por lutador em [-1, 1].
    """
    def __init__(self, max_steps: int = 150):
        self.max_steps = max_steps
        self.action_dim = 8  # [vel_x, vel_y, rot_speed, guard, punch_l, punch_r, slip_x, slip_y]
        # Obs local por boxeador: 28 features contínuas
        self.obs_dim = 28

    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, Boxing3DEnvState]:
        rng, k1, k2 = jax.random.split(rng, 3)
        # Lutador 1 começa em (-1.0, 0, 0.9) olhando para frente (rot=0)
        p1_pos = jnp.array([-1.0, 0.0, 0.9])
        p1_head = jnp.array([-1.0, 0.0, 1.75])
        p1_gl = jnp.array([-0.8, -0.2, 1.5])
        p1_gr = jnp.array([-0.8, 0.2, 1.5])

        # Lutador 2 começa em (1.0, 0, 0.9) olhando para o P1 (rot=pi)
        p2_pos = jnp.array([1.0, 0.0, 0.9])
        p2_head = jnp.array([1.0, 0.0, 1.75])
        p2_gl = jnp.array([0.8, 0.2, 1.5])
        p2_gr = jnp.array([0.8, -0.2, 1.5])

        state = Boxing3DEnvState(
            p1_pos=p1_pos, p1_rot=0.0, p1_head=p1_head,
            p1_glove_l=p1_gl, p1_glove_r=p1_gr,
            p1_guard=0.5, p1_hp=100.0, p1_stamina=100.0,
            
            p2_pos=p2_pos, p2_rot=jnp.pi, p2_head=p2_head,
            p2_glove_l=p2_gl, p2_glove_r=p2_gr,
            p2_guard=0.5, p2_hp=100.0, p2_stamina=100.0,
            step_count=0
        )
        obs = self._get_obs(state)
        return obs, state

    def step(self, rng: jax.Array, state: Boxing3DEnvState, a1: jnp.ndarray, a2: jnp.ndarray):
        # a1, a2: (8,) ações contínuas em [-1, 1]
        a1 = jnp.clip(a1, -1.0, 1.0)
        a2 = jnp.clip(a2, -1.0, 1.0)

        # 1. Footwork & Movimento de Base no Ringue [-2.2, 2.2]
        new_p1_pos = jnp.clip(state.p1_pos + jnp.array([a1[0]*0.06, a1[1]*0.06, 0.0]), -2.2, 2.2)
        new_p2_pos = jnp.clip(state.p2_pos + jnp.array([a2[0]*0.06, a2[1]*0.06, 0.0]), -2.2, 2.2)

        # Distância entre centros dos boxeadores (manter colisão mínima de corpos)
        body_diff = new_p1_pos - new_p2_pos
        body_dist = jnp.sqrt(jnp.sum(body_diff**2) + 1e-6)
        push = jnp.maximum(0.0, 0.7 - body_dist) * 0.5
        dir_vec = body_diff / body_dist
        new_p1_pos = new_p1_pos + dir_vec * push
        new_p2_pos = new_p2_pos - dir_vec * push

        # 2. Rotação de Tronco e Posição da Cabeça com Esquivas (Head Slips)
        new_p1_rot = state.p1_rot + a1[2] * 0.08
        new_p2_rot = state.p2_rot + a2[2] * 0.08
        
        new_p1_head = jnp.array([new_p1_pos[0] + a1[6]*0.15, new_p1_pos[1] + a1[7]*0.15, 1.75 - jnp.abs(a1[6])*0.05])
        new_p2_head = jnp.array([new_p2_pos[0] + a2[6]*0.15, new_p2_pos[1] + a2[7]*0.15, 1.75 - jnp.abs(a2[6])*0.05])

        # 3. Extensão das Luvas (Socos Jab/Cross)
        p1_punch_l = jnp.maximum(0.0, a1[4]) * 0.65
        p1_punch_r = jnp.maximum(0.0, a1[5]) * 0.70
        p2_punch_l = jnp.maximum(0.0, a2[4]) * 0.65
        p2_punch_r = jnp.maximum(0.0, a2[5]) * 0.70

        # Direção de golpe de P1 em direção a P2
        fwd_1 = jnp.array([jnp.cos(new_p1_rot), jnp.sin(new_p1_rot), 0.0])
        fwd_2 = jnp.array([jnp.cos(new_p2_rot), jnp.sin(new_p2_rot), 0.0])

        new_p1_gl = new_p1_pos + jnp.array([-0.2, -0.1, 0.75]) + fwd_1 * (0.2 + p1_punch_l)
        new_p1_gr = new_p1_pos + jnp.array([-0.2, 0.1, 0.75]) + fwd_1 * (0.2 + p1_punch_r)
        
        new_p2_gl = new_p2_pos + jnp.array([0.2, 0.1, 0.75]) + fwd_2 * (0.2 + p2_punch_l)
        new_p2_gr = new_p2_pos + jnp.array([0.2, -0.1, 0.75]) + fwd_2 * (0.2 + p2_punch_r)

        # 4. Detecção de Impacto de Socos (Glove vs Opponent Head/Body e Alcance Cinemático)
        dist_fighters = jnp.sqrt(jnp.sum((new_p1_pos - new_p2_pos)**2) + 1e-6)

        d_p1_hit_l = jnp.sqrt(jnp.sum((new_p1_gl - new_p2_head)**2) + 1e-6)
        d_p1_hit_r = jnp.sqrt(jnp.sum((new_p1_gr - new_p2_head)**2) + 1e-6)
        p1_lands = (d_p1_hit_l < 0.45) | (d_p1_hit_r < 0.45) | ((dist_fighters < 0.70) & ((p1_punch_l > 0.3) | (p1_punch_r > 0.3)))

        # P2 acerta P1?
        d_p2_hit_l = jnp.sqrt(jnp.sum((new_p2_gl - new_p1_head)**2) + 1e-6)
        d_p2_hit_r = jnp.sqrt(jnp.sum((new_p2_gr - new_p1_head)**2) + 1e-6)
        p2_lands = (d_p2_hit_l < 0.45) | (d_p2_hit_r < 0.45) | ((dist_fighters < 0.70) & ((p2_punch_l > 0.3) | (p2_punch_r > 0.3)))

        # Modulação de Guarda (Bloqueio reduz dano em 75%)
        p1_guard_lvl = (a1[3] + 1.0) / 2.0
        p2_guard_lvl = (a2[3] + 1.0) / 2.0

        p1_dmg_dealt = jnp.where(p1_lands, 12.0 * (1.0 - 0.75 * (p2_guard_lvl > 0.45)), 0.0)
        p2_dmg_dealt = jnp.where(p2_lands, 12.0 * (1.0 - 0.75 * (p1_guard_lvl > 0.45)), 0.0)

        new_p2_hp = jnp.maximum(0.0, state.p2_hp - p1_dmg_dealt)
        new_p1_hp = jnp.maximum(0.0, state.p1_hp - p2_dmg_dealt)

        # Recompensas de Boxe
        # Bônus por acertos limpos, esquivas e punição por sofrer dano/gastar stamina à toa
        r1 = (p1_dmg_dealt * 0.3) - (p2_dmg_dealt * 0.35) - 0.01 * (p1_punch_l + p1_punch_r)
        r2 = (p2_dmg_dealt * 0.3) - (p1_dmg_dealt * 0.35) - 0.01 * (p2_punch_l + p2_punch_r)

        # Bônus de Knockdown terminal
        r1 = r1 + jnp.where((new_p2_hp <= 0.0) & (state.p2_hp > 0.0), 15.0, 0.0)
        r2 = r2 + jnp.where((new_p1_hp <= 0.0) & (state.p1_hp > 0.0), 15.0, 0.0)

        new_step = state.step_count + 1
        done = (new_step >= self.max_steps) | (new_p1_hp <= 0.0) | (new_p2_hp <= 0.0)

        new_state = Boxing3DEnvState(
            p1_pos=new_p1_pos, p1_rot=new_p1_rot, p1_head=new_p1_head,
            p1_glove_l=new_p1_gl, p1_glove_r=new_p1_gr,
            p1_guard=p1_guard_lvl, p1_hp=new_p1_hp, p1_stamina=100.0,
            
            p2_pos=new_p2_pos, p2_rot=new_p2_rot, p2_head=new_p2_head,
            p2_glove_l=new_p2_gl, p2_glove_r=new_p2_gr,
            p2_guard=p2_guard_lvl, p2_hp=new_p2_hp, p2_stamina=100.0,
            step_count=new_step
        )
        obs = self._get_obs(new_state)
        rewards = jnp.array([r1, r2])
        return obs, new_state, rewards, done

    def _get_obs(self, s: Boxing3DEnvState) -> jnp.ndarray:
        # Obs para P1 (28D) e Obs para P2 (28D)
        # Features: pos_propria (3), rot (1), head (3), luvas (6), guard (1), hp (1)
        # + pos_oponente (3), rot_oponente (1), head_oponente (3), luvas_oponente (6)
        obs_p1 = jnp.concatenate([
            s.p1_pos, jnp.array([s.p1_rot]), s.p1_head, s.p1_glove_l, s.p1_glove_r,
            jnp.array([s.p1_guard, s.p1_hp / 100.0]),
            s.p2_pos, jnp.array([s.p2_rot]), s.p2_head, s.p2_glove_l, s.p2_glove_r
        ])
        obs_p2 = jnp.concatenate([
            s.p2_pos, jnp.array([s.p2_rot]), s.p2_head, s.p2_glove_l, s.p2_glove_r,
            jnp.array([s.p2_guard, s.p2_hp / 100.0]),
            s.p1_pos, jnp.array([s.p1_rot]), s.p1_head, s.p1_glove_l, s.p1_glove_r
        ])
        return jnp.stack([obs_p1, obs_p2])
