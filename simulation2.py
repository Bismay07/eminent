"""
Autonomous Car RL Environment using Gymnasium + Pygame
- Continuous actions: [steer, throttle]
- Observations: lidar ray distances + speed + heading error
- Simple straight-road task with random obstacles
- Compatible with Stable-Baselines3 (PPO/TD3/SAC)

Run quick training:
    pip install gymnasium pygame numpy stable-baselines3
    python autonomous_car_env.py --train 100000

Play with a random agent (render):
    python autonomous_car_env.py --render
"""
from __future__ import annotations
import math
import random
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# --------------------------
# Utility math / geometry
# --------------------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_angle_rad(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    a = (a + math.pi) % (2 * math.pi) - math.pi
    return a


def rotate_vec(x: float, y: float, ang: float) -> Tuple[float, float]:
    ca, sa = math.cos(ang), math.sin(ang)
    return ca * x - sa * y, sa * x + ca * y


def line_intersection(p: Tuple[float, float], r: Tuple[float, float], q: Tuple[float, float], s: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Return intersection point of param lines p + t r and q + u s where t,u in [0, inf) for rays, [0,1] for segments.
    Here we will use segment q->q+s (obstacle edges) and ray p->p+r (lidar).
    Returns point if intersects with t>=0 and u in [0,1], else None.
    """
    (px, py), (rx, ry) = p, r
    (qx, qy), (sx, sy) = q, s
    rxs = rx * sy - ry * sx
    if abs(rxs) < 1e-9:
        return None  # parallel
    qmpx, qmpy = qx - px, qy - py
    t = (qmpx * sy - qmpy * sx) / rxs
    u = (qmpx * ry - qmpy * rx) / rxs
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return (px + t * rx, py + t * ry)
    return None


def ray_rect_intersections(origin: Tuple[float, float], direction: Tuple[float, float], rect: pygame.Rect) -> List[Tuple[float, float]]:
    x, y = origin
    dx, dy = direction
    corners = [
        (rect.left, rect.top),
        (rect.right, rect.top),
        (rect.right, rect.bottom),
        (rect.left, rect.bottom),
    ]
    edges = [
        (corners[0], (corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])),
        (corners[1], (corners[2][0]-corners[1][0], corners[2][1]-corners[1][1])),
        (corners[2], (corners[3][0]-corners[2][0], corners[3][1]-corners[2][1])),
        (corners[3], (corners[0][0]-corners[3][0], corners[0][1]-corners[3][1])),
    ]
    pts = []
    for q, s in edges:
        inter = line_intersection((x, y), (dx, dy), q, s)
        if inter is not None:
            pts.append(inter)
    return pts

# --------------------------
# Config dataclass
# --------------------------

@dataclass
class CarConfig:
    length: float = 4.0  # meters
    width: float = 1.8
    max_steer: float = math.radians(30)
    max_accel: float = 4.0  # m/s^2
    max_brake: float = 6.0
    max_speed: float = 18.0  # ~65 km/h
    drag: float = 0.05
    steer_rate: float = math.radians(120)  # how fast steering can change


@dataclass
class WorldConfig:
    road_width: float = 8.0
    lane_center_heading: float = 0.0  # radians, road along +x
    speed_limit: float = 12.0
    track_length: float = 400.0  # meters to finish
    n_obstacles: int = 8
    obstacle_w: float = 1.6
    obstacle_h: float = 3.0
    min_gap_ahead: float = 20.0
    lidar_rays: int = 15
    lidar_max_dist: float = 35.0


# --------------------------
# Environment
# --------------------------

class AutonomousCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None,
                 car_cfg: CarConfig = CarConfig(),
                 world_cfg: WorldConfig = WorldConfig(),
                 seed: Optional[int] = None):
        super().__init__()
        self.car_cfg = car_cfg
        self.world_cfg = world_cfg
        self.render_mode = render_mode
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Observation: lidar distances (normalized), speed norm, heading error norm
        obs_len = world_cfg.lidar_rays + 2
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        # Action: [steer, throttle] where steer in [-1,1] -> [-max_steer, +max_steer], throttle in [0,1] accel forward, negative implies brake
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32),
                                       high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        # World state
        self.reset_world()

        # Rendering
        self.screen = None
        self.clock = None
        self.ppm = 8  # pixels per meter
        self.screen_size = (1000, 600)

    # ------------- World / Reset -------------
    def reset_world(self):
        wc = self.world_cfg
        # Car state
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.heading = 0.0
        self.steer = 0.0
        self.prev_x = 0.0
        self.t = 0.0

        # Obstacles: rectangles placed ahead with random lateral position within road
        self.obstacles: List[pygame.Rect] = []
        # use pygame.Rect in world meters (we'll map to pixels in render)
        # store as rects in meters by scaling later when drawing
        # We'll represent rects by (x, y, w, h) with origin at center; pygame.Rect expects ints, but for math we'll keep as floats in tuple.
        self.obs_boxes: List[Tuple[float, float, float, float]] = []

        x_pos = 30.0
        for _ in range(wc.n_obstacles):
            x_pos += self.np_random.uniform(wc.min_gap_ahead, wc.min_gap_ahead + 25.0)
            if x_pos >= wc.track_length - 20:
                break
            lateral = self.np_random.uniform(-wc.road_width * 0.45, wc.road_width * 0.45)
            self.obs_boxes.append((x_pos, lateral, wc.obstacle_w, wc.obstacle_h))

        self.done = False
        self.collision = False
        self.progress = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.reset_world()
        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return obs, info

    # ------------- Step -------------
    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32)
        steer_cmd = float(np.clip(a[0], -1.0, 1.0))
        throttle_cmd = float(np.clip(a[1], -1.0, 1.0))

        cc = self.car_cfg
        wc = self.world_cfg

        # Update steering (rate limited)
        target_steer = steer_cmd * cc.max_steer
        steer_delta = clamp(target_steer - self.steer, -cc.steer_rate * self.dt, cc.steer_rate * self.dt)
        self.steer += steer_delta

        # Longitudinal dynamics
        accel = 0.0
        if throttle_cmd >= 0:
            accel = throttle_cmd * cc.max_accel
        else:
            accel = throttle_cmd * cc.max_brake  # negative (brake)
        accel -= cc.drag * self.v  # simple drag

        self.v = clamp(self.v + accel * self.dt, 0.0, cc.max_speed)

        # Bicycle model kinematics
        beta = 0.0  # simplified single-track
        self.heading += (self.v / max(cc.length, 1e-3)) * math.tan(self.steer) * self.dt
        self.heading = wrap_angle_rad(self.heading)

        dx = self.v * math.cos(self.heading) * self.dt
        dy = self.v * math.sin(self.heading) * self.dt
        self.x += dx
        self.y += dy
        self.t += self.dt

        # Collisions & bounds
        self.collision = self._check_collision()
        off_road = abs(self.y) > (wc.road_width / 2.0)
        reached_goal = self.x >= wc.track_length

        # Reward shaping
        progress_reward = (self.x - self.prev_x) * 0.5  # forward progress
        center_penalty = -1.0 * (abs(self.y) / (wc.road_width / 2.0))
        heading_err = wrap_angle_rad(self.heading - wc.lane_center_heading)
        heading_penalty = -0.1 * abs(heading_err)
        steering_penalty = -0.02 * abs(self.steer) / cc.max_steer
        speed_penalty = -0.02 * max(0.0, self.v - wc.speed_limit) / max(1e-3, cc.max_speed - wc.speed_limit)
        collision_penalty = -10.0 if self.collision else 0.0
        offroad_penalty = -4.0 if off_road else 0.0
        goal_bonus = 10.0 if reached_goal else 0.0

        reward = progress_reward + center_penalty + heading_penalty + steering_penalty + speed_penalty + collision_penalty + offroad_penalty + goal_bonus

        self.prev_x = self.x

        # Termination conditions
        truncated = False
        terminated = False
        if self.collision or reached_goal:
            terminated = True
        if self.t > 40.0:  # time budget
            truncated = True
        if off_road and abs(self.y) > wc.road_width:  # way off
            terminated = True

        obs = self._get_obs()
        info = {
            "speed": self.v,
            "heading": self.heading,
            "steer": self.steer,
            "collision": self.collision,
            "off_road": off_road,
            "x_progress": self.x,
        }

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    # ------------- Observation -------------
    def _get_obs(self) -> np.ndarray:
        wc = self.world_cfg
        # Lidar rays in car frame
        angles = np.linspace(-math.pi/2, math.pi/2, wc.lidar_rays)  # 180 degree FOV
        dists = []
        for a in angles:
            d = self._ray_cast(self.x, self.y, self.heading + a, wc.lidar_max_dist)
            dists.append(d / wc.lidar_max_dist)
        dists = np.array(dists, dtype=np.float32)
        speed_norm = np.float32(self.v / self.car_cfg.max_speed)
        heading_err = wrap_angle_rad(self.heading - wc.lane_center_heading)
        heading_norm = np.float32((heading_err + math.pi) / (2 * math.pi))  # map [-pi,pi] -> [0,1]
        obs = np.concatenate([dists, [speed_norm, heading_norm]]).astype(np.float32)
        return obs

    def _ray_cast(self, x: float, y: float, ang: float, max_dist: float) -> float:
        """Return distance to nearest intersection with road edges or obstacles, up to max_dist."""
        wc = self.world_cfg
        # Intersect with road edges y = +/- road_width/2 (in world frame)
        candidates: List[float] = []
        dx, dy = math.cos(ang), math.sin(ang)
        # Edge lines are infinite; we consider only in front (t>=0)
        if abs(dy) > 1e-6:
            # Intersection with y = +w/2
            t1 = ( (wc.road_width/2) - y ) / dy
            if t1 >= 0:
                x_hit = x + t1 * dx
                if 0 <= x_hit <= (x + max_dist*dx) + max_dist:  # loose bound
                    candidates.append(t1)
            # y = -w/2
            t2 = ( (-wc.road_width/2) - y ) / dy
            if t2 >= 0:
                x_hit = x + t2 * dx
                if 0 <= x_hit <= (x + max_dist*dx) + max_dist:
                    candidates.append(t2)
        # Obstacles (rectangles): compute exact ray-rect intersections
        for (ox, oy, ow, oh) in self.obs_boxes:
            rect = pygame.Rect(0, 0, 0, 0)
            rect.left = int(ox - ow/2)
            rect.right = int(ox + ow/2)
            rect.top = int(oy - oh/2)
            rect.bottom = int(oy + oh/2)
            # Use segment edges intersection with the ray
            pts = ray_rect_intersections((x, y), (dx, dy), rect)
            for (ix, iy) in pts:
                t = math.hypot(ix - x, iy - y)
                if t >= 0:
                    candidates.append(t)
        if len(candidates) == 0:
            return max_dist
        d = min(candidates)
        return float(clamp(d, 0.0, max_dist))

    def _check_collision(self) -> bool:
        cc = self.car_cfg
        # Car footprint corners (axis-aligned approximation by expanded rect)
        # We'll approximate the car by a circle for fast collision with rectangles
        car_radius = 0.6 * max(cc.length, cc.width)
        cx, cy = self.x, self.y
        for (ox, oy, ow, oh) in self.obs_boxes:
            # distance from car center to rectangle
            dx = abs(cx - ox) - ow / 2
            dy = abs(cy - oy) - oh / 2
            if dx <= 0 and dy <= 0:
                return True  # inside obstacle
            dx = max(dx, 0)
            dy = max(dy, 0)
            if math.hypot(dx, dy) <= car_radius:
                return True
        return False

    # ------------- Rendering -------------
    @property
    def dt(self) -> float:
        return 1.0 / self.metadata["render_fps"]

    def _to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        # world (meters) to screen pixels
        sx = int(wx * self.ppm) + 50
        sy = int(self.screen_size[1] / 2 - wy * self.ppm)
        return sx, sy

    def _render_frame(self):
        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode(self.screen_size)
                self.clock = pygame.time.Clock()
            else:
                # headless
                os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
                pygame.display.init()
                self.screen = pygame.Surface(self.screen_size)
                self.clock = pygame.time.Clock()

        # Background
        self.screen.fill((20, 20, 20))

        # Draw road
        wc = self.world_cfg
        road_color = (50, 50, 50)
        lane_color = (200, 200, 50)
        left_edge = self._to_screen(0, wc.road_width/2)[1]
        right_edge = self._to_screen(0, -wc.road_width/2)[1]
        pygame.draw.rect(self.screen, road_color, pygame.Rect(0, right_edge, self.screen_size[0], left_edge - right_edge))

        # Center line
        for x in range(0, self.screen_size[0], 40):
            pygame.draw.line(self.screen, lane_color, (x, (left_edge+right_edge)//2), (x+20, (left_edge+right_edge)//2), 2)

        # Obstacles
        obs_color = (180, 70, 70)
        for (ox, oy, ow, oh) in self.obs_boxes:
            sx, sy = self._to_screen(ox, oy)
            rw = int(ow * self.ppm)
            rh = int(oh * self.ppm)
            rect = pygame.Rect(sx - rw//2, sy - rh//2, rw, rh)
            pygame.draw.rect(self.screen, obs_color, rect)

        # Car
        cc = self.car_cfg
        car_color = (70, 160, 220)
        sx, sy = self._to_screen(self.x, self.y)
        car_len_px = int(cc.length * self.ppm)
        car_w_px = int(cc.width * self.ppm)
        car_surf = pygame.Surface((car_len_px, car_w_px), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, car_color, pygame.Rect(0, 0, car_len_px, car_w_px), border_radius=6)
        car_surf = pygame.transform.rotate(car_surf, -math.degrees(self.heading))
        rect = car_surf.get_rect(center=(sx, sy))
        self.screen.blit(car_surf, rect)

        # Lidar visualization
        lidar_color = (180, 180, 180)
        for a in np.linspace(-math.pi/2, math.pi/2, wc.lidar_rays):
            d = self._ray_cast(self.x, self.y, self.heading + a, wc.lidar_max_dist)
            ex = self.x + d * math.cos(self.heading + a)
            ey = self.y + d * math.sin(self.heading + a)
            ps = self._to_screen(self.x, self.y)
            pe = self._to_screen(ex, ey)
            pygame.draw.line(self.screen, lidar_color, ps, pe, 1)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
        elif self.render_mode == "human":
            self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None


# --------------------------
# SB3 Training helper
# --------------------------

def make_env(render_mode: Optional[str] = None, seed: Optional[int] = None):
    def _thunk():
        env = AutonomousCarEnv(render_mode=render_mode, seed=seed)
        return env
    return _thunk


def train_sb3(total_timesteps: int = 200_000, algo: str = "PPO", save_path: str = "models/autocar_ppo"):
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([make_env(render_mode=None)])

    if algo.upper() == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="tb_logs")
    elif algo.upper() == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="tb_logs")
    elif algo.upper() == "TD3":
        model = TD3("MlpPolicy", env, verbose=1, tensorboard_log="tb_logs")
    else:
        raise ValueError("Unsupported algo. Choose from PPO/SAC/TD3.")

    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()
    print(f"Saved model to {save_path}")


def enjoy(model_path: str = "models/autocar_ppo.zip", algo: str = "PPO"):
    from stable_baselines3 import PPO, SAC, TD3

    env = AutonomousCarEnv(render_mode="human")
    if algo.upper() == "PPO":
        model = PPO.load(model_path)
    elif algo.upper() == "SAC":
        model = SAC.load(model_path)
    elif algo.upper() == "TD3":
        model = TD3.load(model_path)
    else:
        raise ValueError("Unsupported algo. Choose from PPO/SAC/TD3.")

    obs, _ = env.reset()
    done = False
    truncated = False
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()


# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=0, help="Train for N timesteps (0 to skip)")
    parser.add_argument("--algo", type=str, default="PPO", help="RL algorithm: PPO/SAC/TD3")
    parser.add_argument("--render", action="store_true", help="Run a random policy with rendering")
    parser.add_argument("--enjoy", action="store_true", help="Load model and enjoy (render)")
    parser.add_argument("--model", type=str, default="models/autocar_ppo.zip", help="Model path for --enjoy")
    args = parser.parse_args()

    if args.render:
        env = AutonomousCarEnv(render_mode="human")
        obs, _ = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()

    if args.train > 0:
        train_sb3(total_timesteps=args.train, algo=args.algo)

    if args.enjoy:
        enjoy(model_path=args.model, algo=args.algo)
