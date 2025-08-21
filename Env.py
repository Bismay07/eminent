
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import sys

class DrivingEnvHybrid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode="human", num_rays=7, screen_size=800, render_every=1):
        super().__init__()
        self.render_mode = render_mode
        self.num_rays = num_rays
        self.screen_size = screen_size
        self.car_size = (20, 40)
        self.max_speed = 5.0
        self.max_sensor_range = float(self.screen_size)
        self.render_every = max(1, render_every)
        self._render_counter = 0

        # --- Action: [steering, throttle]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- Observation: only sensors (ray distances)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_rays,), dtype=np.float32
        )

        # --- Build map layout (1 = obstacle, 0 = road)
        map_layout = [
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        ]

        obstacle_tuples = []
        margin = int(self.screen_size * 0.08)
        cols, rows = len(map_layout[0]), len(map_layout)
        grid_w, grid_h = self.screen_size - 2*margin, self.screen_size - 2*margin
        cell_w, cell_h = grid_w // cols, grid_h // rows
        for j in range(rows):
            for i in range(cols):
                if map_layout[j][i] == 1:  # obstacle
                    w, h = int(cell_w), int(cell_h)
                    x, y = margin + i * cell_w, margin + j * cell_h
                    obstacle_tuples.append((x, y, w, h))
        self.obstacle_rects = [pygame.Rect(t) for t in obstacle_tuples]

        # --- Pygame setup
        pygame.init()
        self._world_surf = pygame.Surface((self.screen_size, self.screen_size))
        if self.render_mode == "human":
            self._window = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("DrivingEnvHybrid")
            self._clock = pygame.time.Clock()
        else:
            self._window = None
            self._clock = None

        pygame.font.init()
        self._font = pygame.font.SysFont("Arial", 18)

        # --- State
        self.episode_num = 0
        self.step_num = 0
        self.car_pos = np.array([self.screen_size/2, self.screen_size-80], dtype=np.float32)
        self.car_angle = 180.0
        self.speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_num += 1
        self.step_num = 0
        self.car_pos = np.array([self.screen_size/2, self.screen_size-80], dtype=np.float32)
        self.car_angle = 180.0
        self.speed = 0.0
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, {}

    def step(self, action):
        steer, throttle = np.clip(action, -1.0, 1.0)
        self.car_angle += steer * 4.5
        self.speed += throttle * 0.25
        self.speed = np.clip(self.speed, -self.max_speed, self.max_speed)

        rad = math.radians(self.car_angle)
        self.car_pos += np.array([math.sin(rad)*self.speed, -math.cos(rad)*self.speed], dtype=np.float32)
        self.step_num += 1

        # --- Collision check
        car_rect = pygame.Rect(
            int(self.car_pos[0]-self.car_size[0]/2),
            int(self.car_pos[1]-self.car_size[1]/2),
            int(self.car_size[0]),
            int(self.car_size[1])
        )
        collided = car_rect.left < 0 or car_rect.right > self.screen_size or car_rect.top < 0 or car_rect.bottom > self.screen_size
        if not collided:
            for r in self.obstacle_rects:
                if car_rect.colliderect(r):
                    collided = True
                    break

        # --- Reward
        reward = 0.05 + 0.3*(abs(self.speed)/self.max_speed) + 0.02*max(0.0, -math.cos(rad)*self.speed)
        if collided:
            reward -= 5.0

        terminated = collided
        truncated = self.step_num >= 1000
        obs = self._get_obs()
        info = {"collided": collided}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _cast_rays(self):
        distances = []
        fov = 180.0  # Narrow cone in front
        for i in range(self.num_rays):
            # Spread rays evenly in the front cone
            if self.num_rays == 1:
                a_deg = self.car_angle
            else:
                a_deg = self.car_angle - fov/2 + fov * i / (self.num_rays - 1)
            distances.append(self._ray_distance(math.radians(a_deg)))
        return np.array(distances, dtype=np.float32)


    def _ray_distance(self, angle_rad):
        step, r = 4.0, 0.0
        while r <= self.max_sensor_range:
            px = self.car_pos[0] + r * math.sin(angle_rad)
            py = self.car_pos[1] - r * math.cos(angle_rad)
            ix, iy = int(round(px)), int(round(py))
            if ix < 0 or ix >= self.screen_size or iy < 0 or iy >= self.screen_size:
                return min(r, self.max_sensor_range)
            for rect in self.obstacle_rects:
               if rect.collidepoint(ix, iy):
                    return min(r, self.max_sensor_range)
            r += step
        return r


    def _draw_world(self, surf):
        surf.fill((25,25,28))
        pygame.draw.rect(surf, (70,70,80), (0,0,self.screen_size,self.screen_size), 6)
        for rect in self.obstacle_rects:
            pygame.draw.rect(surf, (200,70,70), rect)
        car_surf = pygame.Surface((int(self.car_size[0]), int(self.car_size[1])), pygame.SRCALPHA)
        car_surf.fill((70,200,120))
        pygame.draw.rect(car_surf, (20,20,30), (0,0,int(self.car_size[0]),6))
        rot = pygame.transform.rotate(car_surf, -self.car_angle-90)
        surf.blit(rot, rot.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1]))))
        distances = self._cast_rays()
        # for i, d in enumerate(distances):
        #     if d <= 0: continue
        #     a_deg = self.car_angle - 90.0 + i*(180.0/max(1,self.num_rays-1))
        #     a_rad = math.radians(a_deg)
        #     px = int(round(self.car_pos[0] + d*math.cos(math.pi/2-a_rad)))
        #     py = int(round(self.car_pos[1] + d*math.sin(math.pi/2-a_rad)))
        #     pygame.draw.line(surf, (180,180,80), (int(self.car_pos[0]), int(self.car_pos[1])), (px,py), 1)
        #     pygame.draw.circle(surf, (180,180,80), (px,py), 3)


        fov = 180.0  # match the ray FOV
        for i, d in enumerate(distances):
            if d <= 0:
                continue
            a_deg = self.car_angle - fov/2 + fov*i/(self.num_rays-1)        
            a_rad = math.radians(a_deg)
            px = int(round(self.car_pos[0] + d*math.sin(a_rad)))
            py = int(round(self.car_pos[1] - d*math.cos(a_rad)))
            pygame.draw.line(surf, (180,180,80), (int(self.car_pos[0]), int(self.car_pos[1])), (px,py), 1)
            pygame.draw.circle(surf, (180,180,80), (px,py), 3)




    def _get_obs(self):
        sensors = np.clip(self._cast_rays()/float(self.max_sensor_range), 0.0, 1.0).astype(np.float32)
        return sensors

    def render(self):
        if self._window is None: return
        self._draw_world(self._world_surf)
        if self._font:
            self._world_surf.blit(self._font.render(f"Episode: {self.episode_num}", True, (230,230,230)), (8,8))
            self._world_surf.blit(self._font.render(f"Step: {self.step_num}", True, (230,230,230)), (8,30))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
        self._window.blit(self._world_surf, (0,0))
        pygame.display.flip()
        if self._clock: self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._window:
            pygame.display.quit()
            pygame.quit()
            
            self._window = None
            self._clock = None
            self._world_surf = None
