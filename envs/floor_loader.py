# envs/floor_loader.py
import numpy as np
import random
from pathlib import Path

class FloorLoader:
    """
    Loads and processes real ALFWorld floor layouts
    """

    def __init__(self, floor_dir="data/world/floors", seed=42):
        self.floor_dir = Path(floor_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.floor = None
        self.walkable = None
        self.height = None
        self.width = None

    def load_random_floor(self):
        floor_files = list(self.floor_dir.glob("*.npy"))
        if not floor_files:
            raise FileNotFoundError("No floor layout files found.")

        floor_path = random.choice(floor_files)
        self.floor = np.load(floor_path)

        # ALFWorld: 0 = wall, 1 = walkable
        self.walkable = self.floor > 0
        self.height, self.width = self.floor.shape

        return floor_path.name

    def random_walkable_position(self):
        if self.walkable is None:
            raise RuntimeError("Floor not loaded.")

        ys, xs = np.where(self.walkable)
        idx = random.randint(0, len(xs) - 1)

        # Normalize to [0,1]
        x = xs[idx] / self.width
        y = ys[idx] / self.height

        return np.array([x, y])

    def is_walkable(self, pos):
        x = int(pos[0] * self.width)
        y = int(pos[1] * self.height)

        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return False

        return self.walkable[y, x]

    def clip_to_walkable(self, pos):
        """
        Snap agent back to nearest walkable cell if collision occurs
        """
        if self.is_walkable(pos):
            return pos

        ys, xs = np.where(self.walkable)
        coords = np.stack([xs / self.width, ys / self.height], axis=1)
        dists = np.linalg.norm(coords - pos, axis=1)
        return coords[np.argmin(dists)]
