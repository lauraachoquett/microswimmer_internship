from pathlib import Path

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )


def random_bg_parameters():
    dir = np.random.uniform(-1, 1, 2)
    dir = dir / np.linalg.norm(dir)
    norm = np.random.rand() * 0.6

    a = np.random.rand()
    center = [np.random.rand() * 2, np.random.rand()]
    cir = (np.random.rand() - 0.5) * 2
    return dir, norm, center, a, cir


def courbures(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    numerateur = np.abs(dx * ddy - dy * ddx)
    denominateur = (dx**2 + dy**2) ** 1.5
    courbure = numerateur / (denominateur + 1e-8)  # pour éviter la division par 0

    return courbure


def to_tensor(x, device="cpu", dtype=torch.float32):
    if isinstance(x, np.ndarray):
        x = x.copy()
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    elif isinstance(x, (float, int)):
        x = x.copy()
        return torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    else:
        raise TypeError(f"Unsupported type {type(x)} for conversion to torch.Tensor.")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (float, int, np.ndarray)):
        return np.array(x)
    else:
        raise TypeError(f"Unsupported type {type(x)} for conversion to numpy.")

def create_numbered_run_folder(parent_dir):
    parent_path = Path(parent_dir)
    parent_path.mkdir(parents=True, exist_ok=True)  # crée le dossier parent si besoin

    existing_folders = [p for p in parent_path.iterdir() if p.is_dir() and p.name.isdigit()]
    
    existing_numbers = [int(p.name) for p in existing_folders]
    next_number = max(existing_numbers, default=0) + 1

    new_folder = parent_path / str(next_number)
    new_folder.mkdir()

    print(f"✔ Nouveau dossier créé : {new_folder}")
    return new_folder
