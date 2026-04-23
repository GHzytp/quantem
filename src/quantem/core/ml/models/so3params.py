import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tilted KPlanes ---

# ---------------------------------------------------------------------------
# SO(3) quaternion parameter module
# ---------------------------------------------------------------------------
 
class SO3ParamQuat(nn.Module):
    """
    Stores T unit quaternions as learnable parameters in R^4 and normalises
    them on every call to `as_matrix()`.
 
    Quaternion convention: [x, y, z, w]  (scalar-last, same as scipy).
 
    Initialisation
    --------------
    "random"  – uniform sampling over SO(3) via Shoemake's method.
    "identity" – all rotations start as the identity (good for fine-tuning).
    """
 
    def __init__(self, T: int, init: str = "random"):
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        quats = self._init_quaternions(T, init)   # (T, 4)
        self.quats = nn.Parameter(quats)
 
    # ------------------------------------------------------------------
    # Initialisers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _shoemake_sample(T: int) -> torch.Tensor:
        """Uniform SO(3) sampling via Shoemake (1992). Returns (T, 4) [x,y,z,w]."""
        u = torch.rand(T, 3)
        sqrt1_u0 = torch.sqrt(1.0 - u[:, 0])
        sqrt_u0  = torch.sqrt(u[:, 0])
        two_pi   = 2.0 * math.pi
        x = sqrt1_u0 * torch.sin(two_pi * u[:, 1])
        y = sqrt1_u0 * torch.cos(two_pi * u[:, 1])
        z = sqrt_u0  * torch.sin(two_pi * u[:, 2])
        w = sqrt_u0  * torch.cos(two_pi * u[:, 2])
        return torch.stack([x, y, z, w], dim=-1)   # (T, 4)
 
    @staticmethod
    def _identity(T: int) -> torch.Tensor:
        """All-identity rotations: [0,0,0,1] * T."""
        q = torch.zeros(T, 4)
        q[:, 3] = 1.0
        return q
 
    @classmethod
    def _init_quaternions(cls, T: int, init: str) -> torch.Tensor:
        if init == "random":
            return cls._shoemake_sample(T)
        elif init == "identity":
            return cls._identity(T)
        else:
            raise ValueError(f"Unknown init '{init}'; choose 'random' or 'identity'.")
 
    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
 
    def normalized(self) -> torch.Tensor:
        """Returns (T, 4) unit quaternions."""
        return F.normalize(self.quats, p=2, dim=-1)
 
    def as_matrix(self) -> torch.Tensor:
        """
        Converts the T stored quaternions to (T, 3, 3) rotation matrices.
 
        Uses the standard formula; no trig, just multiplications.
        """
        q = self.normalized()          # (T, 4)  [x, y, z, w]
        x, y, z, w = q.unbind(dim=-1)  # each (T,)
 
        # Precompute products
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
 
        # Row-major: R[i,j]
        R = torch.stack([
            1 - 2*(yy + zz),   2*(xy - wz),       2*(xz + wy),
              2*(xy + wz),    1 - 2*(xx + zz),     2*(yz - wx),
              2*(xz - wy),      2*(yz + wx),      1 - 2*(xx + yy),
        ], dim=-1).reshape(-1, 3, 3)   # (T, 3, 3)
 
        return R
 
    def extra_repr(self) -> str:
        return f"T={self.quats.shape[0]}"


class SO3ParamR9SVD(nn.Module):
    """
    SO(3) rotation bank using R9+SVD parameterization.
    Each rotation is stored as an unconstrained 3x3 matrix M,
    projected to SO(3) via SVD+(M) = U diag(1,1,det(UVt)) Vt.
    """

    def __init__(self, T: int, init: str = "random"):
        super().__init__()
        if init == "random":
            # Initialize near identity with small noise
            M = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
            M = M + 0.1 * torch.randn(T, 3, 3)
        elif init == "identity":
            M = torch.eye(3).unsqueeze(0).repeat(T, 1, 1)
        else:
            raise ValueError(f"Unknown init '{init}'")
        self.M = nn.Parameter(M)   # (T, 3, 3)

    def as_matrix(self) -> torch.Tensor:
        """Projects each M to SO(3) via SVD. Returns (T, 3, 3)."""

        
        U, _, Vh = torch.linalg.svd(self.M)   # U: (T,3,3), Vh: (T,3,3)
        # Fix reflections: det(U Vh) must be +1
        d = torch.det(U @ Vh)                  # (T,)
        diag = torch.ones(self.M.shape[0], 3, device=self.M.device, dtype=self.M.dtype)
        diag[:, 2] = d                          # multiply last singular vector by sign
        return U @ (diag.unsqueeze(-1) * Vh)   # (T, 3, 3)

