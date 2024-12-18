import numpy as np
from typing import Optional

class Zigzag:
    _pattern: Optional[np.ndarray] = None

    @classmethod
    def set_zigzag(cls, n: int) -> None:
        """Pre-compute zigzag scan pattern for nxn blocks"""
        cls._pattern = cls._create_zigzag_pattern(n)

    @classmethod
    def get_zigzag(cls) -> Optional[np.ndarray]:
        """Get the pre-computed zigzag pattern"""
        return cls._pattern

    @staticmethod
    def _create_zigzag_pattern(n: int) -> np.ndarray:
        """Create zigzag pattern indices for nxn matrix"""
        x_idxs = []
        y_idxs = []
        
        # Generate zigzag pattern indices
        # upper left triangle
        for i in range(1, n, 2):
            y_idxs.extend(list(range(i+1)) + list(range(i-1, -1, -1)))
        for i in range(0, n, 2):
            x_idxs.extend(list(range(i+1)) + list(range(i-1, -1, -1)))
        
        # down right triangle
        for i in range(1, n, 2):
            y_idxs.extend(list(range(i, n)) + list(range(n-1, i, -1)))
        for i in range(0, n, 2):
            x_idxs.extend(list(range(i, n)) + list(range(n-1, i, -1)))

        return np.column_stack((x_idxs, y_idxs))
