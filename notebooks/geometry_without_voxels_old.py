from dataclasses import dataclass, field
import numpy as np
from material import Material


@dataclass
class Geometry:
    """
    Defines spatial domain of the sample.
    No material physics inside (only references Material).
    """

    # center of sample in space (m)
    location: tuple = (0.0, 0.0, 0.0)

    # sample dimensions (m)
    lx: float = 1e-6
    ly: float = 1e-6
    lz: float = 5e-8   # thickness (important for NbN films)

    material: Material = field(default_factory=Material)

    def __post_init__(self):
        self.location = np.array(self.location, dtype=float)

        # half extents
        self.hx = self.lx / 2.0
        self.hy = self.ly / 2.0
        self.hz = self.lz / 2.0

    # ========================================================
    # GEOMETRY QUERIES
    # ========================================================

    def contains(self, point):
        """Check if a point lies inside the sample."""
        p = np.array(point, dtype=float)
        r = p - self.location

        return (
            abs(r[0]) <= self.hx and
            abs(r[1]) <= self.hy and
            abs(r[2]) <= self.hz
        )

    def normalize(self, point):
        """Convert global → local coordinates."""
        return np.array(point, dtype=float) - self.location

    def denormalize(self, point):
        """Convert local → global coordinates."""
        return np.array(point, dtype=float) + self.location

    # ========================================================
    # GRID (for SRIM / hotspot fields)
    # ========================================================

    def create_grid(self, nx, ny, nz):
        """
        Create structured voxel grid for simulation.
        """
        x = np.linspace(-self.hx, self.hx, nx) + self.location[0]
        y = np.linspace(-self.hy, self.hy, ny) + self.location[1]
        z = np.linspace(-self.hz, self.hz, nz) + self.location[2]

        return np.meshgrid(x, y, z, indexing="ij")

    def create_mask(self, X, Y, Z):
        """
        Mask of points inside geometry.
        """
        return (
            (np.abs(X - self.location[0]) <= self.hx) &
            (np.abs(Y - self.location[1]) <= self.hy) &
            (np.abs(Z - self.location[2]) <= self.hz)
        )

    # ========================================================
    # SAMPLE PROPERTIES (geometry-only view)
    # ========================================================

    def volume(self):
        return self.lx * self.ly * self.lz

    def area_xy(self):
        return self.lx * self.ly

    def thickness(self):
        return self.lz