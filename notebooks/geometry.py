from dataclasses import dataclass, field
import numpy as np
from material import Material


@dataclass
class Geometry:
    """
    Surface-referenced voxel geometry.

    Convention:
        z = 0     → surface
        z < 0     → material
        z > 0     → vacuum
    """

    # ========================================================
    # SAMPLE DIMENSIONS (material extends into -z)
    # ========================================================

    lx: float = 1e-6
    ly: float = 1e-6
    lz: float = 200e-9

    nx: int = 20
    ny: int = 20
    nz: int = 10

    material: Material = field(default_factory=Material)

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __post_init__(self):

        # voxel spacing
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.dz = self.lz / self.nz

        # -------------------------
        # SURFACE IS FIXED AT z = 0
        # -------------------------

        self.surface_z = 0.0
        self.z_top = 0.0
        self.z_bottom = -self.lz

        # -------------------------
        # lateral centering (beam axis)
        # -------------------------

        self.x0 = -self.lx / 2
        self.y0 = -self.ly / 2
        print(self.dx,self.dy,self.dz)

    # ========================================================
    # GEOMETRY TESTS
    # ========================================================

    def contains(self, point):
        x, y, z = point

        return (
            self.x0 <= x <= self.x0 + self.lx and
            self.y0 <= y <= self.y0 + self.ly and
            self.z_bottom <= z <= self.z_top
        )

    # ========================================================
    # VOXEL CENTER
    # ========================================================

    def voxel_center(self, i, j, k):

        x = self.x0 + (i + 0.5) * self.dx
        y = self.y0 + (j + 0.5) * self.dy
        z = self.z_bottom + (k + 0.5) * self.dz

        return np.array([x, y, z])

    # ========================================================
    # VOXEL INDEXING
    # ========================================================

    def point_to_voxel(self, point):

        x, y, z = point

        i = int((x - self.x0) / self.dx)
        j = int((y - self.y0) / self.dy)
        k = int((z - self.z_bottom) / self.dz)

        if (
            i < 0 or i >= self.nx or
            j < 0 or j >= self.ny or
            k < 0 or k >= self.nz
        ):
            return None

        return (i, j, k)

    # ========================================================
    # VOXEL BOUNDS
    # ========================================================

    def voxel_bounds(self, i, j, k):

        x0 = self.x0 + i * self.dx
        x1 = x0 + self.dx

        y0 = self.y0 + j * self.dy
        y1 = y0 + self.dy

        z0 = self.z_bottom + k * self.dz
        z1 = z0 + self.dz

        return (
            np.array([x0, y0, z0]),
            np.array([x1, y1, z1])
        )

    # ========================================================
    # FIELD CREATION
    # ========================================================

    def create_field(self, value=0.0):
        return np.full((self.nx, self.ny, self.nz), value, dtype=float)

    def voxel_volume(self):
        return self.dx * self.dy * self.dz

    # ========================================================
    # GRID COORDINATES
    # ========================================================

    def voxel_centers(self):

        x = self.x0 + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.dy
        z = self.z_bottom + (np.arange(self.nz) + 0.5) * self.dz

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        return X, Y, Z
    
    def build_voxel_cache(self):
        """
        Precompute voxel coordinate grid once for fast deposition.
        """

        # voxel center grid
        self.X, self.Y, self.Z = self.voxel_centers()

        # voxel volume (scalar, reused everywhere)
        self.dV = self.dx * self.dy * self.dz

        # shape sanity check
        self.shape_cached = self.X.shape

        return self

    # ========================================================
    # INFO
    # ========================================================

    def volume(self):
        return self.lx * self.ly * self.lz

    def shape(self):
        return (self.nx, self.ny, self.nz)

    def spacing(self):
        return (self.dx, self.dy, self.dz)
    

    def summary(self):

        return {
            "coordinate_system": "surface-referenced",
            "surface_z": self.surface_z,
            "z_convention": "0=surface, negative=material",

            "size_m": (self.lx, self.ly, self.lz),
            "voxels": (self.nx, self.ny, self.nz),
            "spacing_m": (self.dx, self.dy, self.dz),

            "voxel_volume_m3": self.voxel_volume(),
            "material": self.material.name
        }
    
    def deposition_center(self, event_pos, event_vel, srim):
        """
        Returns SRIM deposition center in GLOBAL coordinates.

        Parameters
        ----------
        event_pos : (3,) array
            Ion entry position in global coordinates (in vacuum near surface)

        event_vel : (3,) array
            Ion velocity vector (defines direction)

        srim : SRIMModel
            Contains Rp (projected range)

        Returns
        -------
        (x, y, z) : deposition center in global coordinates
        """

        r0 = np.array(event_pos, dtype=float)

        v = np.array(event_vel, dtype=float)

        # normalize direction
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError("Zero velocity vector in event")

        direction = v / v_norm

        # SRIM shift along trajectory
        r_dep = r0 + srim.range_projected * direction

        return r_dep