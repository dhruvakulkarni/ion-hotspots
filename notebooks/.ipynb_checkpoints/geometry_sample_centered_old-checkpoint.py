# geometry.py

from dataclasses import dataclass, field
import numpy as np

from material import Material


@dataclass
class Geometry:
    """
    Defines simulation geometry + voxelized mesh.
    """

    # ========================================================
    # SAMPLE POSITION
    # ========================================================

    location: tuple = (0.0, 0.0, 0.0)

    # ========================================================
    # SAMPLE DIMENSIONS (meters)
    # ========================================================

    lx: float = 1e-6
    ly: float = 1e-6
    lz: float = 200e-9

    # ========================================================
    # VOXEL RESOLUTION
    # ========================================================

    nx: int = 20
    ny: int = 20
    nz: int = 10

    # ========================================================
    # MATERIAL
    # ========================================================

    material: Material = field(default_factory=Material)

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __post_init__(self):

        self.location = np.array(
            self.location,
            dtype=float
        )

        # half dimensions
        self.hx = self.lx / 2
        self.hy = self.ly / 2
        self.hz = self.lz / 2

        # voxel spacing
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.dz = self.lz / self.nz

    # ========================================================
    # GEOMETRY TESTS
    # ========================================================

    def contains(self, point):
        """
        Check if point lies inside sample.
        """

        p = np.array(point, dtype=float)
        r = p - self.location

        return (
            abs(r[0]) <= self.hx and
            abs(r[1]) <= self.hy and
            abs(r[2]) <= self.hz
        )

    # ========================================================
    # COORDINATE TRANSFORMS
    # ========================================================

    def normalize(self, point):
        """
        Global -> local coordinates.
        """
        return np.array(point) - self.location

    def denormalize(self, point):
        """
        Local -> global coordinates.
        """
        return np.array(point) + self.location

    # ========================================================
    # VOXEL INDEXING
    # ========================================================

    def point_to_voxel(self, point):
        """
        Convert physical position -> voxel index.
        """

        p = self.normalize(point)

        x = p[0] + self.hx
        y = p[1] + self.hy
        z = p[2] + self.hz

        i = int(x / self.dx)
        j = int(y / self.dy)
        k = int(z / self.dz)

        # bounds check
        if (
            i < 0 or i >= self.nx or
            j < 0 or j >= self.ny or
            k < 0 or k >= self.nz
        ):
            return None

        return (i, j, k)

    def voxel_center(self, i, j, k):
        """
        Center coordinate of voxel.
        """

        x = (
            -self.hx +
            (i + 0.5) * self.dx
        )

        y = (
            -self.hy +
            (j + 0.5) * self.dy
        )

        z = (
            -self.hz +
            (k + 0.5) * self.dz
        )

        return self.denormalize([x, y, z])

    def voxel_bounds(self, i, j, k):
        """
        Physical bounds of voxel.
        """

        x0 = -self.hx + i * self.dx
        x1 = x0 + self.dx

        y0 = -self.hy + j * self.dy
        y1 = y0 + self.dy

        z0 = -self.hz + k * self.dz
        z1 = z0 + self.dz

        return (
            self.denormalize([x0, y0, z0]),
            self.denormalize([x1, y1, z1])
        )

    # ========================================================
    # VOXEL ARRAYS
    # ========================================================

    def create_field(self, value=0.0):
        """
        Create voxelized scalar field.
        """

        return np.full(
            (self.nx, self.ny, self.nz),
            value,
            dtype=float
        )

    def voxel_volume(self):
        """
        Volume of single voxel.
        """

        return (
            self.dx *
            self.dy *
            self.dz
        )

    # ========================================================
    # GRID COORDINATES
    # ========================================================

    def voxel_centers(self):
        """
        Return meshgrid of voxel centers.
        """

        x = np.linspace(
            -self.hx + self.dx/2,
             self.hx - self.dx/2,
             self.nx
        )

        y = np.linspace(
            -self.hy + self.dy/2,
             self.hy - self.dy/2,
             self.ny
        )

        z = np.linspace(
            -self.hz + self.dz/2,
             self.hz - self.dz/2,
             self.nz
        )

        X, Y, Z = np.meshgrid(
            x, y, z,
            indexing="ij"
        )

        return (
            X + self.location[0],
            Y + self.location[1],
            Z + self.location[2]
        )

    # ========================================================
    # INFO
    # ========================================================

    def volume(self):
        return (
            self.lx *
            self.ly *
            self.lz
        )

    def shape(self):
        return (
            self.nx,
            self.ny,
            self.nz
        )

    def spacing(self):
        return (
            self.dx,
            self.dy,
            self.dz
        )

    def summary(self):

        return {
            "size_m": (
                self.lx,
                self.ly,
                self.lz
            ),

            "voxels": (
                self.nx,
                self.ny,
                self.nz
            ),

            "spacing_m": (
                self.dx,
                self.dy,
                self.dz
            ),

            "voxel_volume_m3":
                self.voxel_volume(),

            "material":
                self.material.name
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

        srim : SRIMOutput
            Contains range_projected
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