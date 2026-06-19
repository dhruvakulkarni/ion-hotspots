from dataclasses import dataclass
import numpy as np


# ============================================================
# ION EVENT
# ============================================================

@dataclass
class IonEvent:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    energy: float
    mass: float
    charge: float


# ============================================================
# BASE CLASS
# ============================================================

class IonSource:
    def sample(self, t_end: float):
        raise NotImplementedError


# ============================================================
# ALPHA EMITTER (Po-210-like)
# TRUE POISSON PROCESS
# ============================================================

@dataclass
class AlphaEmitter(IonSource):
    activity_Bq: float
    position: tuple = (0.0, 0.0, 0.0)
    energy_MeV: float = 5.3
    mass_amu: float = 4.0

    def __post_init__(self):
        self.pos0 = np.array(self.position, dtype=float)
        self.energy = self.energy_MeV * 1.602e-13  # MeV → Joules

    def sample(self, t_end: float):
        events = []
        t = 0.0

        # TRUE Poisson process: exponential inter-arrival times
        while True:
            dt = np.random.exponential(1.0 / self.activity_Bq)
            t += dt

            if t > t_end:
                break

            direction = self._isotropic_direction()
            v = self._velocity_from_energy(self.energy, self.mass_amu)

            events.append(
                IonEvent(
                    t=t,
                    pos=self.pos0.copy(),
                    vel=v * direction,
                    energy=self.energy,
                    mass=self.mass_amu * 1.66e-27
                )
            )

        return events

    @staticmethod
    def _isotropic_direction():
        phi = 2 * np.pi * np.random.rand()
        costheta = 2 * np.random.rand() - 1
        sintheta = np.sqrt(1 - costheta**2)

        return np.array([
            sintheta * np.cos(phi),
            sintheta * np.sin(phi),
            costheta
        ])

    @staticmethod
    def _velocity_from_energy(E, mass_amu):
        m = mass_amu * 1.66e-27
        return np.sqrt(2 * E / m)


# ============================================================
# GAUSSIAN ION BEAM
# TRUE POISSON PROCESS IN TIME
# ============================================================

@dataclass
class GaussianIonBeam(IonSource):
    current_A: float
    energy_eV: float
    mass_amu: float
    charge: int = 1

    direction: tuple = (0.0, 0.0, -1.0)
    origin: tuple = (0.0, 0.0, 0.0)

    sigma_x: float = 1e-6
    sigma_y: float = 1e-6
    angular_spread: float = 0.01

    def __post_init__(self):
        self.origin0 = np.array(self.origin, dtype=float)

        self.energy = self.energy_eV * 1.602e-19
        self.mass = self.mass_amu * 1.66e-27

        self.rate = self.current_A / (self.charge * 1.602e-19)

    def _beam_direction(self):
        theta = np.random.normal(0, self.angular_spread)
        phi = 2 * np.pi * np.random.rand()

        dx = theta * np.cos(phi)
        dy = theta * np.sin(phi)
        dz = -1.0

        v = np.array([dx, dy, dz])
        return self._normalize(v)

    @staticmethod
    def _velocity_from_energy(E, m):
        return np.sqrt(2 * E / m)

    @staticmethod
    def _normalize(v):
        return v / np.linalg.norm(v)

    def sample(self, t_end, surface_z=0.0, height=10e-9):

        events = []
        t = 0.0

        while True:

            t += np.random.exponential(1.0 / self.rate)

            if t > t_end:
                break

            x = np.random.normal(0, self.sigma_x)
            y = np.random.normal(0, self.sigma_y)

            pos = np.array([x, y, surface_z + height])

            direction = np.array([0.0, 0.0, -1.0])
            speed = self._velocity_from_energy(self.energy, self.mass)

            events.append(
                IonEvent(
                    t=t,
                    pos=pos,
                    vel=speed * direction,
                    energy=self.energy,
                    mass=self.mass,
                    charge=self.charge
                )
            )

        return events

    def _velocity_from_energy(self, E, m):
        return np.sqrt(2 * E / m)