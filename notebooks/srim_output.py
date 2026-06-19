from dataclasses import dataclass
import numpy as np



@dataclass
class SRIMOutput:
    """
    SRIM-derived stopping and range information
    for a specific ion/material combination.
    """

    ion_name: str = "Ar+"
    ion_energy_eV: float = 1000.0
    ion_mass_amu: float = 40.0

    # stopping powers (eV/A)
    dEdx_electronic_eVperA: float = 0.0
    dEdx_nuclear_eVperA: float = 0.0

    # projected range statistics (m)
    range_projected: float = 0.0
    straggle_long: float = 0.0
    straggle_trans: float = 0.0

    # ========================================================
    # DERIVED STOPPING QUANTITIES
    # ========================================================

    def __post_init__(self):
        eV_per_A_to_J_per_m = 1.602176634e-9
        self.dEdx_electronic = self.dEdx_electronic_eVperA * eV_per_A_to_J_per_m
        self.dEdx_nuclear = self.dEdx_nuclear_eVperA * eV_per_A_to_J_per_m
        
        
    @property
    def dEdx_total(self):
        return self.dEdx_electronic + self.dEdx_nuclear

    @property
    def electronic_fraction(self):
        return (
            self.dEdx_electronic /
            (self.dEdx_total + 1e-30)
        )

    @property
    def nuclear_fraction(self):
        return (
            self.dEdx_nuclear /
            (self.dEdx_total + 1e-30)
        )

    # ========================================================
    # ENERGY PARTITIONING
    # ========================================================

    def energy_partition(self, E):
        """
        Partition deposited ion energy into:
        - electronic energy
        - nuclear energy
        """
        return (
            E * self.electronic_fraction,
            E * self.nuclear_fraction
        )

    # ========================================================
    # RANGE / DEPOSITION MODEL
    # ========================================================

    def stopping_profile_weight(self, s, Rp=None):
        """
        Gaussian approximation to longitudinal
        SRIM stopping profile.
        """
        if Rp is None:
            Rp = self.range_projected

        sigma = self.straggle_long + 1e-30

        return np.exp(
            -(s - Rp) ** 2 / (2 * sigma ** 2)
        )
    

    # ========================================================
    # QUICK SUMMARY
    # ========================================================

    def summary(self):
        return {
            "ion": self.ion_name,
            "E_eV": self.ion_energy_eV,
            "range_m": self.range_projected,
            "dEdx_electronic": self.dEdx_electronic,
            "dEdx_nuclear": self.dEdx_nuclear,
            "dEdx_total": self.dEdx_total,
            "electronic_fraction": self.electronic_fraction,
            "nuclear_fraction": self.nuclear_fraction,
        }