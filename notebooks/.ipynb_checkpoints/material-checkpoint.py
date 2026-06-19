from dataclasses import dataclass, field
from typing import Optional

from srim_output import SRIMOutput


@dataclass
class Material:
    """
    Material properties for thermal / hotspot / SRIM simulations.
    """

    name: str = "NbN"

    # ========================================================
    # BASIC MATERIAL PROPERTIES
    # ========================================================

    # mass density (kg/m^3)
    density: float = 8400.0

    # volumetric heat capacity (J/m^3/K)
    cv: float = 2.64e6

    # thermal conductivity (W/m/K)
    thermal_conductivity: float = 10.0

    # superconducting critical temperature (K)
    Tc: float = 9.0

    # electron-phonon coupling (W/m^3/K)
    g_ep: float = 1e17 #if we do the TTM: Ce dt/dTe=−g_ep(Te−Tp)

    # ========================================================
    # SRIM RESPONSE
    # ========================================================

    srim: Optional[SRIMOutput] = None

    # empirical scaling factor for SRIM stopping
    stopping_power_scale: float = 1.0

    # empirical defect/damage scaling
    damage_efficiency: float = 1.0
    
    # ========================================================
    # FREE PARAMETERS
    # ========================================================
    
    cylinder_model_radius: float = 5e-9

    # ========================================================
    # SRIM HELPERS
    # ========================================================

    def has_srim(self):
        return self.srim is not None

    @property
    def dEdx_total(self):
        if self.srim is None:
            return 0.0

        return (
            self.srim.dEdx_total *
            self.stopping_power_scale
        )

    @property
    def electronic_fraction(self):
        if self.srim is None:
            return 0.0

        return self.srim.electronic_fraction

    @property
    def nuclear_fraction(self):
        if self.srim is None:
            return 0.0

        return self.srim.nuclear_fraction

    def energy_partition(self, E):
        """
        Partition deposited ion energy into:
        - electronic channel
        - nuclear channel
        """
        if self.srim is None:
            return E, 0.0

        return self.srim.energy_partition(E)

    # ========================================================
    # HOTSPOT / THERMAL HELPERS
    # ========================================================

    def energy_to_temperature(self, E, volume):
        """
        Convert deposited energy into temperature rise.
        """
        return E / (self.cv * volume + 1e-30)

    def thermal_diffusivity(self):
        """
        Thermal diffusivity:
            alpha = k / Cv
        """
        return (
            self.thermal_conductivity /
            (self.cv + 1e-30)
        )