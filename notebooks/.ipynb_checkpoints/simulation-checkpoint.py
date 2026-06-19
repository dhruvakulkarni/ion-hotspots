import numpy as np

def deposit_event_to_geometry_nonumericcheck(event, X, Y, Z, voxel_volume, srim):
    """
    One ion → voxel energy field (FAST version)
    """

    x0, y0, z0 = event.pos

    Rp = srim.range_projected
    sigma_z = srim.straggle_long
    sigma_xy = srim.straggle_trans

    dx2 = (X - x0)**2
    dy2 = (Y - y0)**2
    dz2 = (Z - (z0 + Rp))**2

    E = np.exp(-(dx2 + dy2) / (2 * sigma_xy**2)) \
        * np.exp(-dz2 / (2 * sigma_z**2))

    # energy normalization
    E *= event.energy / np.sum(E * voxel_volume)

    return E


def deposit_event_to_geometry_before_refactor(event, X, Y, Z, voxel_volume, srim):
    """
    One ion → voxel energy field (numerically safe FAST version)
    """

    x0, y0, z0 = event.pos

    Rp = srim.range_projected
    sigma_z = srim.straggle_long
    sigma_xy = srim.straggle_trans

    # avoid division by zero / nonsense widths
    sigma_xy = max(sigma_xy, 1e-12)
    sigma_z  = max(sigma_z, 1e-12)

    dx2 = (X - x0)**2
    dy2 = (Y - y0)**2
    dz2 = (Z - (z0 + Rp))**2

    E = np.exp(-(dx2 + dy2) / (2.0 * sigma_xy**2)) \
      * np.exp(-dz2 / (2.0 * sigma_z**2))

    # compute normalization integral
    S = np.sum(E * voxel_volume)

    # --- safety checks ---
    if not np.isfinite(S) or S < 1e-30:
        print(f"Gaussian is effectively outside domain,no reliable deposition inside grid")
        return np.zeros_like(E)

    # normalize to deposited energy
    E *= event.energy / S

    # final safety: remove NaNs/Infs if any
    E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)
    print(E)

    return E

def deposit_event_to_geometry(geometry,event, X, Y, Z, voxel_volume, srim):

    x0, y0, z0 = event.pos

    # SRIM center shift
    #cx, cy, cz = x0, y0, z0 - srim.range_projected
    cx, cy, cz = geometry.deposition_center(event.pos, event.vel, srim)

    sigma_xy = max(srim.straggle_trans, 1e-12)
    sigma_z  = max(srim.straggle_long, 1e-12)

    dx2 = (X - cx)**2
    dy2 = (Y - cy)**2
    dz2 = (Z - cz)**2

    E = np.exp(-(dx2 + dy2) / (2 * sigma_xy**2)) \
      * np.exp(-(dz2) / (2 * sigma_z**2))

    S = np.sum(E * voxel_volume)

    if not np.isfinite(S) or S < 1e-30:
        print(f"S:{S}")
        print(f"Gaussian is effectively outside domain,no reliable deposition inside grid")
        
        print(f"event.pos = {event.pos}")
        print(f"cx,cy,cz: {cx},{cy},{cz}")
        return np.zeros_like(E)

    E *= event.energy / S

    return E

import numpy as np

def deposit_event_cylinder(event,geometry,srim,material):
    """
    Find voxels intersecting a stochastic ion cylinder
    and deposit energy uniformly.

    Cylinder:
    - axis: event direction
    - height: sampled stopping range
    - radius: srim.straggle_trans
    """

    x0, y0, z0 = event.pos

    # ============================================================
    # 1. direction
    # ============================================================
    d = event.vel / np.linalg.norm(event.vel)

    # ============================================================
    # 2. sample stopping range
    # ============================================================
    R = np.random.normal(
        srim.range_projected,
        srim.straggle_long
    )
    R = max(R, 0.0)
    R = min(R, geometry.lz)

    # cylinder radius
    r_cyl = max(material.cylinder_model_radius,geometry.dx/2) #estimating from srim.straggle_trans is just huge 
    print(r_cyl)

    # ============================================================
    # 3. voxel grid
    # ============================================================
    X, Y, Z = geometry.voxel_centers()

    # shift coordinates into ion frame
    dx = X - x0
    dy = Y - y0
    dz = Z - z0

    # projection along track
    s = dx * d[0] + dy * d[1] + dz * d[2]

    # reject voxels outside cylinder height
    mask_long = (s >= 0) & (s <= R)

    # radial distance squared from axis
    rx = dx - s * d[0]
    ry = dy - s * d[1]
    rz = dz - s * d[2]

    r2 = rx**2 + ry**2 + rz**2

    mask_radial = r2 <= r_cyl**2

    # ============================================================
    # 4. final voxel mask
    # ============================================================
    mask = mask_long & mask_radial

    # ============================================================
    # 5. energy deposition
    # ============================================================
    E = np.zeros_like(X)

    voxels = np.sum(mask)

    if voxels == 0:
        return E

    E[mask] = R*srim.dEdx_electronic / voxels

    return E

def prepare_geometry_cache(geometry):
    """
    Precompute voxel grid once for performance.
    """

    X, Y, Z = geometry.voxel_centers()
    V = geometry.voxel_volume()

    return X, Y, Z, V

def run_simulation_gaussian_energy_old(ion_source, geometry, srim, t_end,max_events=0):

    field = geometry.create_field(0.0)

    # cache geometry ONCE
    X, Y, Z, V = prepare_geometry_cache(geometry)

    events = ion_source.sample(t_end)

    print(f"Generated {len(events)} ion events")
    if max_events==0:
        max_events=len(events)
    for event in events[:max_events]:
        field += deposit_event_to_geometry(geometry,event, X, Y, Z, V, srim)

    return field

def run_simulation(ion_source, geometry, material,srim, t_end, max_events=0):

    # ============================================================
    # 1. field initialization
    # ============================================================
    energy = geometry.create_field(0.0)

    # ============================================================
    # 2. cache geometry ONCE
    # ============================================================
    geometry.build_voxel_cache()  # X, Y, Z cached here

    # voxel volume cache (important for consistency)
    voxel_volume = geometry.voxel_volume()

    # ============================================================
    # 3. generate ion events
    # ============================================================
    events = ion_source.sample(t_end)

    print(f"Generated {len(events)} ion events")

    if max_events == 0:
        max_events = len(events)

    # ============================================================
    # 4. deposition loop (CYLINDER MODEL)
    # ============================================================
    for event in events[:max_events]:

        dE = deposit_event_cylinder(
            event=event,
            geometry=geometry,
            srim=srim,
            material=material
        )

        energy += dE
    
    T = material.Tc + energy/(material.cv*voxel_volume)

    return energy,T