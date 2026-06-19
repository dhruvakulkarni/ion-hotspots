import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.graph_objects as go


# ============================================================
# SHARED MASK (PHYSICS CONSISTENCY)
# ============================================================

def material_mask(Z):
    """
    Only keep voxels inside material (z < 0).
    """
    return Z < 0


# ============================================================
# 1. VOXEL MASK PLOT (3D scatter)
# ============================================================

def plot_voxels(geometry, mask, title="Affected Voxels"):
    """
    Plot voxels where mask == True, restricted to material only.
    """

    X, Y, Z = geometry.voxel_centers()

    mat = material_mask(Z)
    mask = mask & mat

    xs = X[mask]
    ys = Y[mask]
    zs = Z[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, s=2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title + " (z < 0 only)")

    plt.tight_layout()
    plt.show()


# ============================================================
# 2. FIELD SLICE PLOT (2D)
# ============================================================

def plot_field_slice(geometry, field, axis="z", index=None,
                     title="Field Slice"):

    nx, ny, nz = field.shape
    X, Y, Z = geometry.voxel_centers()

    if axis == "z":

        if index is None:
            index = nz // 2

        slice_2d = field[:, :, index]

        # enforce material-only visibility
        mat = Z[:, :, index] < 0
        slice_2d = np.where(mat, slice_2d, np.nan)

        x = X[:, :, index]
        y = Y[:, :, index]

        plt.figure()
        plt.imshow(
            slice_2d.T,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()]
        )

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(title + " (z < 0 only)")
        plt.colorbar(label="Field")

        plt.tight_layout()
        plt.show()


    elif axis == "x":

        if index is None:
            index = nx // 2

        slice_2d = field[index, :, :]
        y = Y[index, :, :]
        z = Z[index, :, :]

        mat = z < 0
        slice_2d = np.where(mat, slice_2d, np.nan)

        plt.figure()
        plt.imshow(
            slice_2d.T,
            origin="lower",
            aspect="auto",
            extent=[y.min(), y.max(), z.min(), z.max()]
        )

        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.title(title + " (z < 0 only)")
        plt.colorbar(label="Field")

        plt.tight_layout()
        plt.show()


    elif axis == "y":

        if index is None:
            index = ny // 2

        slice_2d = field[:, index, :]
        x = X[:, index, :]
        z = Z[:, index, :]

        mat = z < 0
        slice_2d = np.where(mat, slice_2d, np.nan)

        plt.figure()
        plt.imshow(
            slice_2d.T,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), z.min(), z.max()]
        )

        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.title(title + " (z < 0 only)")
        plt.colorbar(label="Field")

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


# ============================================================
# 3. FIELD HISTOGRAM
# ============================================================

def plot_field_hist(field, title="Field Distribution",label='Field'):
    plt.figure()
    plt.hist(field.ravel(), bins=100)
    plt.xlabel(f"{label} value")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. 3D VOXEL HEATMAP (MATPLOTLIB)
# ============================================================

def plot_field_3d_heatmap(geometry, field, threshold=0.0, cmap="inferno"):

    X, Y, Z = geometry.voxel_centers()

    filled = (field > threshold) & (Z < 0)

    norm_field = field / (field.max() + 1e-30)

    colors = np.zeros(filled.shape + (4,))
    cmap_fn = cm.get_cmap(cmap)

    nx, ny, nz = field.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                if filled[i, j, k]:
                    colors[i, j, k] = cmap_fn(norm_field[i, j, k])
                else:
                    colors[i, j, k] = (0, 0, 0, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(filled, facecolors=colors, edgecolor=None)

    ax.set_title("3D Energy Deposition (material only)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. 3D INTERACTIVE VOXEL PLOT (PLOTLY)
# ============================================================

def plot_field_3d_interactive(geometry, field, threshold=0.0,label='energy'):

    X, Y, Z = geometry.voxel_centers()

    mask = (field > threshold) & (Z < 0)
    
    ftotal = np.sum(field[mask])

    fmax = field.max() + 1e-30
    norm = field / fmax

    xs = X[mask]
    ys = Y[mask]
    zs = Z[mask]
    cs = norm[mask]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(
                    size=3,
                    color=cs,
                    colorscale="Inferno",
                    opacity=0.8,
                    colorbar=dict(title="Energy norm max=1"),
                ),
            )
        ]
    )
    _title=''
    if label=='energy':
        _title=f"Interactive 3D Energy Deposition (material only) (ftotal={ftotal/1.602e-19} eV)"
    elif label=='temp':
        _title=f"Interactive 3D Temperature Profile (material only) (fmax={fmax} K)"
    else:
        _title=f"Interactive 3D Energy Deposition (material only)"

    fig.update_layout(
        title=_title,
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


# ============================================================
# 6. VOXEL CUBE RENDER (MATPLOTLIB)
# ============================================================

def plot_voxel_cells(geometry, field, threshold=0.0, cmap="inferno"):

    X, Y, Z = geometry.voxel_centers()

    filled = (field > threshold) & (Z < 0)

    norm = field / (field.max() + 1e-30)

    colors = np.empty(filled.shape + (4,), dtype=float)
    cmap_fn = cm.get_cmap(cmap)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(filled, facecolors=colors, edgecolor="k", linewidth=0.2)

    ax.set_title("Voxel Cells (material only)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. VOXEL CUBE RENDER (PLOTLY)
# ============================================================

def plot_field_3d_cells_plotly(geometry, field, threshold=0.0):

    X, Y, Z = geometry.voxel_centers()

    mask = (field > threshold) & (Z < 0)

    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    fmax = field.max() + 1e-30
    norm = field / fmax

    import matplotlib.cm as cm
    cmap_fn = cm.get_cmap("viridis_r")

    fig = go.Figure()

    def add_cube(xc, yc, zc, color):

        x = [xc - hx, xc + hx, xc + hx, xc - hx,
             xc - hx, xc + hx, xc + hx, xc - hx]

        y = [yc - hy, yc - hy, yc + hy, yc + hy,
             yc - hy, yc - hy, yc + hy, yc + hy]

        z = [zc - hz, zc - hz, zc - hz, zc - hz,
             zc + hz, zc + hz, zc + hz, zc + hz]

        i = [0, 0, 0, 4, 4, 1, 2, 3, 5, 6, 7, 2]
        j = [1, 2, 3, 5, 6, 2, 3, 7, 6, 7, 4, 6]
        k = [2, 3, 1, 6, 7, 6, 7, 4, 5, 4, 5, 7]

        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color=color,
            opacity=0.6,
            showscale=False
        ))

    nx, ny, nz = field.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                if not mask[i, j, k]:
                    continue

                xc, yc, zc = X[i, j, k], Y[i, j, k], Z[i, j, k]

                c = cmap_fn(norm[i, j, k])
                rgba = f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]})"

                add_cube(xc, yc, zc, rgba)

    fig.update_layout(
        title="3D Voxel Cells (material only)",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()