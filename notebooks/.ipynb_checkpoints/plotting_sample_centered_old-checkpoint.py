import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
import plotly.graph_objects as go

# ============================================================
# 1. VOXEL MASK PLOT (3D scatter)
# ============================================================

def plot_voxels(geometry, mask, title="Affected Voxels"):
    """
    Plot voxels where mask == True
    """

    X, Y, Z = geometry.voxel_centers()

    xs = X[mask]
    ys = Y[mask]
    zs = Z[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs, ys, zs, s=2)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


# ============================================================
# 2. FIELD SLICE PLOT (2D)
# ============================================================

def plot_field_slice(geometry, field, axis="z", index=None,
                     title="Field Slice"):
    """
    Plot a 2D slice of a 3D voxel field
    """

    nx, ny, nz = field.shape
    X, Y, Z = geometry.voxel_centers()

    if axis == "z":
        if index is None:
            index = nz // 2

        slice_2d = field[:, :, index]

        x = X[:, :, index]
        y = Y[:, :, index]

        plt.figure()
        plt.imshow(
            slice_2d.T,
            origin="lower",
            aspect="auto",
            extent=[
                x.min(), x.max(),
                y.min(), y.max()
            ]
        )

        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(title)
        plt.colorbar(label="Field")

        plt.tight_layout()
        plt.show()


    elif axis == "x":
        if index is None:
            index = nx // 2

        slice_2d = field[index, :, :]

        y = Y[index, :, :]
        z = Z[index, :, :]

        plt.figure()
        plt.imshow(
            slice_2d.T,
            origin="lower",
            aspect="auto",
            extent=[
                y.min(), y.max(),
                z.min(), z.max()
            ]
        )

        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.title(title)
        plt.colorbar(label="Field")

        plt.tight_layout()
        plt.show()


    elif axis == "y":
        if index is None:
            index = ny // 2

        slice_2d = field[:, index, :]

        x = X[:, index, :]
        z = Z[:, index, :]

        plt.figure()
        plt.imshow(
            slice_2d.T,
            origin="lower",
            aspect="auto",
            extent=[
                x.min(), x.max(),
                z.min(), z.max()
            ]
        )

        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.title(title)
        plt.colorbar(label="Field")

        plt.tight_layout()
        plt.show()


    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


# ============================================================
# 3. FIELD STATISTICS (DEBUG TOOL)
# ============================================================

def plot_field_hist(field, title="Field Distribution"):
    """
    Histogram of voxel field values
    """

    plt.figure()
    plt.hist(field.ravel(), bins=100)

    plt.xlabel("Field value")
    plt.ylabel("Count")
    plt.title(title)

    plt.tight_layout()
    plt.show()
    



def plot_field_3d_heatmap(geometry, field, threshold=0.0, cmap="inferno"):
    """
    3D voxel heatmap of scalar field.

    Parameters
    ----------
    geometry : Geometry
    field    : (nx, ny, nz) array
    threshold: float, mask cutoff
    cmap     : colormap
    """

    nx, ny, nz = field.shape

    # Normalize field for colormap
    norm_field = field / (field.max() + 1e-30)

    # Boolean mask of active voxels
    filled = field > threshold

    # Build voxel coordinate grid (cell edges)
    x = np.arange(nx + 1)
    y = np.arange(ny + 1)
    z = np.arange(nz + 1)

    # Color mapping per voxel
    colors = np.zeros(filled.shape + (4,))

    cmap_fn = cm.get_cmap(cmap)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if filled[i, j, k]:
                    colors[i, j, k] = cmap_fn(norm_field[i, j, k])
                else:
                    colors[i, j, k] = (0, 0, 0, 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(
        filled,
        facecolors=colors,
        edgecolor=None
    )

    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    ax.set_zlabel("z voxel")
    ax.set_title("3D Energy Deposition Heatmap")

    plt.tight_layout()
    plt.show()
    
    


def plot_field_3d_interactive(geometry, field, threshold=0.0):
    """
    Interactive 3D voxel heatmap using Plotly.
    """

    nx, ny, nz = field.shape

    X, Y, Z = geometry.voxel_centers()

    # Normalize field for coloring
    f = field.copy()
    fmax = f.max() + 1e-30
    norm = f / fmax

    # Extract active voxels
    mask = field > threshold

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
                    colorbar=dict(title="Normalized Energy"),
                ),
            )
        ]
    )

    fig.update_layout(
        title="Interactive 3D SRIM Energy Deposition",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()
    
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_voxel_cells(geometry, field, threshold=0.0, cmap="inferno"):
    """
    Render voxel grid as 3D cubes with scalar coloring.

    Each voxel is shown as a cell (not a point).
    """

    nx, ny, nz = field.shape

    # Boolean mask of active voxels
    filled = field > threshold

    # Normalize field for colormap
    norm = field / (field.max() + 1e-30)

    # Build color array (RGBA per voxel)
    colors = np.empty(filled.shape + (4,), dtype=float)

    cmap_fn = cm.get_cmap(cmap)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                if filled[i, j, k]:
                    colors[i, j, k] = cmap_fn(norm[i, j, k])
                else:
                    colors[i, j, k] = (0, 0, 0, 0)  # transparent

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(
        filled,
        facecolors=colors,
        edgecolor="k",
        linewidth=0.2
    )

    ax.set_xlabel("x voxel index")
    ax.set_ylabel("y voxel index")
    ax.set_zlabel("z voxel index")
    ax.set_title("3D Voxel Cell Heatmap")

    plt.tight_layout()
    plt.show()
    
    
import numpy as np
import plotly.graph_objects as go


def plot_field_3d_cells_plotly(geometry, field, threshold=0.0):
    """
    Interactive 3D voxel *cells* using Plotly mesh cubes.
    Each active voxel is rendered as a cube.
    """

    nx, ny, nz = field.shape
    X, Y, Z = geometry.voxel_centers()

    fmax = field.max() + 1e-30
    norm = field / fmax

    mask = field > threshold

    fig = go.Figure()

    # voxel half-size
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
    hx, hy, hz = dx / 2, dy / 2, dz / 2

    def add_cube(xc, yc, zc, color):
        """
        Add a single voxel cube as a mesh.
        """

        # 8 cube vertices
        x = [xc - hx, xc + hx, xc + hx, xc - hx,
             xc - hx, xc + hx, xc + hx, xc - hx]

        y = [yc - hy, yc - hy, yc + hy, yc + hy,
             yc - hy, yc - hy, yc + hy, yc + hy]

        z = [zc - hz, zc - hz, zc - hz, zc - hz,
             zc + hz, zc + hz, zc + hz, zc + hz]

        # cube faces (triangles)
        i = [0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3, 5]
        j = [1, 2, 3, 5, 6, 2, 6, 3, 7, 7, 4, 6]
        k = [2, 3, 1, 6, 7, 6, 5, 7, 6, 4, 5, 7]

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=0.6,
                showscale=False
            )
        )

    # colormap
    cmap = "viridis_r"

    import matplotlib.cm as cm
    cmap_fn = cm.get_cmap(cmap)

    # add cubes
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                if not mask[i, j, k]:
                    continue

                xc = X[i, j, k]
                yc = Y[i, j, k]
                zc = Z[i, j, k]

                color = cmap_fn(norm[i, j, k])
                rgba = f'rgba({int(color[0]*255)},'
                rgba += f'{int(color[1]*255)},'
                rgba += f'{int(color[2]*255)},'
                rgba += f'{color[3]})'

                add_cube(xc, yc, zc, rgba)

    fig.update_layout(
        title="3D Voxel Cell Heatmap (Plotly)",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()