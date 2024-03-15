import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


class FiniteElementAnalysis:
    """
        .-.-.-.-.-.-.-.-.-.-.-.
        |       |       |     |
        .       .       .     .
        |       |       |     |
        .-.-.-.-.-.-.-.-.-.-.-.
        |       |       |     |
        .       .       .     .
        |       |       |     |
        .-.-.-.-.-.-.-.-.-.-.-.

    This class implements a finite element method for some 3-dimensional
    skeleton structure like shown above. Each bar of the skeleton is modeled by
    a series of Euler-Bernoulli beam elements.

    Degrees of freedom are associated with nodes, each node has 6 degrees of
    freedom:
        - 3 translational displacement: u, v, w
        - 3 rotational displacement: theta_x, theta_y, theta_z

    For the beam element, hollow cylindrical cross section is assumed, with
    thickness t, inner radius r and outer radius r + t
    """

    def __init__(
        self, xloc, elem_nodes, bc_nodes, elastic_modulus=1.0, shear_modulus=1.0
    ):
        """
        Args:
            xloc: (array of shape nnodes by 3) xyz coordinates for nodes (dots
            in the diagram)
            elem_nodes: (array of shape nelems by 2) mapping from element number
            to node numbers
            bc_nodes: (1d array) nodes where full-clamped boundary condition
            will be applied, for aircraft wing model, this is the set of root
            nodes
        """
        self.xloc = xloc
        self.elem_nodes = elem_nodes
        self.bc_nodes = bc_nodes

        self.nelems = len(elem_nodes)
        self.nnodes = len(xloc)

        self.elastic_modulus = elastic_modulus
        self.shear_modulus = shear_modulus
        return

    def get_num_elements(self):
        return self.nelems

    def get_num_nodes(self):
        return self.nnodes

    def get_node_dof(self, node):
        """
        (Internal method)

        Args:
            node: node index
        Return:
            a list of the global degrees of freedom indices for a node
        """
        return np.arange(6 * node, 6 * (node + 1))

    def get_element_transform(self, x0, x1):
        """
        (Internal method)

        Args:
            x0, x1: xyz coordinates of the first and second node of a beam element

        Return:
            T: transformation matrix
            L: length
        """

        # Compute the length of the beam
        L = np.sqrt(np.dot(x1 - x0, x1 - x0))

        # The direction cosine matrix
        C = np.zeros((3, 3))

        # The x direction
        C[0, :] = (x1 - x0) / L

        # Direction cosines
        l = C[0, 0]
        m = C[0, 1]
        n = C[0, 2]

        # Compute the remaining contributions
        D = np.sqrt(l**2 + m**2)

        C[1, 0] = -m / D
        C[1, 1] = l / D

        C[2, 0] = -l * n / D
        C[2, 1] = -m * n / D
        C[2, 2] = D

        # Set the transformation matrix
        T = np.zeros((12, 12))

        # Inject C into T
        for i in range(3):
            for j in range(3):
                T[i, j] = C[i, j]
                T[i + 3, j + 3] = C[i, j]
                T[i + 6, j + 6] = C[i, j]
                T[i + 9, j + 9] = C[i, j]

        return T, L

    def get_element_matrix(self, L, EA, GJ, EIz, EIy):
        """
        (Internal method)

        Compute the element stiffness matrix.

        Args:
            L: element length
            EA: Young's modulus times cross-sectional area
            GJ: shear modulus times cross-sectional moment of inertia along the
            axis that is perpendicular to the cross section
            EIz, EIy: Young's modulus times cross-sectional moment of inertia
            along the axes that are parallel to the cross section

        Note: EA, GJ, EIz, EIy are array of size nelems

        Return:
            element stiffness matrix
        """

        # The stiffness matrix in local coordinates
        ke = np.zeros((12, 12))

        # Set the axial stiffness
        ke[0, 0] = ke[6, 6] = EA / L
        ke[0, 6] = ke[6, 0] = -EA / L

        # Set the torsional stiffness
        ke[3, 3] = ke[9, 9] = GJ / L
        ke[3, 9] = ke[9, 3] = -GJ / L

        # Element stiffness coefficients
        k0 = np.array(
            [
                [12.0, -6.0, -12.0, -6.0],
                [-6.0, 4.0, 6.0, 2.0],
                [-12.0, 6.0, 12.0, 6.0],
                [-6.0, 2.0, 6.0, 4.0],
            ]
        )

        # Set the deformation in the z-direction - rotation about y-axis
        cz = np.array([1.0, L, 1.0, L])
        kz = (k0 / L**3) * np.outer(cz, cz)

        # Set the degrees of freedom
        dof = [2, 4, 8, 10]

        # Add the values to the element stiffness matrix
        for i, ie in enumerate(dof):
            for j, je in enumerate(dof):
                ke[ie, je] = EIy * kz[i, j]

        # Set the deformation in the y-direction - rotation about the z-axis
        cy = np.array([1.0, -L, 1.0, -L])
        ky = (k0 / L**3) * np.outer(cy, cy)

        # Set the degrees of freedom
        dof = [1, 5, 7, 11]

        # Add the values to the element stiffness matrix
        for i, ie in enumerate(dof):
            for j, je in enumerate(dof):
                ke[ie, je] = EIz * ky[i, j]

        return ke

    def get_force_bcs_matrix(self):
        """
        Get the matrix that transforms the global force vector into forces
        applied to the finite-element model.

        This matrix will zero-out forces associated with the degrees of freedom
        at boundary conditions
        """

        Fbc = np.eye(6 * self.nnodes)

        # Apply the boundary conditions
        for bc_node in self.bc_nodes:
            bc_dof = self.get_node_dof(bc_node)
            for dof in bc_dof:
                Fbc[dof, :] = 0.0

        return Fbc

    def compute_stiffness_matrix(self, r, t):
        """
        Assemble the global stiffness matrix

        Args:
            r: (numpy array of length nelems) inner radii of beam elements
            t: (numpy array of length nelems) thicknesses of beam elements

        Return:
            The global stiffness matrix with boundary conditions applied
        """
        nelems = self.nelems

        A = np.pi * ((r + t) ** 2 - r**2)
        I = 0.5 * np.pi * ((r + t) ** 4 - r**4)
        J = np.pi * ((r + t) ** 4 - r**4)

        EA = self.elastic_modulus * A
        GJ = self.shear_modulus * J
        EI = self.elastic_modulus * I

        # The global stiffness matrix
        K = np.zeros((6 * self.nnodes, 6 * self.nnodes))

        for i in range(nelems):
            n0, n1 = self.elem_nodes[i]

            x0 = self.xloc[n0]
            x1 = self.xloc[n1]

            # Get the degrees of freedom for the element
            dof = []
            dof.extend(self.get_node_dof(n0))
            dof.extend(self.get_node_dof(n1))

            # Compute the transformation to the global coordinate system
            T, L = self.get_element_transform(x0, x1)

            # Get the element stiffness matrix in the local frame
            ke = self.get_element_matrix(L, EA[i], GJ[i], EI[i], EI[i])

            # Compute the stiffness in the global frame
            kglobal = np.dot(T.T, np.dot(ke, T))

            for i, idof in enumerate(dof):
                for j, jdof in enumerate(dof):
                    # Assemble the global stiffness matrix
                    K[idof, jdof] += kglobal[i, j]

        # Apply the boundary conditions
        for bc_node in self.bc_nodes:
            bc_dof = self.get_node_dof(bc_node)
            for dof in bc_dof:
                K[dof, :] = 0.0
                K[dof, dof] = 1.0

        return K

    def compute_stress(self, nodal_displacements, r, t):
        """
        Given the solution nodal displacements, evaluate the maximum normal
        stress of the beam elements.

        Args:
            nodal_displacements: (numpy array of length 6 * nnodes), solution
            r: (numpy array of length nelems) inner radii of beam elements
            t: (numpy array of length nelems) thicknesses of beam elements
        """

        u = nodal_displacements

        stress = np.zeros(self.nelems)

        for i in range(self.nelems):
            n0, n1 = self.elem_nodes[i]

            # Compute element length
            x0 = self.xloc[n0]
            x1 = self.xloc[n1]
            L = np.sqrt(np.dot(x1 - x0, x1 - x0))

            dof1 = self.get_node_dof(n0)
            dof2 = self.get_node_dof(n1)

            ky = (u[dof2[5]] - u[dof1[5]]) / L  # d2v/dx2
            kz = (u[dof2[4]] - u[dof1[4]]) / L  # d2w/dx2

            sx1 = self.elastic_modulus * ky * (r[i] + t[i])
            sx2 = self.elastic_modulus * kz * (r[i] + t[i])

            stress[i] = np.sqrt(sx1**2 + sx2**2)

        return stress

    def visualize(self, ax3d, disp=None, stress=None):
        """
        Visualize the mesh and optionally deformation if disp is given.

        Args:
            ax3d: an matplotlib Axes3D object
            disp: displacement vector, solution of the linear system
        """

        u = self.xloc.copy()
        if disp is not None:
            u[:, 0] += disp[0::6]
            u[:, 1] += disp[1::6]
            u[:, 2] += disp[2::6]

        color = lambda elem: "black"
        if stress is not None:
            cmap = plt.get_cmap("bwr")
            norm = Normalize(vmin=min(stress), vmax=max(stress))
            mapper = ScalarMappable(norm=norm, cmap=cmap)
            color = lambda elem: mapper.to_rgba(stress[elem])

        for elem, nodes in enumerate(self.elem_nodes):
            n1 = nodes[0]
            n2 = nodes[1]
            x = np.array([u[n1, 0], u[n2, 0]])
            y = np.array([u[n1, 1], u[n2, 1]])
            z = np.array([u[n1, 2], u[n2, 2]])

            ax3d.plot(
                x,
                y,
                z,
                "-o",
                lw=1.0,
                markersize=1.0,
                color=color(elem),
            )
            ax3d.plot(
                x,
                -y,
                z,
                "-o",
                lw=1.0,
                markersize=1.0,
                color=color(elem),
            )

        bcx = self.xloc[self.bc_nodes, 0]
        bcy = self.xloc[self.bc_nodes, 1]
        bcz = self.xloc[self.bc_nodes, 2]

        s = ax3d.scatter(
            bcx, bcy, bcz, color="gray", label="clamped nodes", zorder=100, alpha=0.5
        )

        # Set up a nice camera angle for the 3D plot
        ax3d.view_init(elev=26.0, azim=-28.0)
        ax3d.legend()
        ax3d.set_aspect("equal", "box")

        ax3d.grid(False)
        ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        if stress is not None:
            plt.colorbar(
                mapper,
                ax=ax3d,
                label="stress",
                shrink=0.6,
                aspect=30.0,
                location="top",
                pad=-0.3,
            )

        return


def create_undeformed_half_wing_structure_mesh(
    span=2.0,
    sweep=np.deg2rad(15.0),
    c_root=0.4,
    c_tip=0.2,
    rib_locs=np.array([0.2, 0.5, 1.0]),
    spar_locs=np.array([0.0, 0.25, 1.0]),
    nelems_per_rib_seg=3,
    nelems_per_spar_seg=3,
):
    """
    Create a wing frame that comprises ribs and spars, where each rib and spar
    segment is meshed into several beam elements:

                                .-.--.---.--.
                              /     /      /
                            .     .       .
                           /     /       /
                        .--.---.----.---.
                       /      /        /
                    .       .         .
                   /       /         /
                .---.----.----.-----.                 ^  y
               /        /          /                  |
            .         .           .                   |       x
           /         /           /                     ------>

    In the illustration above, "." represents a node, "/", represents a beam
    element for the spar, contiguous "-"s represent a beam of element for the
    rib. For this illustration, each spar segment and each rib segment are
    divided into two beam elements. i.e. nelems_per_rib_seg ==
    nelems_per_spar_seg == 2

    Args:
        span: full wing span (total length over two sides)
        sweep: sweep angle of the leading edge in rad
        c_root: length of the root chord (a chord without the rib)
        c_tip: length of the tip chord
        nribs: number of ribs
        rib_locs: (array of float), y locations of ribs, each value is from 0 to 1
        spar_locs: (array of float), x locations of spars, each value is from 0 to 1
        nelems_per_rib_seg: number of beam elements per rib segment
        nelems_per_spar_seg: number of beam elements per spar segment
    """

    nribs = len(rib_locs)
    nspars = len(spar_locs)

    nnodes_per_rib = 1 + (nspars - 1) * nelems_per_rib_seg
    nnodes_per_spar = 1 + nribs * nelems_per_spar_seg
    nnodes = nnodes_per_rib * nribs + nnodes_per_spar * nspars - nribs * nspars

    nelems_per_rib = nnodes_per_rib - 1
    nelems_per_spar = nnodes_per_spar - 1
    nelems = nelems_per_rib * nribs + nelems_per_spar * nspars

    # rib lengths
    rib_lens = c_root + rib_locs * (c_tip - c_root)

    # chord lengths at each span-wise nodal standing
    chord_lens = np.zeros(nnodes_per_spar)
    chord_lens[0] = c_root

    # node coordinates on the leading edge
    le_y = np.zeros(nnodes_per_spar)
    loc0 = 0.0
    len0 = c_root
    for i, (loc1, len1) in enumerate(zip(rib_locs, rib_lens)):
        begin = 1 + i * nelems_per_spar_seg
        end = 1 + (i + 1) * nelems_per_spar_seg
        le_y[begin:end] = np.linspace(loc0, loc1, nelems_per_spar_seg + 1)[1:]
        chord_lens[begin:end] = np.linspace(len0, len1, nelems_per_spar_seg + 1)[1:]
        loc0 = loc1
        len0 = len1

    le_y *= 0.5 * span
    le_x = le_y * np.tan(sweep)

    # Initialize nodal coordinates and element connectivity
    xloc = np.zeros((nnodes, 3))
    elem_nodes = np.zeros((nelems, 2), dtype=int)

    # We number nodes on the spars first spar by spar, from root to tip, leading
    # edge to trailing edge
    def spar_node(spar, i):
        """
        Args:
            spar: spar index
            i: node index local to the spar
        """
        return spar * nnodes_per_spar + i

    # We then number the remaining nodes on the ribs rib by rib, from root to
    # tip, leading edge to trailing edge
    def rib_node(rib, i):
        """
        Args:
            rib: rib index
            i: node index local to the rib
        """
        offset = nspars * nnodes_per_spar
        if i % nelems_per_rib_seg == 0:
            return spar_node(i // nelems_per_rib_seg, nelems_per_spar_seg * (rib + 1))
        else:
            n = rib * nnodes_per_rib + i
            return offset + n - rib * nspars - i // nelems_per_rib_seg - 1

    # Nodes on the spars
    for i in range(nspars):
        for j in range(nnodes_per_spar):
            node = spar_node(i, j)
            xloc[node] = le_x[j] + spar_locs[i] * chord_lens[j], le_y[j], 0.0

    # Nodes on the ribs
    for i in range(nribs):
        spar_i = (i + 1) * nelems_per_spar_seg
        for j in range(nnodes_per_rib):
            jj = j % nelems_per_rib_seg
            kk = j // nelems_per_rib_seg
            if jj == 0:
                continue

            node = rib_node(i, j)
            xloc[node] = le_x[spar_i], le_y[spar_i], 0.0
            xloc[node][0] += (
                spar_locs[kk]
                + jj / nelems_per_rib_seg * (spar_locs[kk + 1] - spar_locs[kk])
            ) * chord_lens[spar_i]

    # elements on the spars
    elem = 0
    for i in range(nspars):
        for j in range(nelems_per_spar):
            elem_nodes[elem] = spar_node(i, j), spar_node(i, j + 1)
            elem += 1

    # elements on the ribs
    for i in range(nribs):
        for j in range(nelems_per_rib):
            elem_nodes[elem] = rib_node(i, j), rib_node(i, j + 1)
            elem += 1

    root_nodes = np.zeros(nspars, dtype=int)
    for spar in range(nspars):
        root_nodes[spar] = spar_node(spar, 0)

    return xloc, elem_nodes, root_nodes


def demo_solve_static_structure(analysis: FiniteElementAnalysis, nodal_forces):
    """
    This function demonstrates how to set up the structural linear system and
    solve for the displacement

    Args:
        analysis: the analysis object of class FiniteElementAnalysis
        nodal_forces: (array of length nnodes * 6), nodal force vector
    """
    # Get the total number of beam elements
    nelems = analysis.get_num_elements()
    nnodes = analysis.get_num_nodes()

    # Set up the stiffness matrix
    r = np.ones(nelems)
    t = 0.5 * np.ones(nelems)
    K = analysis.compute_stiffness_matrix(r=r, t=t)

    # Create the right-hand-side of the linear system of equations and apply
    # boundary conditions
    Fbc = analysis.get_force_bcs_matrix()
    rhs = Fbc.dot(nodal_forces)

    # Solve the linear system for nodal displacements
    nodal_displacements = np.linalg.solve(K, rhs)

    # Evaluate stress
    stress = analysis.compute_stress(nodal_displacements, r, t)

    return nodal_displacements, stress


def demo_structural_analysis():
    """
    This script demonstrates how to set up the mesh, solve the static structural
    problem, and visualize the deformation
    """
    # Create the mesh
    xloc, elem_nodes, bc_nodes = create_undeformed_half_wing_structure_mesh(
        span=3.0,
        sweep=np.deg2rad(30.0),
        c_root=0.6,
        c_tip=0.2,
        rib_locs=np.linspace(0.0, 1.0, 6)[1:],
        spar_locs=np.array([0.0, 0.25, 1.0]),
        nelems_per_rib_seg=5,
        nelems_per_spar_seg=5,
    )

    # Create the analysis, **kwargs object
    analysis = FiniteElementAnalysis(
        xloc=xloc, elem_nodes=elem_nodes, bc_nodes=bc_nodes
    )

    # Create the load
    forces = np.zeros((analysis.get_num_nodes(), 6))
    forces[:, 2] = 0.2  # apply a force along +z direction to all the nodes
    forces = forces.flatten()

    # Solve the linear system
    disp, stress = demo_solve_static_structure(analysis, forces)

    # Visualize the mesh and displacement
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111, projection="3d")
    analysis.visualize(ax1, disp=disp, stress=stress)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_structural_analysis()
