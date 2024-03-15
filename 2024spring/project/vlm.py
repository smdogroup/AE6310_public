import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def biot_savart(p1, p2, p):
    """
    Given directional segment p1 -> p2 and a point p outside the segment,
    compute the induced velocity coefficient using Biot Savart law
    Args:
        p1: (array of size 3): start point coordinates
        p2: (array of size 3): end point coordinates
        p: (array of size 3): an arbitrary point
    Return:
        Induced velocity vector
    """
    r1v = p - p1
    r2v = p - p2

    r1 = np.linalg.norm(r1v)
    r2 = np.linalg.norm(r2v)

    vec = (
        np.cross(r1v, r2v)
        * (r1 + r2)
        * (1.0 - np.dot(r1v, r2v) / r1 / r2)
        / ((r1 * r2) ** 2 - np.dot(r1v, r2v) ** 2)
        / 4.0
        / np.pi
    )
    return vec


class VortexLatticeMethod:
    def __init__(
        self,
        xloc,
        panel_nodes,
        alpha,
        rho_inf=1.0,
        vinf=1.0,
    ):
        """
        Vortex lattice method for a half-wing:

                          _________
                        /  /     /
                      /   /     /
                    /-----------
                  /    /      /
                /     /      /                ^  y
              /--------------                 |
            /      /       /                  |        x
          /       /       /                     ------>
         -----------------

        Each panel has a horseshoe vortex attached to it.  For each panel, the
        horse-shoe vortex is at 25% chord location, the control point is at 75%
        chord location. Total number of degrees of freedom of the half wing is
        the number of panels of the half wing. For the illustration, ndof = 6.
        """
        self.xloc = xloc
        self.panel_nodes = panel_nodes
        self.alpha = alpha
        self.rho_inf = rho_inf
        self.vinf = vinf

        # diractions of drag and lift
        self.d_norm = np.array([np.cos(self.alpha), 0.0, np.sin(self.alpha)])
        self.l_norm = np.array([-np.sin(self.alpha), 0.0, np.cos(self.alpha)])

        self.npanels = len(panel_nodes)
        self.span = 2.0 * (xloc[:, 1].max() - xloc[:, 1].min())

        # Normal directions of panels, magnitude is twice the area
        self.panel_norms = np.array(
            [
                np.cross(xloc[n1] - xloc[n3], xloc[n2] - xloc[n4])
                for n1, n2, n3, n4 in panel_nodes
            ]
        )

        cps = []  # control points (at 25% location of each panel)
        vps = []  # vertex line reference points (at 75% location of each panel)

        # Contruct the coefficient matrix
        A = np.zeros((self.npanels, self.npanels))
        for i in range(self.npanels):  # control point i
            for j in range(self.npanels):  # vortex j
                coef, cp, vp1, vp2 = self.eval_induced_velocity_coeff(i, j)
                cps.append(cp)
                vps.append(0.5 * (vp1 + vp2))
                A[i, j] = np.dot(coef, self.panel_norms[i])

        self.A = A
        self.cps = np.array(cps)
        self.vps = np.array(vps)
        return

    def eval_induced_velocity_coeff(self, i, j):
        """
        Evaluate the induced velocity at control point i by vortex on panel j
        """
        INF = 100.0
        ni0, ni1, ni2, ni3 = self.panel_nodes[i]
        nj0, nj1, nj2, nj3 = self.panel_nodes[j]

        # Control point on panel i
        pf = 0.5 * (self.xloc[ni1] + self.xloc[ni2])
        pe = 0.5 * (self.xloc[ni0] + self.xloc[ni3])
        cp = pf + 0.75 * (pe - pf)

        # Horseshoe vertices on panel j
        p0 = self.xloc[nj0].copy()
        p1 = self.xloc[nj1].copy()
        p2 = self.xloc[nj2].copy()
        p3 = self.xloc[nj3].copy()
        p1 += 0.25 * (p0 - p1)  # move backward by 0.25
        p2 += 0.25 * (p3 - p2)  # move backward by 0.25
        p0 += INF * (p0 - p1)  # move backward to INF
        p3 += INF * (p3 - p2)  # move backward to INF
        vp1 = p1.copy()
        vp2 = p2.copy()

        v = np.zeros(3)
        v += biot_savart(p0, p1, cp)
        v += biot_savart(p1, p2, cp)
        v += biot_savart(p2, p3, cp)

        # consider the induced velocity from the other half
        flip = np.array([1.0, -1.0, 1.0])
        p0 *= flip
        p1 *= flip
        p2 *= flip
        p3 *= flip

        v += biot_savart(p3, p2, cp)
        v += biot_savart(p2, p1, cp)
        v += biot_savart(p1, p0, cp)
        return v, cp, vp1, vp2

    def solve(self):
        """
        Solve for circulation gamma for each panel
        """
        b = np.zeros(self.npanels)
        for i in range(self.npanels):  # control point i
            b[i] = -self.vinf * self.panel_norms[i].dot(self.d_norm)

        gamma = np.linalg.solve(self.A, b)
        return gamma

    def compute_panel_lift_drag(self, gamma):
        """
        Compute the lift for each panel, the total force of a penal is given by
        Kutta Joukowski theorem:
            →                     →   →
            F = rho_inf * gamma * v x I

        where v is the composition of free-stream velocity and induced velocity,
        I is the unit vector of the transverse segment of the vortex. The lift
        is the component of F perpendicular to the free-stream velocity, the
        drag is the component of F parallel to the free-stream velocity.
        """
        v_vec = np.zeros((self.npanels, 3))
        I_vec = np.zeros((self.npanels, 3))

        for j in range(self.npanels):  # vortex j
            for i in range(self.npanels):  # control point i
                coef, _, vp1, vp2 = self.eval_induced_velocity_coeff(i, j)
                v_vec[i] += gamma[j] * coef
            I_vec[j] = vp2 - vp1
            I_vec[j] /= np.linalg.norm(I_vec[j])
        v_vec += self.vinf * self.d_norm

        vxI = np.array([np.cross(v, i) for v, i in zip(v_vec, I_vec)])
        F_vec = self.rho_inf * gamma[:, np.newaxis] * vxI

        panel_lift = F_vec.dot(self.l_norm)
        panel_drag = F_vec.dot(self.d_norm)
        return panel_lift, panel_drag

    def compute_wing_CL_CD(self, gamma):
        panel_lift, panel_drag = self.compute_panel_lift_drag(gamma)
        area = 0.5 * np.sum(np.linalg.norm(self.panel_norms, axis=1))
        cl = panel_lift.sum() / 0.5 / self.rho_inf / self.vinf**2 / area
        cd = panel_drag.sum() / 0.5 / self.rho_inf / self.vinf**2 / area
        return cl, cd

    def visualize(self, ax3d, gamma=None):
        for nodes in self.panel_nodes:
            x = np.array([self.xloc[nodes[i % 4], 0] for i in range(5)])
            y = np.array([self.xloc[nodes[i % 4], 1] for i in range(5)])
            z = np.array([self.xloc[nodes[i % 4], 2] for i in range(5)])
            ax3d.plot(x, y, z, "-k", lw=1.0)
            ax3d.plot(x, -y, z, "-k", lw=1.0)

        if gamma is not None:
            panel_lift, panel_drag = self.compute_panel_lift_drag(gamma)

            # Scale properly for visualization
            lift_vec = panel_lift[:, np.newaxis] * self.l_norm
            drag_vec = panel_drag[:, np.newaxis] * self.d_norm

            scale = 0.25 * self.span / panel_lift.max()

            lift_vec *= scale
            drag_vec *= scale

            for panel, (lift, drag) in enumerate(zip(lift_vec, drag_vec)):
                lx = np.array([self.vps[panel, 0], self.vps[panel, 0] + lift[0]])
                ly = np.array([self.vps[panel, 1], self.vps[panel, 1] + lift[1]])
                lz = np.array([self.vps[panel, 2], self.vps[panel, 2] + lift[2]])
                ax3d.plot(lx, ly, lz, color="green", lw=0.5)
                ax3d.plot(lx, -ly, lz, color="green", lw=0.5)

                dx = np.array([self.vps[panel, 0], self.vps[panel, 0] + drag[0]])
                dy = np.array([self.vps[panel, 1], self.vps[panel, 1] + drag[1]])
                dz = np.array([self.vps[panel, 2], self.vps[panel, 2] + drag[2]])
                ax3d.plot(dx, dy, dz, color="pink", lw=0.5)
                ax3d.plot(dx, -dy, dz, color="pink", lw=0.5)

        ax3d.set_aspect("equal", "box")

        ax3d.grid(False)
        ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        # Set up a nice camera angle for the 3D plot
        ax3d.view_init(elev=23.0, azim=-47.0)

        cl, cd = self.compute_wing_CL_CD(gamma)

        ax3d.set_title(
            "alpha: %.2f deg\nCL: %.4f, CD: %.4f, L/D: %.4f"
            % (np.rad2deg(self.alpha), cl, cd, cl / cd),
            y=0.8,
        )
        return


def create_undeformed_half_wing_aero_mesh(
    span=3.0,
    sweep=np.deg2rad(10.0),
    c_root=0.4,
    c_tip=0.2,
    rib_locs=np.linspace(0.0, 1.0, 6),
    spar_locs=np.linspace(0.0, 1.0, 5),
):
    """
    Create a wing mesh for aerodynamic analysis.

                        _________
                      /  /     /
                    /   /     /
                  /-----------
                /    /      /
              /     /      /                ^  y
            /--------------                 |
          /      /       /                  |        x
        /       /       /                     ------>
       -----------------

    Args:
        span: full wing span (total length over two sides)
        sweep: sweep angle of the leading edge in rad
        c_root: length of the root chord
        c_tip: length of the tip chord
        rib_locs: (array of float), y locations of ribs, each value is from 0 to 1
        spar_locs: (array of float), x locations of spars, each value is from 0 to 1

    Note:
        Here "rib" and "spar" are for the convenience of naming the geometry
        entities of the mesh, it has nothing to do with the structural (i.e.
        physical) ribs and spars.
    """
    nribs = len(rib_locs)
    nspars = len(spar_locs)
    npanels = (nribs - 1) * (nspars - 1)
    nnodes = nribs * nspars

    xloc = np.zeros((nnodes, 3))
    panel_nodes = np.zeros((npanels, 4), dtype=int)

    # leading edge y coords for each node
    le_y = 0.5 * span * rib_locs
    le_x = le_y * np.tan(sweep)
    rib_lens = c_root + rib_locs * (c_tip - c_root)

    def get_node(spar, rib):
        return spar * nribs + rib

    def get_panel(spar, rib):
        return spar * (nribs - 1) + rib

    for i in range(nspars):
        for j in range(nribs):
            x = le_x[j] + rib_lens[j] * spar_locs[i]
            y = le_y[j]
            xloc[get_node(i, j)] = x, y, 0.0

    for i in range(nspars - 1):
        for j in range(nribs - 1):
            panel_nodes[get_panel(i, j)] = (
                get_node(i + 1, j),
                get_node(i, j),
                get_node(i, j + 1),
                get_node(i + 1, j + 1),
            )
    return xloc, panel_nodes


def demo_aero_analysis():
    """
    This script demonstrates how to set up the mesh, solve for the flow field,
    and visualize the results
    """
    # Define the wing geometry
    xloc_flat, panel_nodes = create_undeformed_half_wing_aero_mesh(
        span=3.0,
        sweep=np.deg2rad(30.0),
        c_root=0.6,
        c_tip=0.2,
        rib_locs=np.linspace(0.0, 1.0, 6),
        spar_locs=np.linspace(0.0, 1.0, 5),
    )

    xloc_bend = xloc_flat.copy()
    xloc_bend[:, 2] += 0.3 * xloc_flat[:, 1] ** 2  # add bending
    # xloc_bend[:, 2] += 0.01 * xloc_flat[:, 0]  # add twisting

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    # Instantiate the solver object and sole
    for xloc, ax3d in zip([xloc_flat, xloc_bend], [ax1, ax2]):
        vlm = VortexLatticeMethod(xloc, panel_nodes, alpha=np.deg2rad(2.1))
        gamma = vlm.solve()
        vlm.visualize(ax3d, gamma=gamma)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_aero_analysis()
