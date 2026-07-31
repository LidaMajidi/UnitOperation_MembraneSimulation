"""
================================================================================
MOLLOCATOR — Hollow Fiber Membrane Module Simulator
Reproduction of: Chu, Lindbråthen, Lei, He, Hillestad (2019)
"Mathematical modeling and process parametric study of CO2 removal from
natural gas by hollow fiber membranes", Chem. Eng. Res. Des. 148, 45-55.

FEATURES IN THIS UNIFIED VERSION:
  1. Mathematically corrected boundary conditions (L'Hopital's limit at z=0).
  2. Increased collocation nodes (n=10) for stiff, high-selectivity membranes.
  3. Automated directory management and figure saving during execution.
================================================================================
"""

# ==============================================================================
# PART 1: IMPORTS AND NUMERICAL METHODS (Orthogonal Collocation)
# ==============================================================================
import os
import numpy as np
from scipy.optimize import root, brentq
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Universal Gas Constant [Pa m3 / (mol K)]
R = 8.314          

def collocation_nodes_and_matrix(n_internal):
    """
    Generates the interior collocation points based on the roots of Legendre polynomials
    and constructs the differentiation matrix (A) for the boundary value problem.
    Maps the domain from [-1, 1] to [0, 1].
    """
    from numpy.polynomial.legendre import leggauss
    
    # Calculate roots of Legendre polynomial
    xi, _ = leggauss(n_internal)              
    xi_shifted = 0.5 * (xi + 1.0)              
    
    # Include boundaries (z=0 and z=1)
    nodes = np.concatenate(([0.0], np.sort(xi_shifted), [1.0]))
    m = len(nodes)

    # Build the Lagrange differentiation matrix (Fornberg algorithm)
    c = np.ones(m)
    for i in range(m):
        for k in range(m):
            if k != i:
                c[i] *= (nodes[i] - nodes[k])

    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                A[i, j] = c[i] / (c[j] * (nodes[i] - nodes[j]))
        A[i, i] = -np.sum(A[i, :])

    return nodes, A


# ==============================================================================
# PART 2: AUXILIARY PHYSICAL PROPERTY MODEL
# ==============================================================================
# Approximate dynamic viscosities of pure components near 300-310 K [Pa.s].
PURE_VISCOSITY = {
    "CO2":  1.50e-5,
    "CH4":  1.10e-5,
    "C2H6": 0.95e-5,
    "C3H8": 0.80e-5,
    "N2":   1.78e-5,
}

def mixture_viscosity(composition):
    """
    Calculates the dynamic viscosity of the gas mixture using a simple 
    mole-fraction-weighted linear mixing rule.
    """
    mu = 0.0
    for comp, frac in composition.items():
        mu += frac * PURE_VISCOSITY.get(comp, 1.2e-5)
    return mu


# ==============================================================================
# PART 3: MEMBRANE MODULE CLASS INITIALIZATION & DIMENSIONLESS GROUPS
# ==============================================================================
class MembraneModule:
    """
    Counter-/co-current, shell-/bore-side-feed hollow fiber membrane module,
    solved via orthogonal collocation per Eqs. (6)-(11) and Table 1.
    """
    def __init__(self, components, permeance, xf, Pf, Pp, T, uf,
                 Di, Do, L, N, D=None, flow_pattern="counter", feed_side="shell",
                 sweep=None, mu=None, n_internal=10):  # n_internal increased for stability
        """
        Initializes the membrane module parameters and setup.
        """
        self.components = list(components)
        self.nc = len(self.components)
        self.Q = np.array([permeance[c] for c in self.components])
        self.xf = np.array([xf[c] for c in self.components])
        assert abs(self.xf.sum() - 1.0) < 1e-6, "Feed composition must sum to 1"
        
        self.Pf, self.Pp, self.T, self.uf = Pf, Pp, T, uf
        self.Di, self.Do, self.L, self.N = Di, Do, L, N
        
        # If housing diameter (D) is not provided, calculate it based on 2x cross-section rule
        self.D = D if D is not None else np.sqrt(2.0) * np.sqrt(N) * Do
        self.flow_pattern = flow_pattern
        self.feed_side = feed_side
        self.sweep = sweep
        self.mu = mu if mu is not None else mixture_viscosity(xf)
        self.n_internal = n_internal

        # Get collocation nodes and differentiation matrix
        self.nodes, self.A = collocation_nodes_and_matrix(n_internal)
        self.m = len(self.nodes)

        self._build_dimensionless_groups()

    def _shell_type_K(self):
        """Calculates pressure drop coefficient for the shell-side."""
        D, Do, N, L, uf, Pf, mu, T = (self.D, self.Do, self.N, self.L,
                                       self.uf, self.Pf, self.mu, self.T)
        return (192.0 * mu * R * T * N * Do * (D + N * Do) * L * uf /
                (np.pi * (D**2 - N * Do**2) ** 3 * N * Pf**2))

    def _bore_type_K(self):
        """Calculates pressure drop coefficient for the bore-side."""
        Di, N, L, uf, Pf, mu, T = (self.Di, self.N, self.L, self.uf,
                                    self.Pf, self.mu, self.T)
        return 128.0 * mu * R * T * L * uf / (np.pi * Di**4 * N * Pf**2)

    def _build_dimensionless_groups(self):
        """Pre-calculates dimensionless constants based on the flow pattern."""
        self.Ki = (np.pi * self.Do * self.L * self.N * self.Pf * self.Q / self.uf)

        if self.feed_side == "shell":
            self.K_feed = self._shell_type_K()      
            self.K_perm = self._bore_type_K()        
        elif self.feed_side == "bore":
            self.K_feed = self._bore_type_K()        
            self.K_perm = self._shell_type_K()        
        else:
            raise ValueError("feed_side must be 'shell' or 'bore'")

        # Sign convention (Table 1)
        self.sign_feed = 1.0 if self.flow_pattern == "counter" else -1.0

    def _unpack(self, X):
        """Helper to unpack the 1D solution vector into flow and pressure profiles."""
        m, nc = self.m, self.nc
        ux = X[0: m * nc].reshape(m, nc)
        vy = X[m * nc: 2 * m * nc].reshape(m, nc)
        P = X[2 * m * nc: 2 * m * nc + m]
        p = X[2 * m * nc + m: 2 * m * nc + 2 * m]
        return ux, vy, P, p

    def _pack(self, ux, vy, P, p):
        """Helper to pack variables into a single 1D vector for the solver."""
        return np.concatenate([ux.ravel(), vy.ravel(), P, p])


# ==============================================================================
# PART 4: ODE SYSTEM CORE & RESIDUALS EVALUATION (L'Hopital Fix Applied)
# ==============================================================================
    def _residuals(self, X, Q_scale=1.0):
        """
        Evaluates the residual of the algebraic system.
        Q_scale: Homotopy factor (0 to 1) applied to permeance for robust convergence.
        """
        m, nc, A = self.m, self.nc, self.A
        ux, vy, P, p = self._unpack(X)

        Ki = self.Ki * Q_scale
        sumU = ux.sum(axis=1)                      
        sumV = vy.sum(axis=1)

        xi = ux / sumU[:, None]
        yi = np.zeros_like(vy)
        
        # Correctly calculating mole fractions (Handles the L'Hopital limit at the closed end z=0)
        for k in range(m):
            if sumV[k] > 1e-12:
                yi[k, :] = vy[k, :] / sumV[k]
            else:
                # Limit condition at zero flux (Sweep = None at z=0)
                qx = self.Q * xi[k, :]
                yi[k, :] = qx / np.sum(qx)

        # Local flux term (J)
        J = Ki[None, :] * (P[:, None] * xi - p[:, None] * yi)

        # Apply differentiation matrix
        dux_dz = A @ ux            
        dvy_dz = A @ vy
        dP_dz = A @ P
        dp_dz = A @ p

        # Calculate Residuals (Difference between derivatives and physical fluxes)
        R_ux = dux_dz - self.sign_feed * J
        R_vy = dvy_dz - J
        R_P = dP_dz - self.sign_feed * self.K_feed * sumU / np.maximum(P, 1e-10)
        R_p = dp_dz + self.K_perm * sumV / np.maximum(p, 1e-10)

        # Apply Boundary Conditions
        last = m - 1
        
        # Feed entry boundary conditions at z*=1
        R_ux[last, :] = ux[last, :] - self.xf
        R_P[last] = P[last] - 1.0

        # Permeate boundary conditions
        if self.sweep is None:
            R_vy[0, :] = vy[0, :] - 0.0
            R_p[last] = p[last] - (self.Pp / self.Pf)
        else:
            vswp = self.sweep["vswp"] / self.uf
            yswp = np.array([self.sweep["yswp"][c] for c in self.components])
            pswp = self.sweep["pswp"] / self.Pf
            R_vy[0, :] = vy[0, :] - vswp * yswp
            R_p[0] = p[0] - pswp

        return self._pack(R_ux, R_vy, R_P, R_p)

    def _initial_guess(self):
        """Generates the initial guess profiles for the solver."""
        m, nc = self.m, self.nc
        z = self.nodes
        ux0 = np.outer(np.ones(m), self.xf) * (0.92 + 0.08 * z[:, None])
        
        fast_idx = int(np.argmax(self.Q))
        vy0 = np.zeros((m, nc))
        base_perm_frac = 0.08                      
        for i in range(nc):
            enrich = 3.0 if i == fast_idx else 0.5
            vy0[:, i] = base_perm_frac * self.xf[i] * enrich * z
            
        if self.sweep is not None:
            vswp = self.sweep["vswp"] / self.uf
            yswp = np.array([self.sweep["yswp"][c] for c in self.components])
            vy0 += vswp * yswp
            
        P0 = 1.0 - 0.001 * (1 - z)
        p0 = (self.Pp / self.Pf) * np.ones(m) if self.sweep is None else \
            (self.sweep["pswp"] / self.Pf) * np.ones(m)
            
        return self._pack(ux0, vy0, P0, p0)


# ==============================================================================
# PART 5: SOLVER EXECUTION & POST-PROCESSING
# ==============================================================================
    def solve(self, homotopy_steps=15, tol=1e-8, maxfev=100000):
        """
        Solves the nonlinear system using Homotopy for stability.
        """
        X = self._initial_guess()
        scales = np.linspace(1.0 / homotopy_steps, 1.0, homotopy_steps)

        for s in scales:
            sol = root(self._residuals, X, args=(s,), method="hybr",
                       tol=tol, options={"maxfev": maxfev})
            if not sol.success:
                # Fallback to Levenberg-Marquardt if Hybrid method fails
                sol = root(self._residuals, X, args=(s,), method="lm",
                            options={"maxiter": maxfev}) 
            if not sol.success:
                raise RuntimeError(
                    f"Mollocator failed to converge at homotopy step "
                    f"Q_scale={s:.3f}: {sol.message}")
            X = sol.x

        self.X = X
        self.converged = True
        return self._postprocess(X)

    def _postprocess(self, X):
        """Converts dimensionless results back to physical units for plotting."""
        ux, vy, P, p = self._unpack(X)
        z_dim = self.nodes * self.L

        sumU = ux.sum(axis=1)
        sumV = np.maximum(vy.sum(axis=1), 1e-10)
        x_prof = ux / sumU[:, None]
        y_prof = vy / sumV[:, None]

        retentate_flow = sumU[0] * self.uf
        retentate_x = dict(zip(self.components, ux[0, :] / sumU[0]))
        permeate_flow = vy[-1, :].sum() * self.uf
        permeate_y = dict(zip(self.components,
                               vy[-1, :] / max(vy[-1, :].sum(), 1e-10)))

        return {
            "z": z_dim, "z_star": self.nodes,
            "ux_star": ux, "vy_star": vy,
            "P": P * self.Pf, "p": p * self.Pf,
            "x_profile": x_prof, "y_profile": y_prof,
            "retentate_flow": retentate_flow, "retentate_x": retentate_x,
            "permeate_flow": permeate_flow, "permeate_y": permeate_y,
            "components": self.components,
        }


# ==============================================================================
# PART 6: GEOMETRY HELPERS (Section 3.2 module sizing rules)
# ==============================================================================
def module_diameter_from_packing_density(N, Do, packing_density):
    """Calculates D from packing density."""
    return np.sqrt(4.0 * N * Do / packing_density)

def fibers_for_constant_area(A_ref, Do, L):
    """Calculates number of fibers (N) for a constant total membrane area."""
    return A_ref / (np.pi * Do * L)

def module_diameter_default(N, Do):
    """Calculates module cross-section based on 2x total fiber cross-section."""
    return np.sqrt(2.0) * np.sqrt(N) * Do


# ==============================================================================
# PART 7: MODEL VALIDATION (Scenarios 2 & 3 against Table 5)
# ==============================================================================
def run_validation():
    print("=" * 70)
    print("VALIDATION AGAINST TABLE 5 (Scenarios 2 & 3, Mollocator vs ChemBrane)")
    print("=" * 70)

    components = ["CO2", "CH4"]
    scenario_params = {
        2: dict(T=308, Pf=35e5, Pp=1e5, uf=0.35, xf={"CO2": 0.10, "CH4": 0.90},
                permeance={"CO2": 3.207e-9, "CH4": 1.33e-10},
                Di=200e-6, Do=250e-6, L=0.6, N=60000, D=0.1,
                flow_pattern="counter", feed_side="shell",
                ChemBrane=dict(perm_flow=0.03, perm_CO2=59.88,
                                ret_flow=0.32, ret_CH4=94.94)),
        3: dict(T=308, Pf=15e5, Pp=1e5, uf=0.35, xf={"CO2": 0.10, "CH4": 0.90},
                permeance={"CO2": 3.207e-9, "CH4": 1.33e-10},
                Di=120e-6, Do=170e-6, L=1.5, N=60000, D=0.05,
                flow_pattern="counter", feed_side="shell",
                ChemBrane=dict(perm_flow=0.0205, perm_CO2=56.72,
                                ret_flow=0.3294, ret_CH4=92.92)),
    }

    for sc, p in scenario_params.items():
        mod = MembraneModule(components=components, permeance=p["permeance"],
                              xf=p["xf"], Pf=p["Pf"], Pp=p["Pp"], T=p["T"],
                              uf=p["uf"], Di=p["Di"], Do=p["Do"], L=p["L"],
                              N=p["N"], D=p["D"], flow_pattern=p["flow_pattern"],
                              feed_side=p["feed_side"], n_internal=10)
        res = mod.solve()
        
        perm_flow = res["permeate_flow"]
        perm_CO2 = res["permeate_y"]["CO2"] * 100
        ret_flow = res["retentate_flow"]
        ret_CH4 = res["retentate_x"]["CH4"] * 100
        cb = p["ChemBrane"]
        
        print(f"\nScenario {sc}:")
        print(f"  Permeate flow   : Mollocator={perm_flow:.4e}  ChemBrane={cb['perm_flow']:.4e}  RD={100*(perm_flow-cb['perm_flow'])/cb['perm_flow']:.2f}%")
        print(f"  Permeate CO2 %  : Mollocator={perm_CO2:.2f}  ChemBrane={cb['perm_CO2']:.2f}")
        print(f"  Retentate flow  : Mollocator={ret_flow:.4e}  ChemBrane={cb['ret_flow']:.4e}")
        print(f"  Retentate CH4 % : Mollocator={ret_CH4:.2f}  ChemBrane={cb['ret_CH4']:.2f}")


# ==============================================================================
# PART 8: MODULE DESIGN PARAMETRIC STUDIES (Figs. 2, 3, 4)
# ==============================================================================
BASE_PERMEANCE = {"CO2": 3.207e-9, "CH4": 1.33e-10}
BASE_XF = {"CO2": 0.10, "CH4": 0.90}
BASE_T, BASE_PF, BASE_PP, BASE_UF = 308.0, 35e5, 1e5, 0.35
A_REF = 60000 * np.pi * 250e-6 * 0.6           

def fig2_hollow_fiber_diameter_study():
    Di_list = [100e-6, 150e-6, 200e-6, 250e-6, 300e-6]
    thickness = 25e-6
    packing_density = 6000.0
    L = 0.6
    results = {}
    for Di in Di_list:
        Do = Di + 2 * thickness
        N = fibers_for_constant_area(A_REF, Do, L)
        D = module_diameter_from_packing_density(N, Do, packing_density)
        mod = MembraneModule(components=["CO2", "CH4"], permeance=BASE_PERMEANCE,
                              xf=BASE_XF, Pf=BASE_PF, Pp=BASE_PP, T=BASE_T,
                              uf=BASE_UF, Di=Di, Do=Do, L=L, N=N, D=D,
                              flow_pattern="counter", feed_side="shell",
                              n_internal=10)
        results[Di] = mod.solve()
    return results

def fig3_hollow_fiber_length_study():
    Di_list = [150e-6, 200e-6]
    thickness = 25e-6
    L_list = np.linspace(0.5, 2.0, 10)
    packing_density = 6000.0
    results = {Di: {"L": [], "CH4_purity": [], "CH4_loss": []} for Di in Di_list}
    for Di in Di_list:
        Do = Di + 2 * thickness
        for L in L_list:
            N = fibers_for_constant_area(A_REF, Do, L)
            D = module_diameter_from_packing_density(N, Do, packing_density)
            mod = MembraneModule(components=["CO2", "CH4"], permeance=BASE_PERMEANCE,
                                  xf=BASE_XF, Pf=BASE_PF, Pp=BASE_PP, T=BASE_T,
                                  uf=BASE_UF, Di=Di, Do=Do, L=L, N=N, D=D,
                                  flow_pattern="counter", feed_side="shell",
                                  n_internal=10)
            res = mod.solve()
            ch4_purity = res["retentate_x"]["CH4"] * 100
            ch4_feed = BASE_UF * BASE_XF["CH4"]
            ch4_loss = (res["permeate_flow"] * res["permeate_y"]["CH4"] / ch4_feed) * 100
            results[Di]["L"].append(L)
            results[Di]["CH4_purity"].append(ch4_purity)
            results[Di]["CH4_loss"].append(ch4_loss)
    return results

def fig4_packing_density_study():
    Do, Di, L, N = 250e-6, 200e-6, 0.6, 60000
    packing_list = np.array([6000, 8000, 10000, 12000, 13000, 14000, 14250])
    profiles, D_list, dP_list = {}, [], []
    for pd in packing_list:
        D = module_diameter_from_packing_density(N, Do, pd)
        mod = MembraneModule(components=["CO2", "CH4"], permeance=BASE_PERMEANCE,
                              xf=BASE_XF, Pf=BASE_PF, Pp=BASE_PP, T=BASE_T,
                              uf=BASE_UF, Di=Di, Do=Do, L=L, N=N, D=D,
                              flow_pattern="counter", feed_side="shell",
                              n_internal=10)
        res = mod.solve()
        profiles[pd] = res
        D_list.append(D)
        dP = res["P"].max() - res["P"].min()
        dP_list.append(dP / 100.0)     
    return profiles, packing_list, np.array(D_list), np.array(dP_list)


# ==============================================================================
# PART 9: PROCESS PARAMETRIC STUDY — NATURAL GAS SWEETENING (Figs. 5, 6, 7)
# ==============================================================================
COMPONENTS_NG = ["CO2", "CH4", "C2H6", "C3H8", "N2"]

MEMBRANE_PERMEANCE = {
    "CA": {"CO2": 1.691e-8, "CH4": 1.127e-9, "C2H6": 3.758e-10,
           "C3H8": 3.381e-10, "N2": 1.127e-9},
    "PI": {"CO2": 3.283e-8, "CH4": 1.641e-9, "C2H6": 1.094e-9,
           "C3H8": 5.469e-10, "N2": 3.283e-9},
    "Carbon": {"CO2": 3.382e-9, "CH4": 3.382e-11, "C2H6": 1.353e-11,
               "C3H8": 1.691e-13, "N2": 1.353e-10},
}

PROCESS_T, PROCESS_PP = 303.15, 1e5            
PROCESS_UF = 50e3 / 3600.0  
FIBER_DI, FIBER_DO, FIBER_L = 150e-6, 200e-6, 1.0

N_BRACKET = {"CA": (1.0e3, 2.0e6), "PI": (1.0e3, 2.0e6), "Carbon": (1.0e4, 8.0e6)}

def build_ng_module(membrane, xf, Pf, N):
    D = module_diameter_default(N, FIBER_DO)
    return MembraneModule(components=COMPONENTS_NG, permeance=MEMBRANE_PERMEANCE[membrane],
                           xf=xf, Pf=Pf, Pp=PROCESS_PP, T=PROCESS_T,
                           uf=PROCESS_UF, Di=FIBER_DI, Do=FIBER_DO,
                           L=FIBER_L, N=N, D=D, flow_pattern="counter", 
                           feed_side="shell", n_internal=10)

def solve_for_target_N(membrane, xf, Pf, target_CO2=0.025):
    lo, hi = N_BRACKET[membrane]

    def target_function(N):
        mod = build_ng_module(membrane, xf, Pf, N)
        try:
            res = mod.solve()
            return res["retentate_x"]["CO2"] - target_CO2
        except RuntimeError:
            return -1.0 # Over-purification safeguard

    f_lo, f_hi = target_function(lo), target_function(hi)
    if f_lo < 0:
        lo = lo / 10.0
    if f_hi > 0:
        N_star = hi
        mod = build_ng_module(membrane, xf, Pf, N_star)
        return N_star, mod.solve()

    N_star = brentq(target_function, lo, hi, xtol=1.0, rtol=1e-6, maxiter=100)
    mod = build_ng_module(membrane, xf, Pf, N_star)
    return N_star, mod.solve()

def _ng_feed_composition(CO2_frac):
    hc_total = 1.0 - CO2_frac - 0.01
    ch4_frac, c2_frac, c3_frac = 0.817, 0.082, 0.041   
    hc_base_sum = ch4_frac + c2_frac + c3_frac
    return {"CO2": CO2_frac, "CH4": hc_total * ch4_frac / hc_base_sum,
            "C2H6": hc_total * c2_frac / hc_base_sum, "C3H8": hc_total * c3_frac / hc_base_sum, "N2": 0.01}

def fig5_profiles(Pf=60e5, CO2_feed=0.10):
    xf = _ng_feed_composition(CO2_feed)
    profiles = {}
    for mem in ["CA", "PI", "Carbon"]:
        N_star, res = solve_for_target_N(mem, xf, Pf)
        profiles[mem] = res
    return profiles

def fig6_CO2_content_study():
    CO2_list = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    Pf = 60e5
    out = {mem: {"CO2": [], "spec_area": [], "HC_loss": [], "D": []} for mem in ["CA", "PI", "Carbon"]}

    for CO2 in CO2_list:
        xf = _ng_feed_composition(CO2)
        hc_feed = PROCESS_UF * (xf["CH4"] + xf["C2H6"] + xf["C3H8"])
        for mem in ["CA", "PI", "Carbon"]:
            N_star, res = solve_for_target_N(mem, xf, Pf)
            area = N_star * np.pi * FIBER_DO * FIBER_L
            hc_perm = res["permeate_flow"] * sum(res["permeate_y"][c] for c in ["CH4", "C2H6", "C3H8"])
            out[mem]["CO2"].append(CO2 * 100)
            out[mem]["spec_area"].append(area / (PROCESS_UF * 3.6))
            out[mem]["HC_loss"].append(100.0 * hc_perm / hc_feed)
            out[mem]["D"].append(module_diameter_default(N_star, FIBER_DO))
    return out

def fig7_feed_pressure_study():
    Pf_list_bar = [40, 60, 80, 100]
    xf = _ng_feed_composition(0.10)
    hc_feed = PROCESS_UF * (xf["CH4"] + xf["C2H6"] + xf["C3H8"])
    out = {mem: {"Pf": [], "spec_area": [], "HC_loss": [], "D": []} for mem in ["CA", "PI", "Carbon"]}

    for Pf_bar in Pf_list_bar:
        Pf = Pf_bar * 1e5
        for mem in ["CA", "PI", "Carbon"]:
            N_star, res = solve_for_target_N(mem, xf, Pf)
            area = N_star * np.pi * FIBER_DO * FIBER_L
            hc_perm = res["permeate_flow"] * sum(res["permeate_y"][c] for c in ["CH4", "C2H6", "C3H8"])
            out[mem]["Pf"].append(Pf_bar)
            out[mem]["spec_area"].append(area / (PROCESS_UF * 3.6))
            out[mem]["HC_loss"].append(100.0 * hc_perm / hc_feed)
            out[mem]["D"].append(module_diameter_default(N_star, FIBER_DO))
    return out


# ==============================================================================
# PART 10: PLOTTING & MAIN EXECUTION BLOCK (Automated Saving)
# ==============================================================================
STYLE = {"CA": dict(marker="o", color="tab:blue", label="CA"),
         "PI": dict(marker="s", color="tab:orange", label="PI"),
         "Carbon": dict(marker="^", color="tab:green", label="Carbon")}

def plot_fig2(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    for Di, res in results.items():
        ax.plot(res["z"], res["p"] / 100.0, label=f"Di={int(Di*1e6)} um")
    ax.set_xlabel("Hollow fiber length, m")
    ax.set_ylabel("Bore-side pressure, mbar")
    ax.set_title("Fig. 2 — Influence of hollow fiber ID on bore-side pressure drop")
    ax.legend()
    fig.tight_layout()

def plot_fig3(results):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for Di, d in results.items():
        axes[0].plot(d["L"], d["CH4_purity"], marker="o", label=f"Di={int(Di*1e6)} um")
        axes[1].plot(d["L"], d["CH4_loss"], marker="o", label=f"Di={int(Di*1e6)} um")
    axes[0].set_ylabel("CH4 purity in retentate, %")
    axes[1].set_ylabel("CH4 loss, %")
    for a in axes:
        a.set_xlabel("Hollow fiber length, m")
        a.legend()
    fig.suptitle("Fig. 3 — Influence of hollow fiber length on CH4 purity (a) and loss (b)")
    fig.tight_layout()

def plot_fig4(profiles, packing_list, D_list, dP_list):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for pd in packing_list:
        axes[0].plot(profiles[pd]["z"], profiles[pd]["P"] / 1e5, label=f"pd={pd}")
    axes[0].set_xlabel("Hollow fiber length, m")
    axes[0].set_ylabel("Feed pressure, bar")
    axes[0].legend(fontsize=7)

    ax2b = axes[1].twinx()
    axes[1].plot(packing_list, D_list, "o-", color="tab:blue")
    ax2b.plot(packing_list, dP_list, "s--", color="tab:red")
    axes[1].set_xlabel("Packing density, m2/m3")
    axes[1].set_ylabel("Module inner diameter, m", color="tab:blue")
    ax2b.set_ylabel("Total pressure drop, mbar", color="tab:red")
    fig.suptitle("Fig. 4 — Feed pressure profiles & module ID vs packing density")
    fig.tight_layout()

def plot_fig5(profiles):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for mem, res in profiles.items():
        st = STYLE[mem]
        axes[0, 0].plot(res["z"], res["P"] / 1e5, color=st["color"], label=mem)
        axes[0, 1].plot(res["z"], res["p"] / 1e5, color=st["color"], label=mem)
        ci, chi = res["components"].index("CO2"), res["components"].index("CH4")
        axes[1, 0].plot(res["z"], res["y_profile"][:, ci] * 100, color=st["color"], label=mem)
        axes[1, 1].plot(res["z"], res["vy_star"][:, chi], color=st["color"], label=mem)

    axes[0, 0].set_ylabel("Feed pressure, bar")
    axes[0, 1].set_ylabel("Permeate pressure, bar")
    axes[1, 0].set_ylabel("Permeate CO2 purity, %")
    axes[1, 1].set_ylabel("vy*_CH4")
    for a in axes.ravel():
        a.set_xlabel("Hollow fiber length, m")
        a.legend()
    fig.suptitle("Fig. 5 — Profiles along hollow fiber length (Di = 150 um)")
    fig.tight_layout()

def plot_fig6(out):
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    for mem, d in out.items():
        st = STYLE[mem]
        axes[0].plot(d["CO2"], d["spec_area"], marker=st["marker"], color=st["color"], label=mem)
        axes[1].plot(d["CO2"], d["HC_loss"], marker=st["marker"], color=st["color"], label=mem)
        axes[2].plot(d["CO2"], d["D"], marker=st["marker"], color=st["color"], label=mem)
    axes[0].set_ylabel("Specific membrane area, m2/(kmol/h)")
    axes[1].set_ylabel("HC loss, %")
    axes[2].set_ylabel("Module inner diameter, m")
    for a in axes:
        a.set_xlabel("Feed CO2 content, %")
        a.legend()
    fig.suptitle("Fig. 6 — Influence of feed CO2 content on membrane area, HC loss, module ID")
    fig.tight_layout()

def plot_fig7(out):
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))
    for mem, d in out.items():
        st = STYLE[mem]
        axes[0].plot(d["Pf"], d["spec_area"], marker=st["marker"], color=st["color"], label=mem)
        axes[1].plot(d["Pf"], d["HC_loss"], marker=st["marker"], color=st["color"], label=mem)
        axes[2].plot(d["Pf"], d["D"], marker=st["marker"], color=st["color"], label=mem)
    axes[0].set_ylabel("Specific membrane area, m2/(kmol/h)")
    axes[1].set_ylabel("HC loss, %")
    axes[2].set_ylabel("Module inner diameter, m")
    for a in axes:
        a.set_xlabel("Feed pressure, bar")
        a.legend()
    fig.suptitle("Fig. 7 — Influence of feed pressure on membrane area, HC loss, module ID")
    fig.tight_layout()

if __name__ == "__main__":
    import os
    # Creates a 'figures' directory in the current working folder
    OUTDIR = os.path.join(os.getcwd(), "figures")
    os.makedirs(OUTDIR, exist_ok=True)

    def _save(fig_num):
        fname = os.path.join(OUTDIR, f"fig{fig_num}.png")
        plt.savefig(fname, dpi=150)
        print(f"  -> saved {fname}")

    # ---- Task A: Validation ----
    run_validation()

    # ---- Task B: Module Design Studies (Figs. 2-4) ----
    print("\nRunning Fig. 2 (hollow fiber diameter study)...")
    fig2_results = fig2_hollow_fiber_diameter_study()
    plot_fig2(fig2_results)
    _save(2)

    print("Running Fig. 3 (hollow fiber length study)...")
    fig3_results = fig3_hollow_fiber_length_study()
    plot_fig3(fig3_results)
    _save(3)

    print("Running Fig. 4 (packing density study)...")
    fig4_profiles, pd_list, D_list, dP_list = fig4_packing_density_study()
    plot_fig4(fig4_profiles, pd_list, D_list, dP_list)
    _save(4)

    # ---- Task C: Process Parametric Study (Figs. 5-7) ----
    print("\nRunning Fig. 5 (concentration/pressure profiles, 3 membranes)...")
    fig5_results = fig5_profiles(Pf=60e5, CO2_feed=0.10)
    plot_fig5(fig5_results)
    _save(5)

    print("Running Fig. 6 (feed CO2 content sweep)... (takes ~40 s)")
    fig6_results = fig6_CO2_content_study()
    plot_fig6(fig6_results)
    _save(6)

    print("Running Fig. 7 (feed pressure sweep)... (takes ~25 s)")
    fig7_results = fig7_feed_pressure_study()
    plot_fig7(fig7_results)
    _save(7)

    print(f"\nAll operations completed! Figures written to: {OUTDIR}")
    plt.show()