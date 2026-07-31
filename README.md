# Hollow Fiber Membrane Module Simulator

Python-based numerical simulator for $\text{CO}_2$ removal from natural gas using hollow fiber membrane modules, reproducing the mathematical model by **Chu et al. (2019)**.

## Features
- **Orthogonal Collocation Method:** Transforms stiff ODEs into algebraic systems using Legendre polynomial roots.
- **Boundary Singularity Resolution:** Applies L'Hôpital's rule at the closed permeate end ($z=0$) to eliminate artificial concentration jumps.
- **Homotopy Continuation:** Implements a permeance-ramping technique for robust solver convergence.
- **Adaptive Sizing:** Uses bisection search (`scipy.optimize.brentq`) to optimize fiber count ($N$) for target pipeline specifications ($2.5 \text{ mol\% } \text{CO}_2$).

## Project Structure
- `simulator.py`: Main Python script containing the numerical engine and parametric studies.
- `figures/`: Generated validation and parametric study plots (Figs. 2-7).
- `report/`: Complete LaTeX report (in Persian and English).
