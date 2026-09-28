# Markowitz Portfolio Optimization with No-Short-Selling Constraints (KKT System & Quadratic Programming) - R Studio
A production-grade implementation of Markowitz Mean-Variance Optimization under non-negativity (no-short-selling) KKT constraints, solved via the **Goldfarb-Idnani Dual Active-Set Method** using R `quadprog` package.
## Project Overview
This repository demonstrates the end-to-end mathematical formulation, KKT system verification, and numerical solution of a 4-asset portfolio allocation problem (Money Market, Capital Stable, Balance, Growth) over historical fund data (2004–2013).
* **Objective:** Minimize annualized portfolio risk (variance) subject to a targeted annualized return of 8.00%.
* **Boundary Constraints:** Realistic institutional long-only setting ($`w_i \ge 0`$, no short-selling).
* **Core Theoretical Bridge:** Validating Karush-Kuhn-Tucker (KKT) stationarity, primal feasibility, dual feasibility, and complementary slackness directly against numerical solver outputs.
---
## Mathematical & Theoretical Derivation
### 1. Quadratic Programming (QP) Standard Form
In continuous portfolio optimization, the standard convex quadratic minimization formulation is defined as:

```math
\min_{\mathbf{w}} \left( \frac{1}{2}\mathbf{w}^{\top}\mathbf{D}\mathbf{w} - \mathbf{d}^{\top}\mathbf{w} \right)
```

Where:
* $`\mathbf{w} = (w_1, w_2, w_3, w_4)^{\top}`$ is the portfolio weight allocation vector.
* $`\mathbf{D} = \boldsymbol{\Sigma}`$ is the $`4 \times 4`$ covariance matrix, constructed via:

```math
\boldsymbol{\Sigma} = \text{diag}(\boldsymbol{\sigma})\,\mathbf{C}\,\text{diag}(\boldsymbol{\sigma})
```

  where $`\mathbf{C}`$ is the correlation matrix and $`\boldsymbol{\sigma}`$ is the annualized volatility vector.
* $`\mathbf{d} = \mathbf{0}_{4 \times 1}`$ denotes pure variance minimization.
---
### 2. Constraints & Karush-Kuhn-Tucker (KKT) System
The portfolio selection problem is governed by:

```math
\min \quad \frac{1}{2}\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}
```

```math
\text{s.t.} \quad
\begin{cases}
\boldsymbol{\mu}^{\top}\mathbf{w} = 0.08 & \text{(Target Return Constraint)} \\
\mathbf{1}^{\top}\mathbf{w} = 1 & \text{(Full Investment Budget)} \\
w_i \ge 0, \quad \forall i \in \{1, 2, 3, 4\} & \text{(No-Short-Selling Bounds)}
\end{cases}
```

Because $`\boldsymbol{\Sigma} \succ 0`$ (strictly positive definite), the objective function is strictly convex over a convex polyhedral feasible set. The KKT first-order conditions provide both **necessary and sufficient** criteria for the unique global minimum:
1. **Stationarity (Gradient Balance)**:

   ```math
   \boldsymbol{\Sigma}\mathbf{w} - \lambda_1\boldsymbol{\mu} - \lambda_2\mathbf{1} - \mathbf{v} = \mathbf{0}
   ```

2. **Primal Feasibility (Constraint Satisfaction)**:

   ```math
   \boldsymbol{\mu}^{\top}\mathbf{w} = 0.08, \quad \mathbf{1}^{\top}\mathbf{w} = 1, \quad w_i \ge 0 \quad (\forall i)
   ```

3. **Dual Feasibility (Inward Boundary Forces)**:

   ```math
   v_i \ge 0 \quad (\forall i \in \{1, 2, 3, 4\})
   ```

4. **Complementary Slackness**:

   ```math
   w_i \cdot v_i = 0 \quad (\forall i \in \{1, 2, 3, 4\})
   ```

---
### 3. Solver Interface Alignment (`solve.QP`)
The `solve.QP` engine implements the **Goldfarb-Idnani Dual Active-Set Method**. The internal Fortran routine expects constraints in the canonical transposed format:

```math
\mathbf{A}^{\top}\mathbf{w} \ge \mathbf{b}
```

To enforce the first $`m_{\text{eq}} = 2`$ constraints as strict equalities, $`\mathbf{A}_{4 \times 6}`$ and $`\mathbf{b}_{6 \times 1}`$ are formulated such that each column of $`\mathbf{A}`$ maps to one linear constraint:

```math
\mathbf{A} = \begin{pmatrix}
\mu_1 & 1 & 1 & 0 & 0 & 0 \\
\mu_2 & 1 & 0 & 1 & 0 & 0 \\
\mu_3 & 1 & 0 & 0 & 1 & 0 \\
\mu_4 & 1 & 0 & 0 & 0 & 1
\end{pmatrix}_{4 \times 6},
\quad
\mathbf{b} = \begin{pmatrix}
0.08 \\
1.00 \\
0 \\
0 \\
0 \\
0
\end{pmatrix}_{6 \times 1}
```

In R:
```R
Amat <- cbind(mu, rep(1, 4), diag(4))
bvec <- c(0.08, 1.00, rep(0, 4))
meq  <- 2
```
---
## Execution & Implementation
### Prerequisites
Install the R package("quadprog")
```r
if (!require("quadprog")) install.packages("quadprog")
library(quadprog)
```
### Run Script
```r
source("portfolio_optimization.R")
```
---
## Numerical Output & Optimization Report
### When executed, the console produces the structured output:
```text
=======================================================
      Markowitz Portfolio Optimization Results (QP)    
=======================================================
1. Optimal Portfolio Weights (w_i >= 0)
  Money Market Capital Stable        Balance         Growth 
        0.0000         0.4512         0.5488         0.0000 
2. Portfolio Performance Metrics
Target Expected Return :   8.00%
Actual Expected Return :   8.00%
Annualized Risk (sd)   :  10.81%
3. Dual Variables & KKT Multipliers
Equality Constraints: Shadow Prices (lambda)
   Target Return (lambda_1) Budget Constraint (lambda_2) 
                   0.254789                     0.008707
Inequality Constraints: Non-negativity Forces (v_i >= 0)
  Money Market Capital Stable        Balance         Growth 
      0.004246       0.000000       0.000000       0.000197 
---
=======================================================
KKT Complementary Slackness Check: VERIFIED (w_i * v_i = 0 holds)
=======================================================
```

## Economic & Quantitative Conclusion
### 1. Dual Active-Set Identification & Complementary Slackness
**Interior Assets (Capital Stable: $`45.12\%`$, Balance: $`54.88\%`$)**:
Both assets are assigned strictly positive weights. As mathematically mandated by KKT Complementary Slackness ($`w_i \cdot v_i = 0`$), their boundary reaction forces are exactly zero ($`v_2 = 0, v_3 = 0`$). They operate freely within the interior of the feasible convex polytope.

**Boundary Assets (Money Market: $`0.00\%`$, Growth: $`0.00\%`$)**:
Both assets hit the non-negativity barrier ($`w_1 = 0, w_4 = 0`$). Their corresponding dual multipliers are strictly non-negative ($`v_1 = 0.004246, v_4 = 0.000197`$). This confirms dual feasibility: the no-short-sale walls prevent the solver from assigning negative weights to these assets to artificially suppress portfolio variance.

### 2. Financial Interpretation of Asset Allocation
**Redundancy of Growth Fund**:
While Growth offers an expected return of $`10.30\%`$, it incurs high volatility ($`16.64\%`$) and near-perfect correlation with Balance ($`\rho_{34} = 0.997`$). The optimal trade-off combines Capital Stable ($`\mu = 6.65\%`$) and Balance ($`\mu = 9.11\%`$), which neatly brackets the target return ($`8.00\%`$) while capturing significant diversification benefits, rendering Growth redundant.

**Economic Interpretation of Shadow Prices**:
The Lagrangian multiplier $`\lambda_1 = 0.254789`$ reflects the marginal variance cost of the target return constraint:

```math
\lambda_1 = \frac{\partial \bigl(\frac{1}{2}\sigma_p^2\bigr)}{\partial \mu_0}
```

Increasing the target return by $`1.00\%`$ (i.e. $`\Delta \mu_0 = 0.01`$) incurs an approximate increase in portfolio variance of:

```math
\Delta \left( \frac{1}{2}\sigma_p^2 \right) \approx 0.254789 \times 0.01 \approx 0.002548
```

## Repository Structure
```text
├── portfolio_optimization.R   # Production-ready R script with automated KKT verification
└── README.md                  # Complete mathematical derivation and analytical report
```
