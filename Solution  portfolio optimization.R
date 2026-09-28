# ==============================================================================
# Target Return with No Short selling (Markowitz Model + Kuhn-Tucker Conditions)
# Solved via Quadratic Programming in R using Goldfarb-Idnani Dual Active Set Method
# ==============================================================================
# 1. Load Required Libraries
if (!require("quadprog")) install.packages("quadprog")
library(quadprog)

# 2. Input Historical Data (Sep 2004 - Aug 2013)
fund_names <- c("Money Market", "Capital Stable", "Balance", "Growth")
mu <- c(0.0173, 0.0665, 0.0911, 0.1030)  # annualized expected return
sigma <- c(0.0056, 0.0777, 0.1348, 0.1664)  # annualized volatility(risk)

# Construct correlation matrix:
# By symmetry, only the lower triangular elements(6 correlation coefficients) are required:
corr_mat <- matrix(0,nrow = 4,ncol = 4)
corr_mat[lower.tri(corr_mat)] <- c(-0.070,-0.095,-0.095,0.959,0.936,0.997)# lower.tri() fills column-by-column by default
corr_mat <- corr_mat + t(corr_mat) +diag(4)

# 3. Calculate covariance matrix and ensure numerical symmetry
Sigma <- diag(sigma) %*% corr_mat %*% diag(sigma)
rownames(Sigma) <- colnames(Sigma) <- fund_names

# To ensure exact numerical symmetry to avoid round-off errors and to pass strict positive-definiteness check of solve.QP 
Sigma <- (Sigma + t(Sigma)) / 2

# 4. Formulate Quadratic Programming Problem (KKT Optimization Formulation):
# Standard QP Objective: min (1/2) * w' * Dmat * w - dvec' * w
# For pure variance minimization, Dmat = Sigma, and linear cost vector dvec = 0
Dmat <- Sigma
dvec <- rep(0, 4) # a one-dimension vector

# Theoretical Foundation:
# Since Sigma is strictly positive definite, the objective function is strictly convex.
# The Karush-Kuhn-Tucker (KKT) first-order conditions serve as both necessary and
# sufficient conditions for the unique global minimum variance portfolio.

# Constraints Definition:
# (1) Target Expected Return : mu' * w = 0.08
# (2) Full Investment Budget : 1'  * w = 1.00
# (3-6) No-Short-Sales Bounds: w_i    >= 0 (for i = 1, 2, 3, 4)

# solve.QP format expects constraints in the form as: t(Amat) %*% w >= bvec.
# Therefore, each column of Amat corresponds to one constraint row:
# Col 1: mu              ->  mu' * w  = 0.08 (Equality)
# Col 2: rep(1, 4)       ->  1'  * w  = 1.00 (Equality)
# Col 3-6: diag(4)       ->  I   * w >= 0    (Inequality, w_i >= 0)
Amat <- cbind(mu, rep(1, 4), diag(4))
bvec <- c(0.08, 1, rep(0, 4))
meq  <- 2# The first 2 equations are equal

# 5. Solve via Goldfarb-Idnani Dual Active-Set Method:
solution <- solve.QP(Dmat = Dmat, dvec = dvec, Amat = Amat, bvec = bvec, meq = meq)

# 6. Extract & Interpret Results
weights <- round(solution$solution, 4)
names(weights) <- fund_names

port_return <- sum(weights * mu)
port_risk <- sqrt(as.numeric(t(weights) %*% Sigma %*% weights))# do not use sd() because it's not a time series data, as.numeric() to convert a 1x1 matrix into a scalar

# Structure Lagrangian multipliers:
# - Indices 1:2 correspond to equality multipliers (lambda: shadow prices of constraints)
# - Indices 3:6 correspond to inequality multipliers (v_i: non-negativity boundary push forces)
lambda_eq <- solution$Lagrangian[1:2]
names(lambda_eq) <- c("Target Return (lambda_1)", "Budget Constraint (lambda_2)")

v_ineq <- solution$Lagrangian[3:6]
names(v_ineq) <- fund_names

# 7. Print Formatted Optimization Report
cat("\n=======================================================\n")
cat("      Markowitz Portfolio Optimization Results (QP)    \n")
cat("=======================================================\n\n")

cat("1. Optimal Portfolio Weights (w_i >= 0)\n")
print(weights)
cat("\n")

cat("2. Portfolio Performance Metrics\n")
cat(sprintf("Target Expected Return : %6.2f%%\n", 0.08 * 100))
cat(sprintf("Actual Expected Return : %6.2f%%\n", port_return * 100))
cat(sprintf("Annualized Risk (sd)   : %6.2f%%\n\n", port_risk * 100))

cat("3. Dual Variables & KKT Multipliers\n")
cat("Equality Constraints: Shadow Prices (lambda)\n")
print(round(lambda_eq, 6))
cat("\n")

cat("Inequality Constraints: Non-negativity Forces (v_i >= 0)\n")
print(round(v_ineq, 6))
cat("\n")

# Verify KKT Complementary Slackness: w_i * v_i == 0
# is_slack_valid is a single boolean, it's TRUE only when all weights * v_ineq = 0
is_slack_valid <- all(abs(weights * v_ineq) < 1e-6)
cat(sprintf("KKT Complementary Slackness Check: %s\n", 
            ifelse(is_slack_valid, 
                   "VERIFIED (w_i * v_i = 0 holds strictly)",
                   "FAILED")))