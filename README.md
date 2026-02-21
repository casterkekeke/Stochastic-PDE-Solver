# Feynman-Kac Engine: Finite Difference Suite

This repository contains a high-performance numerical engine for solving the Black-Scholes Partial Differential Equation (PDE). 
The current implementation focuses on Finite Difference Methods (FDM) to price European options and analyze their Greeks.
Theoretical Foundation

The Black-Scholes PDE is defined as:

#### $$\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0$$

To solve this numerically, we discretize the spatial domain $S$ into $M$ steps and the temporal domain $t$ into $N$ steps. 
We define $V_j^n$ as the option price at time step $n$ and asset price level $j$.#

## Implemented Numerical Schemes


1.  ### Explicit Method
    
    The Explicit scheme uses a forward difference in time and central differences in space. 
    It is computationally fast but conditionally stable, requiring the CFL (Courant-Friedrichs-Lewy) condition to be met to prevent numerical explosion.
    #### Discretization:
    $$\frac{V_j^{n+1} - V_j^n}{\Delta t} + rS_j \frac{V_{j+1}^{n+1} - V_{j-1}^{n+1}}{2\Delta S} + \frac{1}{2}\sigma^2 S_j^2 \frac{V_{j+1}^{n+1} - 2V_j^{n+1} + V_{j-1}^{n+1}}{\Delta S^2} = rV_j^{n+1}$$

3. ### Implicit Method

    The Implicit scheme uses a backward difference in time. 
    This results in a system of linear equations that must be solved at each time step. 
    Its primary advantage is that it is unconditionally stable, allowing for much larger time steps $\Delta t$ than the explicit method.
    #### Discretization:
   $$\frac{V_j^{n+1} - V_j^n}{\Delta t} + rS_j \frac{V_{j+1}^{n} - V_{j-1}^{n}}{2\Delta S} + \frac{1}{2}\sigma^2 S_j^2 \frac{V_{j+1}^{n} - 2V_j^{n} + V_{j-1}^{n}}{\Delta S^2} = rV_j^{n}$$

4. ### Crank-Nicolson Method

    The Crank-Nicolson scheme is the industry standard for 1D PDEs. 
    It averages the Explicit and Implicit operators, achieving second-order accuracy in time $O(\Delta t^2)$. 
    Like the Implicit method, it is unconditionally stable but requires solving a tridiagonal system at each step.

    #### Discretization:
    $$\frac{V_j^{n+1} - V_j^n}{\Delta t} + \frac{1}{2} \left( \mathcal{L}V_j^{n+1} + \mathcal{L}V_j^{n} \right) = 0$$

    Where $\mathcal{L}$ is the spatial differential operator. In matrix form, this is solved as:
    #### $$(I - \frac{\Delta t}{2}A)V^n = (I + \frac{\Delta t}{2}A)V^{n+1}$$

## Numerical Discretization and Matrix Derivation

This section details the formal transition from the continuous Black-Scholes PDE to the discrete linear systems utilized in the fdm_solver.py engine.
1.  ### Spatial Discretization 

    We define a uniform spatial grid where the asset price $S$ is discretized into $M$ intervals of size $\Delta S$, such that $S_j = j \Delta S$ for $j \in \{0, 1, \dots, M\}$. 
    To transform the PDE into a system of algebraic equations, we apply second-order central difference approximations to the spatial derivatives at each interior node $j$:
    #### Delta Approximation ($\frac{\partial V}{\partial S}$):$$\frac{V_{j+1} - V_{j-1}}{2\Delta S}$$
    #### Gamma Approximation ($\frac{\partial^2 V}{\partial S^2}$):$$\frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2}$$
 
2. ### Derivation of the Spatial Operator $\mathcal{L}V_j$

    Substituting these finite difference stencils into the Black-Scholes PDE and evaluating at price level $S_j$:
    $$\frac{\partial V}{\partial t} + \underbrace{r(j \Delta S) \left[ \frac{V_{j+1} - V_{j-1}}{2\Delta S} \right] + \frac{1}{2}\sigma^2 (j \Delta S)^2 \left[ \frac{V_{j+1} - 2V_j + V_{j-1}}{\Delta S^2} \right] - rV_j}_{\mathcal{L}V_j} = 0$$
    By canceling the $\Delta S$ terms and grouping the coefficients by their spatial index $j$, we define the operator $\mathcal{L}V_j = a_j V_{j-1} + b_j V_j + c_j V_{j+1}$ with the following system coefficients:

   > **System Coefficients for Matrix A:**
    >   * **$a_j$ (Lower Diagonal):** $\frac{1}{2}j(\sigma^2 j - r)$
    >   * **$b_j$ (Main Diagonal):** $-(\sigma^2 j^2 + r)$
    >   * **$c_j$ (Upper Diagonal):** $\frac{1}{2}j(\sigma^2 j + r)$


3. ### Application to Time-Stepping Schemes
    
    The matrix $\mathbf{A}$ is a tridiagonal matrix constructed from the coefficients $(a_j, b_j, c_j)$. 
    The Identity matrix $\mathbf{I}$ represents the current state. 
    The numerical operator is applied differently across the three solvers to transition from time $n+1$ (maturity) back to time $n$:

    #### Summary of Temporal Operators
    
    | Method | Algebraic Operator Form | Role of $\mathcal{L}$ |
    | :--- | :--- | :--- |
    | **Explicit** | $V^n = (\mathbf{I} + \Delta t \mathbf{A}) V^{n+1}$ | Linear projection (weighted sum) of known values. |
    | **Implicit** | $(\mathbf{I} - \Delta t \mathbf{A}) V^n = V^{n+1}$ | Inversion of the spatial system via matrix solve. |
    | **Crank-Nicolson** | $(\mathbf{I} - \frac{\Delta t}{2} \mathbf{A}) V^n = (\mathbf{I} + \frac{\Delta t}{2} \mathbf{A}) V^{n+1}$ | Averaged operator for $O(\Delta t^2)$ accuracy. |
    
4.    ### Variable Definitions
      | Variable | Description |
    | :--- | :--- |
    | $V_j^n$ | Option value at time step $n$ and price node $j$. |
    | $S_j$ | Asset price at node $j$, defined as $j \cdot \Delta S$. |
    | $\Delta t, \Delta S$ | Step sizes for time and space discretization, respectively. |
    | $r, \sigma$ | Risk-free interest rate and asset volatility. |
    | $\mathbf{A}$ | **Transition Matrix**: Tridiagonal matrix containing PDE coefficients. |
    | $\mathbf{I}$ | **Identity Matrix**: Square matrix with ones on the diagonal, zeros elsewhere. |
    | $\mathcal{L}$ | **Infinitesimal Generator**: The discrete spatial differential operator. |
