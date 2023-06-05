# main code

using LinearAlgebra
using Plotly

include("InitialFunctions.jl")

## Definition of basic parameters

# Courant number
c = 1.0

# Grid settings
xL = - pi / 2
xR = pi
Nx = 160
h = (xR - xL) / Nx

# Velocity
u = 2.0

# Time
tau = c * h / u
# Ntau = Int(Nx / 10)
Ntau = 16

# Initial condition
phi_0(x) = piecewiseConstant(x);

# Exact solution
phi_exact(x, t) = phi_0(x - u * t);

## Comptutation

# Grid initialization
x = range(xL, xR, length = Nx + 1)
phi = zeros(Nx + 1, Ntau + 1);
phi_predictor = zeros(Nx + 1, Ntau + 1);
phi_firts_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor[:, 1] = phi_0.(x);
phi_firts_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi[2, :] = phi_exact.(x[2], range(0, Ntau * tau, length = Ntau + 1));
phi[Nx + 1, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));

phi_firts_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_firts_order[2, :] = phi_exact.(x[2], range(0, Ntau * tau, length = Ntau + 1));
phi_firts_order[Nx + 1, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));

# ENO parameters
w = 1 / 2;

# Space loop
for i = 3:Nx
    # Time loop
    for n = 1:Ntau

        # First order solution
        phi_firts_order[i, n + 1] = ( phi_firts_order[i, n] + c * phi_firts_order[i - 1, n + 1] ) / ( 1 + c );
        
        # Predictor
        phi_predictor[i, n + 1] = ( phi_predictor[i, n] + c * phi[i - 1, n + 1] ) / ( 1 + c );

        # Corrector
        r_downwind = - phi_predictor[i, n + 1] + phi_predictor[i + 1, n] - phi[i, n] + phi[i - 1, n + 1];
        r_upwind = phi[i, n] - phi[i - 1, n] - phi[i - 1, n + 1] + phi[i - 2, n + 1];

        # Second order solution
        phi[i, n + 1] = ( phi[i, n] + c * phi[i - 1, n + 1] - 0.5 * ( 1 - w ) * r_downwind - 0.5 * w * r_upwind ) / ( 1 + c );

        # Predictor for next time step
        phi_predictor[i + 1, n + 1] = ( phi_predictor[i + 1, n] + c * (phi[i, n + 1] ) ) / ( 1 + c );

    end
end

# Plot of the result at the final time together with the exact solution

trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Compact scheme solution")
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact solution")
trace3 = scatter(x = x, y = phi_firts_order[:, end], mode = "lines", name = "First order solution")

layout = Layout(title = "Linear advection equation", xaxis_title = "x", yaxis_title = "phi")

plot([trace1, trace2, trace3], layout)





