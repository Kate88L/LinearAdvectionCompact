# main code

using LinearAlgebra
using Plots

## Definition of basic parameters

# Courant number
c = 1.0

# Grid settings
xL = - pi / 2
xR = 7 * pi / 2
Nx = 10
h = (xR - xL) / Nx

# Velocity
u = 2.0

# Time
tau = c * h / u
Ntau = 10

# Initial condition
phi_0(x) = sin(x);

# Exact solution
phi_exact(x, t) = phi_0(x - u * t);

## Comptutation

# Grid initialization
x = range(xL, xR, length = Nx + 1);
phi = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);

# Boundary conditions (cyclic)
phi[1, :] = phi[end, :];
phi[2, :] = phi[end - 1, :];

# Space loop
for i = 2:Nx
    # Time loop
    for n = 1:Ntau
        


    end
end



