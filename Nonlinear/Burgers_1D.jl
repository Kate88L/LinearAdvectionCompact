# Compact finite difference scheme for the 1D Burgers' equation

using NLsolve
using LinearAlgebra
using PlotlyJS

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")


## Definition of basic parameters

# Burger's equation 
H(x) = ( x.^2 ) / 2.0

# Initial condition
phi_0(x) = trafficJam(x)

# Mesh
xL = -1
xR = 1

level = 4 # Level of refinement
Nx = 100 * 2^level
h = (xR - xL) / Nx

x = range(xL, xR, length = Nx + 1)

# Time
T = 1
Nτ = 10 * 2^level
τ = T / Nτ

## Comptutation
phi = zeros(Nx + 1, Nτ + 1)

# Initial condition
phi[:, 1] = phi_0.(x)

# Time loop

for n = 1:Nτ

    # Boundary conditions
    phi[1, n + 1] = phi[1, 1]

    for i = 2:Nx + 1
        # First order scheme
        function firstOrderScheme(S, x)
            S[1] = x[1] - phi[i, n] + τ * H( (x[1] - phi[i - 1, n + 1]) / h )
        end

        solution = nlsolve(firstOrderScheme, [phi[i, n]]) 
        phi[i, n + 1] = solution.zero[1];
    end

end

## Compute the numerical derivative of the solution
∂x_phi = zeros(Nx + 1, Nτ + 1)

for n = 1:Nτ + 1
    ∂x_phi[1, n] = (phi[2, n] - phi[1, n]) / h
    for i = 2:Nx + 1
        ∂x_phi[i, n] = (phi[i, n] - phi[i - 1, n]) / h
    end
end

## Plot of the result 

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "First order scheme", line=attr(color="firebrick", width=2))

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([trace1], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = ∂x_phi[:, end], mode = "lines", name = "First order sol. gradient")

plot_phi_d = plot([trace1_d], layout)

p = [plot_phi; plot_phi_d]
relayout!(p, width = 1000, height = 500)
p