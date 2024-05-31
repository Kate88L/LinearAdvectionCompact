# Compact finite difference scheme for the 1D Burgers' equation

using NLsolve
using LinearAlgebra
using PlotlyJS
using QuadGK
using Trapz
using Interpolations

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")


## Definition of basic parameters

# Burger's equation 
H(x) = ( x.^2 ) / 2.0

# Mesh
xL = 0
xR = 1

level = 1 # Level of refinement
Nx = 100 * 2^level
h = (xR - xL) / Nx

x = range(xL, xR, length = Nx + 1)

# Initial condition
F = integrate_function(burgersLozanoAslam, xL, xR)
# phi_0(x) = simpleBurgers(x)
phi_0(x) = F(x)

# Time
T = 0.25
Nτ = 10 * 2^level
τ = T / Nτ

# Nτ = 1

## Comptutation
phi = zeros(Nx + 1, Nτ + 1)
phi_first_order = zeros(Nx + 1, Nτ + 1)
phi_predictor = zeros(Nx + 1, Nτ + 1)
phi_predictor_n2 = zeros(Nx + 1, Nτ + 1)

# Initial condition
phi[:, 1] = phi_0.(x)
phi_predictor[:, 1] = phi_0.(x)
phi_first_order[:, 1] = phi_0.(x)

# phi[:, 1] = F[:]
# phi_first_order[:, 1] = F[:]
# phi_predictor[:, 1] = F[:]

ω = zeros(Nx + 1) .+ 1/3;

# Compute the exact solution using the method of characteristics
phi_exact = zeros(Nx + 1, Nτ + 1)
for n = 1:Nτ + 1
    function exactLozanoAslam_t(x)
        return exactLozanoAslam(x, (n-1) * τ)
    end
    phi_e = integrate_function(exactLozanoAslam_t, xL, xR)
    phi_exact[:, n] = phi_e(x)
end

function exactLozanoAslam_t(x)
    return exactLozanoAslam(x, -τ)
end
phi_e = integrate_function(exactLozanoAslam_t, xL, xR)
phi_old = phi_e(x)

# Time loop
for n = 1:Nτ

    # Boundary conditions
    phi[1, n + 1] = phi[1, n]
    phi_first_order[1, n + 1] = phi_first_order[1, n]
    phi_predictor[1, n + 1] = phi_predictor[1, n]
    phi_predictor_n2[1, n + 1] = phi_predictor_n2[1, n]

    if n > 1
        global phi_old = phi[:, n - 1]
    end

    for i = 2:Nx + 1
        # First order scheme
        function firstOrderScheme(S, u)
            S[1] = u[1] - phi_first_order[i, n] + τ * H( (u[1] - phi_first_order[i - 1, n + 1]) / h )
        end

        # Predictor
        function firstOrderPredictor(S, u)
            S[1] = u[1] - phi[i, n] + τ * H( (u[1] - phi_predictor[i - 1, n + 1]) / h )
        end

        prediction = nlsolve(firstOrderScheme, [phi_predictor[i, n]]);
        phi_predictor[i, n + 1] = prediction.zero[1];
        phi_predictor_n2[i, n] = phi_predictor[i, n + 1];

        r_downwind_n = phi_predictor[i, n] - phi_predictor[i - 1, n + 1] - phi_predictor[i, n + 1] + phi_predictor_n2[i - 1, n + 1];
        r_upwind_n = phi[i - 1, n] + phi[i - 1, n + 1] - phi[i, n] - phi_old[i];

        # Second order scheme
        function secondOrderScheme(S, x)
            S[1] = x[1] - phi[i, n] + τ * H( (x[1] - phi[i - 1, n + 1]) / h  +  ω[i] * r_upwind_n +  (1 - ω[i]) * r_downwind_n )
        end

        # First order solution
        solution = nlsolve(firstOrderScheme, [phi_first_order[i, n]]) 
        phi_first_order[i, n + 1] = solution.zero[1];

        # Second order solution
        solution = nlsolve(secondOrderScheme, [phi[i, n]])
        phi[i, n + 1] = solution.zero[1];

        # Predictor
        function firstOrderPredictorFuture(S, u)
            S[1] = u[1] - phi[i, n + 1] + τ * H( (u[1] - phi_predictor_n2[i - 1, n + 1]) / h )
        end

        prediction = nlsolve(firstOrderPredictorFuture, [phi_predictor_n2[i, n]]);
        phi_predictor_n2[i, n + 1] = prediction.zero[1];
    end

end

## Compute the numerical derivative of the solution
∂x_phi_first_order = zeros(Nx + 1, Nτ + 1)
∂x_phi = zeros(Nx + 1, Nτ + 1)
∂x_phi_exact = zeros(Nx + 1, Nτ + 1)

for n = 1:Nτ + 1

    ∂x_phi_exact[:, n] = exactLozanoAslam.(x, (n-1) * τ)

    ∂x_phi[1, n] = (phi[2, n] - phi[1, n]) / h
    ∂x_phi_first_order[1, n] = (phi_first_order[2, n] - phi_first_order[1, n]) / h
    for i = 2:Nx + 1
        ∂x_phi[i, n] = (phi[i, n] - phi[i - 1, n]) / h
        ∂x_phi_first_order[i, n] = (phi_first_order[i, n] - phi_first_order[i - 1, n]) / h
    end
end

## Plot of the result 

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Second order scheme", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact[:,end], mode = "lines", name = "Exact solution", line=attr(color="royalblue", width=2))
trace3 = scatter(x = x, y = phi_first_order[:,end], mode = "lines", name = "First order scheme", line=attr(color="black", width=2))

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([trace1, trace2, trace3], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = ∂x_phi[:, end], mode = "lines", name = "Second order sol. gradient")
trace2_d = scatter(x = x, y = ∂x_phi_exact[:, end], mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = ∂x_phi_first_order[:, end], mode = "lines", name = "First order sol. gradient")


plot_phi_d = plot([trace1_d, trace2_d, trace3_d], layout)

p = [plot_phi; plot_phi_d]
relayout!(p, width = 1000, height = 500)
p