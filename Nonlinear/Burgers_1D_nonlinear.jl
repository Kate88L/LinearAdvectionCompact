# Compact finite difference scheme for the 1D Burgers' equation

using NLsolve
using LinearAlgebra
using PlotlyJS
using QuadGK
using Trapz
using Interpolations

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")
include("../Utils/Solvers.jl")


## Definition of basic parameters

# Burger's equation 
H(x) = ( x.^2 ) / 2.0

# Mesh
xL = 0
xR = 1

level = 3 # Level of refinement
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

# Nτ = 6

## Comptutation
phi = zeros(Nx + 1, Nτ + 1);
phi_predictor_i = zeros(Nx + 1, Nτ + 1); # predictor in time n+1
phi_predictor_n = zeros(Nx + 1, Nτ + 1); # predictor in time n
phi_predictor_n2 = zeros(Nx + 1, Nτ + 1); # predictor in time n+2
phi_first_order = zeros(Nx + 1, Nτ + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor_i[:, 1] = phi_0.(x);
phi_predictor_n[:, 1] = phi_0.(x);
phi_predictor_n2[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Compute the exact solution using the method of characteristics
phi_exact = zeros(Nx + 3, Nτ + 3)
x_map = range(xL - h, xR + h, length = Nx + 3)
for n = 1:Nτ + 3
    function exactLozanoAslam_t(x)
        return exactLozanoAslam(x, (n-2) * τ)
    end
    local phi_e = integrate_function(exactLozanoAslam_t, xL - h, xR + h)
    phi_exact[:, n] = phi_e(x_map)
end

phi[1, :] = phi_exact[2, 2:end-1];
phi_predictor_i[1, :] = phi_exact[2, 2:end-1];
phi_predictor_n[1, :] = phi_exact[2, 2:end-1];
phi_first_order[1, :] = phi_exact[2, 2:end-1];
phi_predictor_n2[1, :] = phi_exact[2, 3:end];

# phi[end, :] = phi_exact[end - 1, 2:end-1];
# phi_predictor_i[end, :] = phi_exact[end - 1, 2:end-1];
# phi_predictor_n[end, :] = phi_exact[end - 1, 2:end-1];
# phi_first_order[end, :] = phi_exact[end - 1, 2:end-1];
# phi_predictor_n2[end, :] = phi_exact[end - 1, 3:end];

# Ghost point on the right side 
ghost_point_right = phi_exact[end, 2:end-2]
# Ghost point on the left side 
ghost_point_left = phi_exact[1, 3:end-1]
# Ghost point on the time -1
ghost_point_time = phi_exact[2:end-1, 1]

# WENO parameters
ϵ = 1e-16;
ω0 = 1/3;
α0 = 1/3;

ω = zeros(Nx + 1)
α = zeros(Nx + 1)

l = zeros(Nx + 1)
s = zeros(Nx + 1)

# Time loop
for n = 1:Nτ

    # Boundary conditions
    if n > 1
        phi_old = phi[:, n-1];
    else
        phi_old = ghost_point_time;
    end

    for i = 2:Nx + 1
        # First order scheme
        function firstOrderScheme(u)
            return u - phi_first_order[i, n] + τ * H( (u - phi_first_order[i - 1, n + 1]) / h )
        end

        phi_first_order[i, n + 1] = newtonMethod(firstOrderScheme, x -> (firstOrderScheme(x + ϵ) - firstOrderScheme(x)) / ϵ, phi_first_order[i, n])

        # Predictor in time n+1 of the non-linear scheme
        function firstOrderPredictor(u)
            return u - phi[i, n] + τ * H( (u - phi[i - 1, n + 1]) / h )
        end

        phi_predictor_i[i, n + 1] = newtonMethod(firstOrderPredictor, x -> (firstOrderPredictor(x + ϵ) - firstOrderPredictor(x)) / ϵ, phi[i, n])

        if i < Nx + 1
            phi_right = phi[i + 1, n]
        else
            phi_right = ghost_point_right[n]
        end

        if i > 2
            phi_left = phi[i - 2, n + 1];
        else
            phi_left = ghost_point_left[n];
        end

        # Correction
        r_downwind_i = - phi[i, n] + phi_predictor_i[i - 1, n + 1] - phi_predictor_i[i, n + 1] + phi_right;
        # r_downwind_n = phi_predictor_n[i, n] - phi_predictor_n[i - 1, n + 1] - phi_predictor_n[i, n + 1] + phi_predictor_n2[i - 1, n + 1];

        r_upwind_i = - phi[i - 1, n] + phi_left + phi[i, n] - phi[i - 1, n + 1];
        # r_upwind_n = phi[i - 1, n + 1] - phi[i, n] - phi[i - 1, n] + phi_old[i];

        # WENO SHU
        # U = ω0 * ( 1 / ( ϵ + r_upwind_i^2 )^2 );
        # D = ( 1 - ω0 ) * ( 1 / ( ϵ + r_downwind_i^2 )^2 );
        # ω[i] = U / ( U + D );

        # U = α0 * ( 1 / ( ϵ + r_upwind_n^2 )^2 );
        # D = ( 1 - α0 ) * ( 1 / ( ϵ + r_downwind_n^2 )^2 );
        # α[i] = U / ( U + D );

        # ENO
        if (abs(r_downwind_i) + ϵ) < (abs(r_upwind_i) + ϵ)
            ω[i] = 0;
        else
            ω[i] = 1;
        end

        # if (abs(r_downwind_n) + ϵ) < (abs(r_upwind_n) + ϵ)
        #     α[i] = 0;
        # else
        #     α[i] = 1;
        # end

        # Final scheme 
        function secondOrderScheme(u)
            first_order_term = (u - phi[i - 1, n + 1]) / h;
            second_order_term = ( ( 1 - ω[i] ) * r_downwind_i + ω[i] * r_upwind_i ) / (2 * h);
            return u - phi[i, n] + τ * H( first_order_term + second_order_term )
        end

        phi[i, n + 1] = newtonMethod(secondOrderScheme, x -> (secondOrderScheme(x + ϵ) - secondOrderScheme(x)) / ϵ, phi[i, n]);

    end

end

## Compute and print the error
println("Error L2 first order scheme: ", norm(phi_first_order[:,end] - phi_exact[2:end-1, end-1], 2) * h)
println("Error L2 final scheme: ", norm(phi[:,end] - phi_exact[2:end-1, end-1], 2) * h)
Error_t_h = τ * h * sum(abs(phi[i, n] - phi_exact[i+1, n+1]) for n in 1:Nτ+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)


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