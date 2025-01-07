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

# Definition of Numerical Hamiltonian based on Godunov's Flux
function H_godunov(u, v, H)
    """
    Computes the numerical Hamiltonian using the Godunov flux.

    Arguments:
    - u: Gradient on the left of the cell interface
    - v: Gradient on the right of the cell interface
    - H: A function representing the Hamiltonian H(p)

    Returns:
    - The numerical Hamiltonian H*
    """
    if u <= v 
        return minimum(H(p) for p in [u , v])
    else
        return maximum(H(p) for p in [v , u])
    end
end

# Definition of Numerical Hamiltonian based on Lax-Friedrichs Flux
function H_lax_friedrichs(u, v, H, α = 100)
    """
    Computes the numerical Hamiltonian using the Lax-Friedrichs flux.

    Arguments:
    - u: Gradient on the left of the cell interface
    - v: Gradient on the right of the cell interface
    - H: A function representing the Hamiltonian H(p)

    Returns:
    - The numerical Hamiltonian H* splitted based on the arguments u and v
    """
    H_plus = u != 0 ? H(u) + α * u : 0
    H_minus = v != 0 ? H(v) - α * v : 0

    return 0.5 * (H_plus + H_minus)
end


## Definition of basic parameters

# Burger's equation 
H(x) = ( x.^2 ) / 2.0

# Mesh
xL = -1 # -2
xR = 1 # 2

level = 0 # Level of refinement
Nx = 100 * 2^level
h = (xR - xL) / Nx

x = range(xL - h, xR + h, length = Nx + 3)

# Initial condition
# F = integrate_function(burgersLozanoAslam, xL, xR)
# F = integrate_function(smoothBurgers, xL, xR)
F(x) = x.^2 / 2;
phi_0(x) = F(x)

# phi_0(x) = smoothBurgers(x)

# Time
T = 0.25 # 5 / 10
Nτ = 10 * 2^level
τ = T / Nτ

# Nτ = 2

t = range(0, (Nτ + 2) * τ, length = Nτ + 3)


## Comptutation
phi = zeros(Nx + 3, Nτ + 3);
phi_predictor_i = zeros(Nx + 3, Nτ + 3); # predictor in time n+1
phi_predictor_n = zeros(Nx + 3, Nτ + 3); # predictor in time n
phi_predictor_n2 = zeros(Nx + 3, Nτ + 3); # predictor in time n+2
phi_first_order = zeros(Nx + 3, Nτ + 3);

# Ghost point in time
phi[:, 1] = phi_0.(x);
phi_predictor_i[:, 1] = phi_0.(x);
phi_predictor_n[:, 1] = phi_0.(x);
phi_predictor_n2[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Compute the exact solution using the method of characteristics
phi_exact = zeros(Nx + 3, Nτ + 3)

for n = 1:Nτ + 3
    function exactBurgers(x)
        return exactLozanoAslam(x, (n-1) * τ)
        # return exactSmoothBurgersDerivative(x, (n-1) * τ)
    end
    # local phi_e = integrate_function(exactBurgers, xL - h, xR + h)
    local phi_e(x,t) = x.^2 / (2*(1+t.^2))
    # phi_exact[:, n] = phi_e(x)
    phi_exact[:, n] = phi_e.(x, (n-1) * τ)
    # if n < 3
    #     phi_exact[2:end-1, n] = phi_0.(x)
    #     phi_exact[1, n] = phi_exact[2, n]
    #     phi_exact[end, n] = phi_exact[end-1, n]
    # else
    #     phi_exact[:, n] = phi_e.(x_map)
    # end
end

# Initial condition
phi[:, 2] = phi_exact[:, 2];
phi_predictor_i[:, 2] = phi_exact[:, 2];
phi_predictor_n[:, 2] = phi_exact[:, 2];
phi_predictor_n2[:, 2] = phi_exact[:, 2];
phi_first_order[:, 2] = phi_exact[:, 2];

# Boundary conditions
phi[1:2, :] = phi_exact[1:2, :];
phi_predictor_i[1:2, :] = phi_exact[1:2, :];
phi_predictor_n[1:2, :] = phi_exact[1:2, :];
phi_first_order[1:2, :] = phi_exact[1:2, :];
phi_predictor_n2[1:2, :] = phi_exact[1:2, :];

phi[end-1:end, :] = phi_exact[end-1:end, :];
phi_predictor_i[end-1:end, :] = phi_exact[end-1:end, :];
phi_predictor_n[end-1:end, :] = phi_exact[end-1:end, :];
phi_first_order[end-1:end, :] = phi_exact[end-1:end, :];
phi_predictor_n2[end-1:end, :] = phi_exact[end-1:end, :];

# WENO parameters
ϵ = 1e-7;
ω0 = 1/3;
α0 = 1/3;

ω = zeros(Nx + 3)
α = zeros(Nx + 3)

# Lax Friedrichs parameter
LF_α = 4

# Time loop
for n = 2:Nτ + 1

    # First Sweep
    for i = 3:Nx + 2
        # First order scheme
        function firstOrderScheme(u)
            return u - phi_first_order[i, n] + τ * H_lax_friedrichs( (u - phi_first_order[i - 1, n + 1]) / h, 0, H, LF_α )
        end

        phi_first_order[i, n + 1] = newtonMethod(firstOrderScheme, x -> (firstOrderScheme(x + ϵ) - firstOrderScheme(x)) / ϵ, phi_first_order[i, n])

        # Predictor in time n+1 of the non-linear scheme
        # function firstOrderPredictor(u)
        #     return u - phi[i, n] + τ * H( (u - phi[i - 1, n + 1]) / h )
        # end

        # function firstOrderPredictor_n(u)
        #     return u - phi[i, n] + τ * H( (u - phi_predictor_n[i - 1, n + 1]) / h )
        # end

        # phi_predictor_i[i, n + 1] = newtonMethod(firstOrderPredictor, x -> (firstOrderPredictor(x + ϵ) - firstOrderPredictor(x)) / ϵ, phi[i, n])
        # phi_predictor_n[i, n + 1] = newtonMethod(firstOrderPredictor_n, x -> (firstOrderPredictor_n(x + ϵ) - firstOrderPredictor_n(x)) / ϵ, phi_predictor_n[i, n])

        # for p = 1 : 2

        #     # phi_predictor_n2[i - 1, n + 1] = phi_exact[i, n+2]
        #     # phi_predictor_n[:, :] = phi_exact[2:end-1, 2:end-1]

        #     # Correction
        #     # r_downwind_i = - phi[i, n] + phi_predictor_i[i - 1, n + 1] - phi_predictor_i[i, n + 1] + phi_right;
        #     # r_downwind_n = phi_predictor_n[i, n] - phi_predictor_n[i - 1, n + 1] - phi_predictor_n[i, n + 1] + phi_predictor_n2[i - 1, n + 1];

        #     # r_upwind_i = - phi[i - 1, n] + phi_left + phi[i, n] - phi[i - 1, n + 1];
        #     # r_upwind_n = phi[i - 1, n + 1] - phi[i, n] - phi[i - 1, n] + phi_old[i];

        #     # WENO SHU
        #     # U = ω0 * ( 1 / ( ϵ + r_upwind_i^2 )^2 );
        #     # D = ( 1 - ω0 ) * ( 1 / ( ϵ + r_downwind_i^2 )^2 );
        #     # ω[i] = U / ( U + D );

        #     # U = α0 * ( 1 / ( ϵ + r_upwind_n^2 )^2 );
        #     # D = ( 1 - α0 ) * ( 1 / ( ϵ + r_downwind_n^2 )^2 );
        #     # α[i] = U / ( U + D );

        #     # ENO
        #     # if (abs(r_downwind_i) + ϵ) < (abs(r_upwind_i) + ϵ)
        #     #     ω[i] = 0;
        #     # else
        #     #     ω[i] = 1;
        #     # end

        #     # if (abs(r_downwind_n) + ϵ) < (abs(r_upwind_n) + ϵ)
        #     #     α[i] = 0;
        #     # else
        #     #     α[i] = 1;
        #     # end

        #     # Final scheme 
        #     # function secondOrderScheme(u)

        #     #     first_order_term = (u - phi[i - 1, n + 1]) / h;
        #     #     second_order_term_i = ( ( 1 - ω[i] ) * r_downwind_i + ω[i] * r_upwind_i ) / (2 * h);
        #     #     second_order_term_n = ( ( 1 - α[i] ) * r_downwind_n + α[i] * r_upwind_n ) / (2 * τ);

        #     #     # Compute derivative of phi_predictor_i
        #     #     ∂x_phi = i > 1 ? (phi_predictor_i[i, n] - phi_predictor_i[i - 1, n]) / h : (phi_predictor_i[i + 1, n] - phi_predictor_i[i, n]) / h;

        #     #     C = ∂x_phi * τ / h;

        #     #     # if C > 1 
        #     #         # println("CFL condition not satisfied for x[i] = ", x[i], " and t = ", n * τ)
        #     #     # end

        #     #     A = min(1, C) / C;
        #     #     B = max(0, C - A) / C;

        #     #     A = C <= 1 ? 1 : 0;
        #     #     B = C < 1 ? 0 : 1;

        #     #     # println("second_order_term: ", second_order_term)
        #     #     return ( u - phi[i, n] ) / τ + B * second_order_term_n + H( first_order_term + A * second_order_term_i )
        #     # end

        #     # phi[i, n + 1] = newtonMethod(secondOrderScheme, x -> (secondOrderScheme(x + ϵ) - secondOrderScheme(x)) / ϵ, phi[i, n]);

        #     phi_predictor_i[i, n + 1] = phi[i, n + 1];
        #     phi_predictor_n[i, n + 1] = phi[i, n + 1];
        # end

        # function futureFirstOrderPredictor_n(u)
        #     return u - phi[i, n + 1] + τ * H( (u - phi_predictor_n2[i - 1, n + 1]) / h )
        # end

        # phi_predictor_n2[i, n + 1] = newtonMethod(futureFirstOrderPredictor_n, x -> (futureFirstOrderPredictor_n(x + ϵ) - futureFirstOrderPredictor_n(x)) / ϵ, phi[i, n + 1]);

    end

    # Extrapolation
    phi[Nx + 3, n + 1] = 3 * phi[Nx + 2, n + 1] - 3 * phi[Nx + 1, n + 1] + phi[Nx, n + 1];
    phi_first_order[Nx + 3, n + 1] = 3 * phi_first_order[Nx + 2, n + 1] - 3 * phi_first_order[Nx + 1, n + 1] + phi_first_order[Nx, n + 1];

    # Second sweep
    for i = Nx+2:-1:2
        # First order scheme
        function firstOrderScheme(u)
            return u - phi_first_order[i, n + 1] + τ * H_lax_friedrichs( 0, (phi_first_order[i + 1, n + 1] - u) / h, H , LF_α)
        end

        phi_first_order[i, n + 1] = newtonMethod(firstOrderScheme, x -> (firstOrderScheme(x + ϵ) - firstOrderScheme(x)) / ϵ, phi_first_order[i, n + 1])

        # Predictor in time n+1 of the non-linear scheme
        # function firstOrderPredictor(u)
        #     return u - phi[i, n] + τ * H( (u - phi[i - 1, n + 1]) / h )
        # end

        # function firstOrderPredictor_n(u)
        #     return u - phi[i, n] + τ * H( (u - phi_predictor_n[i - 1, n + 1]) / h )
        # end

        # phi_predictor_i[i, n + 1] = newtonMethod(firstOrderPredictor, x -> (firstOrderPredictor(x + ϵ) - firstOrderPredictor(x)) / ϵ, phi[i, n])
        # phi_predictor_n[i, n + 1] = newtonMethod(firstOrderPredictor_n, x -> (firstOrderPredictor_n(x + ϵ) - firstOrderPredictor_n(x)) / ϵ, phi_predictor_n[i, n])

        # for p = 1 : 2

        #     # phi_predictor_n2[i - 1, n + 1] = phi_exact[i, n+2]
        #     # phi_predictor_n[:, :] = phi_exact[2:end-1, 2:end-1]

        #     # Correction
        #     # r_downwind_i = - phi[i, n] + phi_predictor_i[i - 1, n + 1] - phi_predictor_i[i, n + 1] + phi_right;
        #     # r_downwind_n = phi_predictor_n[i, n] - phi_predictor_n[i - 1, n + 1] - phi_predictor_n[i, n + 1] + phi_predictor_n2[i - 1, n + 1];

        #     # r_upwind_i = - phi[i - 1, n] + phi_left + phi[i, n] - phi[i - 1, n + 1];
        #     # r_upwind_n = phi[i - 1, n + 1] - phi[i, n] - phi[i - 1, n] + phi_old[i];

        #     # WENO SHU
        #     # U = ω0 * ( 1 / ( ϵ + r_upwind_i^2 )^2 );
        #     # D = ( 1 - ω0 ) * ( 1 / ( ϵ + r_downwind_i^2 )^2 );
        #     # ω[i] = U / ( U + D );

        #     # U = α0 * ( 1 / ( ϵ + r_upwind_n^2 )^2 );
        #     # D = ( 1 - α0 ) * ( 1 / ( ϵ + r_downwind_n^2 )^2 );
        #     # α[i] = U / ( U + D );

        #     # ENO
        #     # if (abs(r_downwind_i) + ϵ) < (abs(r_upwind_i) + ϵ)
        #     #     ω[i] = 0;
        #     # else
        #     #     ω[i] = 1;
        #     # end

        #     # if (abs(r_downwind_n) + ϵ) < (abs(r_upwind_n) + ϵ)
        #     #     α[i] = 0;
        #     # else
        #     #     α[i] = 1;
        #     # end

        #     # Final scheme 
        #     # function secondOrderScheme(u)

        #     #     first_order_term = (u - phi[i - 1, n + 1]) / h;
        #     #     second_order_term_i = ( ( 1 - ω[i] ) * r_downwind_i + ω[i] * r_upwind_i ) / (2 * h);
        #     #     second_order_term_n = ( ( 1 - α[i] ) * r_downwind_n + α[i] * r_upwind_n ) / (2 * τ);

        #     #     # Compute derivative of phi_predictor_i
        #     #     ∂x_phi = i > 1 ? (phi_predictor_i[i, n] - phi_predictor_i[i - 1, n]) / h : (phi_predictor_i[i + 1, n] - phi_predictor_i[i, n]) / h;

        #     #     C = ∂x_phi * τ / h;

        #     #     # if C > 1 
        #     #         # println("CFL condition not satisfied for x[i] = ", x[i], " and t = ", n * τ)
        #     #     # end

        #     #     A = min(1, C) / C;
        #     #     B = max(0, C - A) / C;

        #     #     A = C <= 1 ? 1 : 0;
        #     #     B = C < 1 ? 0 : 1;

        #     #     # println("second_order_term: ", second_order_term)
        #     #     return ( u - phi[i, n] ) / τ + B * second_order_term_n + H( first_order_term + A * second_order_term_i )
        #     # end

        #     # phi[i, n + 1] = newtonMethod(secondOrderScheme, x -> (secondOrderScheme(x + ϵ) - secondOrderScheme(x)) / ϵ, phi[i, n]);

        #     phi_predictor_i[i, n + 1] = phi[i, n + 1];
        #     phi_predictor_n[i, n + 1] = phi[i, n + 1];
        # end

        # function futureFirstOrderPredictor_n(u)
        #     return u - phi[i, n + 1] + τ * H( (u - phi_predictor_n2[i - 1, n + 1]) / h )
        # end

        # phi_predictor_n2[i, n + 1] = newtonMethod(futureFirstOrderPredictor_n, x -> (futureFirstOrderPredictor_n(x + ϵ) - futureFirstOrderPredictor_n(x)) / ϵ, phi[i, n + 1]);

    end

    # Extrapolation
    phi[1, n + 1] = 3 * phi[2, n + 1] - 3 * phi[3, n + 1] + phi[4, n + 1];
    phi_first_order[1, n + 1] = 3 * phi_first_order[2, n + 1] - 3 * phi_first_order[3, n + 1] + phi_first_order[4, n + 1];
end

## Compute and print the error
Error_t_h = τ * h * sum(abs(phi[i, n] - phi_exact[i, n]) for n in 2:Nτ+2 for i in 2:Nx+2)
println("Error t*h: ", Error_t_h)
Error_t_h_1 = τ * h * sum(abs(phi_first_order[i, n] -  phi_exact[i, n]) for n in 2:Nτ+2 for i in 2:Nx+2)
println("Error t*h first order: ", Error_t_h_1)


## Compute the numerical derivative of the solution
∂x_phi_first_order = zeros(Nx + 3, Nτ + 3)
∂x_phi = zeros(Nx + 3, Nτ + 3)
∂x_phi_exact = zeros(Nx + 3, Nτ + 3)

for n = 1:Nτ + 2

    # ∂x_phi_exact[:, n] = exactLozanoAslam.(x, (n-1) * τ)
    ∂x_phi_exact[1, n] = (phi_exact[2, n + 1] - phi_exact[1, n + 1]) / h;
    ∂x_phi[1, n] = (phi[2, n] - phi[1, n]) / h
    ∂x_phi_first_order[1, n] = (phi_first_order[2, n] - phi_first_order[1, n]) / h
    for i = 2:Nx + 3
        ∂x_phi_exact[i, n] = (phi_exact[i, n + 1] - phi_exact[i - 1, n + 1]) / h
        ∂x_phi[i, n] = (phi[i, n] - phi[i - 1, n]) / h
        ∂x_phi_first_order[i, n] = (phi_first_order[i, n] - phi_first_order[i - 1, n]) / h
    end
end

## Plot of the result 

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,1], mode = "lines", name = "Second order scheme", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact[:,end - 1], mode = "lines", name = "Exact solution", line=attr(color="royalblue", width=2))
trace3 = scatter(x = x, y = phi_first_order[:, end - 1], mode = "lines", name = "First order scheme", line=attr(color="black", width=2))

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([trace1, trace2, trace3], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = ∂x_phi[:, 1], mode = "lines", name = "Second order sol. gradient")
trace2_d = scatter(x = x, y = ∂x_phi_exact[:, end-1], mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = ∂x_phi_first_order[:, end-1], mode = "lines", name = "First order sol. gradient")


plot_phi_d = plot([trace1_d, trace2_d, trace3_d], layout)

# plot of α
trace1_α = scatter(x = x, y = α, mode = "lines", name = "α")

plot_α = plot([trace1_α], layout)


p = [plot_phi; plot_phi_d]
relayout!(p, width = 1000, height = 500)
p