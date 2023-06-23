# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS

include("InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

# Courant number
c = 0.5

# Grid settings
xL = - pi / 2
xR = pi
Nx = 80 * 2^level
h = (xR - xL) / Nx

# Velocity
u = 2.0

# Time
tau = c * h / u
Ntau = Int(Nx / 10)

# Initial condition
phi_0(x) = cos(x);

# Exact solution
phi_exact(x, t) = phi_0(x - u * t);

## Comptutation

# Grid initialization
x = range(xL, xR, length = Nx + 1)
phi = zeros(Nx + 1, Ntau + 1);
phi_predictor_i = zeros(Nx + 1, Ntau + 1); # predictor in time n+1
phi_predictor_n = zeros(Nx + 1, Ntau + 1); # predictor in time n
phi_predictor_n2 = zeros(Nx + 1, Ntau + 1); # predictor in time n+2
phi_first_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor_i[:, 1] = phi_0.(x);
phi_predictor_n[:, 1] = phi_0.(x);
phi_predictor_n2[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_i[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n2[1, :] = phi_exact.(x[1], range(tau, (Ntau + 1) * tau, length = Ntau + 1));

# Ghost point on the right side 
ghost_point_right = phi_exact.(xR + h, range(0, Ntau * tau, length = Ntau + 1));
# Ghost point on the left side 
ghost_point_left = phi_exact.(xL - h, range(tau, (Ntau+1) * tau, length = Ntau + 1));
# Ghost point on the time -1
ghost_point_time = phi_exact.(x, -tau);

# ENO parameters
w = 1 / 2; # normal logic
s = 1 / 2; # inverted logic

# Time Loop
for n = 1:Ntau

    if n > 1
        phi_old = phi[:, n-1];
    else
        phi_old = ghost_point_time;
    end

    # Space loop
    for i = 2:Nx + 1

        if i < Nx + 1
            phi_right_n = phi[i + 1, n];
        else
            phi_right_n = ghost_point_right[n];
        end

        if i > 2
            phi_left_n_plus = phi_predictor_i[i - 2, n + 1];
        else
            phi_left_n_plus = ghost_point_left[n];
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c * phi_first_order[i - 1, n + 1] ) / ( 1 + c );
        
        # Predictor
        phi_predictor_i[i, n + 1] = ( phi[i, n] + c * phi[i - 1, n + 1] ) / ( 1 + c );
        phi_predictor_n[i, n + 1] = ( phi[i, n] + c * phi_predictor_n[i - 1, n + 1] ) / ( 1 + c );

        # Corrector
        r_downwind_i = - phi_predictor_i[i, n + 1] + phi_right_n - phi[i, n] + phi[i - 1, n + 1];
        r_upwind_i = phi[i, n] - phi[i - 1, n] - phi[i - 1, n + 1] + phi_left_n_plus;

        r_downwind_n = - phi_predictor_n[i, n + 1] + phi_predictor_n2[i - 1, n + 1] + phi[i, n] - phi[i - 1, n + 1];
        r_upwind_n = phi[i - 1, n + 1] - phi[i - 1, n] - phi[i, n] + phi_old[i];

        # Second order solution
        phi[i, n + 1] = ( phi[i, n] + c * phi[i - 1, n + 1] - 1 * c * min( 1, max( 0, 3/2 - c ) ) * ( 0.5 * ( 1 - w ) * r_downwind_i + 0.5 * w * r_upwind_i ) 
                                                            - 1 * max( 0, min( 1, c - 1/2 ) ) * ( 0.5 * (1 - s) * r_downwind_n + 0.5 * s * r_upwind_n  ) ) / ( 1 + c );

        # Predictor for next time step
        phi_predictor_n2[i, n + 1] = ( phi[i, n + 1] + c * phi_predictor_n2[i - 1, n + 1] ) / ( 1 + c );

    end
end

# Print the error
println("Error L2: ", sum(abs.(phi[:,end] - phi_exact.(x, Ntau * tau))) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)

# Print first order error
println("Error L2 first order: ", sum(abs.(phi_first_order[:,end] - phi_exact.(x, Ntau * tau))) * h)
println("Error L_inf firts order: ", norm(phi_first_order[:, end] - phi_exact.(x, Ntau * tau), Inf)* h)

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Compact scheme solution")
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact solution")
trace3 = scatter(x = x, y = phi_first_order[:, end], mode = "lines", name = "First order solution")

layout = Layout(title = "Linear advection equation", xaxis_title = "x", yaxis_title = "phi")

plot_phi = plot([trace1, trace2, trace3], layout)

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[:, end]) / h, mode = "lines", name = "Compact sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x, Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = diff(phi_first_order[:, end]) / h, mode = "lines", name = "First order sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace1_d, trace2_d, trace3_d], layout_d)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p
