# Normal linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS

include("InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 5;

# Courant number
c = 15

# Grid settings
xL = - 1
xR = 1
Nx = 80 * 2^level
h = (xR - xL) / Nx

# Velocity
u = 1.0

# Time
tau = c * h / u
Ntau = Int(Nx / 10)
# Ntau = 1

# Initial condition
# phi_0(x) = piecewiseLinear(x);
# phi_0(x) = piecewiseConstant(x);
# phi_0(x) = makePeriodic(nonSmooth,-1,1)(x - 0.5);
# phi_0(x) = cos(x);
phi_0(x) = makePeriodic(allInOne,-1,1)(x);

# Exact solution
phi_exact(x, t) = phi_0(x - u * t);

## Comptutation

# Grid initialization
x = range(xL, xR, length = Nx + 1)
phi = zeros(Nx + 1, Ntau + 1);
phi_predictor = zeros(Nx + 1, Ntau + 1);
phi_first_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));

# Ghost point on the right side (u > 0)
ghost_point_right = phi_exact.(xR + h, range(0, Ntau * tau, length = Ntau + 1));
# Ghost point on the left side 
ghost_point_left = phi_exact.(xL - h, range(tau, (Ntau+1) * tau, length = Ntau + 1));

# ENO parameters
s = zeros(Nx + 1, Ntau + 1);
eps = 1e-8;

# Time loop
for n = 1:Ntau
    # Space loop
    for  i = 2:Nx+1

        if i < Nx + 1
            phi_right = phi[i + 1, n];
        else
            phi_right = ghost_point_right[n];
        end

        if i > 2
            phi_left_n_plus = phi[i - 2, n + 1];
        else
            phi_left_n_plus = ghost_point_left[n];
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c * phi_first_order[i - 1, n + 1] ) / ( 1 + c );
        
        # Predictor
        phi_predictor[i, n + 1] = ( phi_predictor[i, n] + c * phi[i - 1, n + 1] ) / ( 1 + c );

        # Corrector
        r_downwind_i_minus = - phi[i, n] + phi_predictor[i - 1, n + 1];
        r_upwind_i_minus = - phi[i - 1, n] + phi_left_n_plus;

        r_downwind_i = - phi_predictor[i, n + 1] + phi_right;
        r_upwind_i = phi[i, n] - phi[i - 1, n + 1];

        # ENO parameter
        # if i == 2
        #     abs(r_downwind_i_minus) < eps ? s[i - 1, n + 1] = 0 : s[i - 1, n + 1] = max(0, min(1, r_upwind_i_minus / r_downwind_i_minus));
        # end

        # abs(r_downwind_i) <  eps ? s[i, n + 1] = 0 : s[i, n + 1] = max(0, min(1, r_upwind_i / r_downwind_i)); 

        if abs(r_downwind_i_minus + r_downwind_i) <= abs(r_upwind_i_minus + r_upwind_i)
            s[i,n+1] = 1
        else
            s[i,n+1] = 0
        end

        # Second order solution
        # phi[i, n + 1] = ( phi[i, n] + c * ( phi[i - 1, n + 1] - 0.5 * s[i - 1, n + 1] * r_downwind_i_minus - 0.5 * s[i, n + 1] * r_downwind_i ) ) / ( 1 + c );


        phi[i, n + 1] = ( phi[i, n] + c * ( phi[i - 1, n + 1] - 0.5 * s[i, n + 1] * (r_downwind_i_minus + r_downwind_i) 
        - 0.5 * (1-s[i, n + 1]) * (r_upwind_i + r_upwind_i_minus ) ) ) / ( 1 + c );


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
