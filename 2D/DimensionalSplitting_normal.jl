# Normal linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS

include("../Utils/InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

# Courant number
C = 5;

# Grid settings
x1L = -1
x1R = 2
x2L = -1
x2R = 2
Nx = 100 * 2^level
h = (x1R - x1L) / Nx

# Velocity
U = [1.0, 0.0]

# Time settings
tau = C * h / maximum(abs.(U))
Ntau = Int(Nx / 10)
Ntau = 1;

c = zeros(Nx + 1, 1) .+ U[1] * tau / h;
d = zeros(Nx + 1, 1) .+ U[2] * tau / h;

# Initial condition
# phi_0(x1, x2) = cos.(x1) .* cos.(x2);
# phi_0(x1, x2) = x1.^2 + x2.^2;
phi_0(x1, x2) = exp.(-10 * (x1.^2 + x2.^2));

# Exact solution
phi_exact(x1, x2, t) = phi_0.(x1 - U[1] * t, x2 - U[2] * t);
phi_exact_x(x1, x2, t1, t2) = phi_0.(x1 - U[1] * t1, x2 - U[2] * t2);

## Comptutation

# Grid initialization
x1 = range(x1L, x1R, length = Nx + 1)
x2 = range(x2L, x2R, length = Nx + 1)
t = range(0, Ntau * tau, length = Ntau + 1)

X1 = repeat(x1, 1, Nx+1)
X2 = repeat(x2', Nx+1, 1)
T = repeat(t, 1, Nx+1)

phi = zeros(Nx + 1, Nx + 1, Ntau + 1);
phi_first_order = zeros(Nx + 1, Nx + 1, Ntau + 1);
phi_predictor = zeros(Nx + 1, Nx + 1, Ntau + 1); # predictor in time n+1

# Initial condition
phi[:, :, 1] = phi_0.(X1, X2);
phi_first_order[:, :, 1] = phi_0.(X1, X2);
phi_predictor[:, :, 1] = phi_0.(X1, X2);

# Ghost point on the time -1
ghost_point_time = phi_exact.(X1, X2, -tau);

# ENO parameters
sx = zeros(Nx + 1, Ntau + 1); 
sy = zeros(Nx + 1,  Ntau + 1); 
eps = 1e-8;

# Time loop
for n = 1:Ntau

    # Boundary conditions
    phi_first_order[1, :, n + 1] = phi_exact_x.(x1[1], x2, n * tau, (n-1)* tau);
    phi[1, :, n + 1] = phi_exact_x.(x1[1], x2, n * tau, (n-1)* tau);
    phi_predictor[1, :, n + 1] = phi_exact_x.(x1[1], x2, n * tau, (n-1)* tau);
    ghost_point_left = phi_exact_x.(x1[1]-h, x2, n * tau, (n-1)* tau);
    ghost_point_right = phi_exact.(x1[end] + h, x2, (n-1) * tau);

    # Space loop - X direction
    for  i = 2:1:Nx + 1
 
        if i > 2
            phi_left = phi[i - 2, :, n + 1];
        else
            phi_left = ghost_point_left;
        end

        if i < Nx + 1
            phi_right = phi_predictor[i + 1, :, n];
        else
            phi_right = ghost_point_right;
        end

        # First order solution
        phi_first_order[i, :, n + 1] = ( phi_first_order[i, :, n] + c[i] * phi_first_order[i - 1, :, n + 1]  ) / ( 1 + c[i] );
        
        # Predictor
        phi_predictor[i, :, n + 1] = ( phi_predictor[i, :, n] + c[i] * phi[i - 1, :, n + 1] ) / ( 1 + c[i] );

        # Corrector
        r_downwind_i_minus = - phi_predictor[i, :, n] + phi_predictor[i - 1, :, n + 1];
        r_upwind_i_minus = -phi[i - 1, :, n] + phi_left;

        r_downwind_i = - phi_predictor[i, :, n + 1] + phi_right;
        r_upwind_i = phi[i, :, n] - phi[i - 1, :, n + 1];

        r_upwind_i = r_upwind_i_minus + r_upwind_i;
        r_downwind_i = r_downwind_i_minus + r_downwind_i;

        # ENO parameter
        abs.(r_downwind_i) <= abs.(r_upwind_i) ? sx[i, n + 1] = 1 : sx[i, n + 1] = 0;

        # Second order solution
        phi[i, :, n + 1] = ( phi[i, :, n] + c[i] * ( phi[i - 1, :, n + 1] - 0.5 * sx[i, n + 1] .* r_downwind_i - 0.5 * ( 1 - sx[i, n + 1] ) .* r_upwind_i ) ) / ( 1 + c[i] );

    end

    continue;

    phi[:, :, n] = phi[:, :, n + 1];
    phi_first_order[:, :, n] = phi_first_order[:, :, n + 1];
    phi_predictor[:, :, n] = phi_predictor[:, :, n + 1];

    phi_first_order[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);
    phi[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);
    phi_predictor[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);

    ghost_point_down = phi_exact.(x1, x2[1]-h, n * tau);
    ghost_point_up = phi_exact.(x1, x2[end]+h, (n-1) * tau);

    # Space loop - Y direction
    for j = 2:1:Nx + 1

        if j > 2
            phi_down = phi[:, j - 2, n + 1];
        else
            phi_down = ghost_point_down;
        end

        if j < Nx + 1
            phi_up = phi_predictor[:, j + 1, n];
        else
            phi_up = ghost_point_up;
        end

        # First order solution
        phi_first_order[:, j, n + 1] = ( phi_first_order[:, j, n] + d[j] * phi_first_order[:, j - 1, n + 1] ) / ( 1 + d[j] );
        
        # Predictor
        phi_predictor[:, j, n + 1] = ( phi_predictor[:, j, n] + d[j] * phi[:, j - 1, n + 1] ) / ( 1 + d[j] );

        # Corrector
        r_downwind_j_minus = - phi_predictor[:, j, n] + phi_predictor[:, j - 1, n + 1];
        r_upwind_j_minus = - phi[:, j - 1, n] + phi_down;

        r_downwind_j = - phi_predictor[:, j, n + 1] + phi_up;
        r_upwind_j = phi[:, j, n] - phi[:, j - 1, n + 1];

        r_upwind_j = r_upwind_j_minus + r_upwind_j;
        r_downwind_j = r_downwind_j_minus + r_downwind_j; 

        # ENO parameter 
        abs.(r_downwind_j) <= abs.(r_upwind_j) ? sy[j, n + 1] = 1 : sy[j, n + 1] = 0;

        # Second order solution
        phi[:, j, n + 1] = ( phi[:, j, n] + d[j] * ( phi[:, j - 1, n + 1] - 0.5 * sy[j, n + 1] .* r_downwind_j - 0.5 * ( 1 - sy[j, n + 1] ) .* r_upwind_j ) ) / ( 1 + d[j] );

    end
end

# Print the error
println("Error L2: ", sum(abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))) * h^2)
println("Error L_inf: ", norm(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau), Inf)* h^2)

# Print first order error
println("Error L2 first order: ", sum(abs.(phi_first_order[:, :, end] - phi_exact.(X1, X2, Ntau * tau))) * h^2)
println("Error L_inf firts order: ", norm(phi_first_order[:, :, end] - phi_exact.(X1, X2, Ntau * tau), Inf)* h^2)

# Plot of the result at the final time together with the exact solution
trace1 = contour(x = x1, y = x2, z = phi_exact.(X1, X2, Ntau * tau), name = "Exact solution", showscale=false, contours_coloring="lines", colorscale="Greys", line_width=2)
trace2 = contour(x = x1, y = x2, z = phi[:, :, end], name = "Compact", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1)
trace3 = contour(x = x1, y = x2, z = phi_first_order[:, :, end], name = "First order", showscale=false, colorscale = "Viridis", contours_coloring="lines", line_width=1)
layout = Layout(title = "Linear advection equation", xaxis_title = "x2", yaxis_title = "x1", zaxis_title = "phi", colorbar = false)

plot_phi = plot([trace1, trace3], layout)

# plot_error = plot(surface(x = x1, y = x2, z = abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))* h^2))

