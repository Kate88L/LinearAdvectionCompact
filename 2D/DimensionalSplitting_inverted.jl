# Normal linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS

include("Utils\\InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 1;

# Courant number
C = 3;

# Grid settings
x1L = -1
x1R = 1
x2L = -1
x2R = 1
Nx = 100 * 2^level
h = (x1R - x1L) / Nx

# Velocity
U = [1.0, 1.0]

# Time settings
tau = C * h / maximum(abs.(U))
Ntau = Int(Nx / 10)
# Ntau = 1;

c = zeros(Nx + 1, 1) .+ U[1] * tau / h;
d = zeros(Nx + 1, 1) .+ U[2] * tau / h;

# Initial condition
# phi_0(x1, x2) = cos.(x1) .* cos.(x2);
phi_0(x1, x2) = x1.^2 + x2.^2;

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
phi_predictor_n2 = zeros(Nx + 1, Nx + 1, Ntau + 1); # predictor in time n+2

# Initial condition
phi[:, :, 1] = phi_0.(X1, X2);
phi_first_order[:, :, 1] = phi_0.(X1, X2);
phi_predictor[:, :, 1] = phi_0.(X1, X2);
phi_predictor_n2[:, :, 1] = phi_0.(X1, X2);

# ENO parameters
px = zeros(Nx + 1, Ntau + 1); 
py = zeros(Nx + 1,  Ntau + 1); 
eps = 1e-8;

# Time loop
for n = 1:Ntau

    # Boundary conditions
    phi_first_order[1, :, n + 1] = phi_exact_x.(x1[1], x2, n * tau, (n-1)* tau);
    phi[1, :, n + 1] = phi_exact_x.(x1[1], x2, n * tau, (n-1)* tau);
    phi_predictor[1, :, n + 1] = phi_exact_x.(x1[1], x2, n * tau, (n-1)* tau);
    phi_predictor_n2[1, :, n + 1] = phi_exact_x.(x1[1], x2, (n+1) * tau, n * tau);
    # phi_predictor_n2[:, :, n + 1] = phi_exact.(X1, X2, (n+1) * tau);

    if n > 1
        phi_old = phi[:, :, n - 1];
    else
        phi_old = phi_exact.(X1, X2, -tau);
    end

    # Space loop - X direction
    for  i = 2:1:Nx + 1

        # First order solution
        phi_first_order[i, :, n + 1] = ( phi_first_order[i, :, n] + c[i] * phi_first_order[i - 1, :, n + 1]  ) / ( 1 + c[i] );
        
        # Predictor
        phi_predictor[i, :, n + 1] = ( phi[i, :, n] + c[i] * phi_predictor[i - 1, :, n + 1] ) / ( 1 + c[i] );

        # Corrector
        r_downwind_n_minus = phi_predictor[i, :, n] - phi_predictor[i - 1, :, n + 1];
        r_upwind_n_minus = - phi[i - 1, :, n] + phi_old[i, :];

        r_downwind_n = - phi_predictor[i, :, n + 1] + phi_predictor_n2[i - 1, :, n + 1];
        r_upwind_n = - phi[i, :, n] + phi[i - 1, :, n + 1];

        r_upwind_n = r_upwind_n_minus + r_upwind_n;
        r_downwind_n = r_downwind_n_minus + r_downwind_n;

        # ENO parameter
        abs.(r_downwind_n) <= abs.(r_upwind_n) ? px[i, n + 1] = 0 : px[i, n + 1] = 0;

        # Second order solution
        phi[i, :, n + 1] = ( phi[i, :, n] + 0.5/c[i] * (c[i] - c[i-1]) * phi[i, :, n] + c[i] * phi[i - 1, :, n + 1] 
                                          - 0.5 * ( px[i, n + 1] .* r_upwind_n + ( 1 - px[i, n + 1] ) .* r_downwind_n  ) )  / ( 1 + c[i] + 0.5 / c[i] * (c[i] - c[i-1]) );

        # Predictor for next time step
        phi_predictor_n2[i, :, n + 1] = ( phi[i, :, n + 1] + c[i] * phi_predictor_n2[i - 1, :, n + 1] ) / ( 1 + c[i] );


    end

    phi_old = phi[:, :, n];
    phi[:, :, n] = phi[:, :, n + 1];
    phi_first_order[:, :, n] = phi_first_order[:, :, n + 1];
    phi_predictor[:, :, n] = phi_predictor[:, :, n + 1];

    phi_first_order[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);
    phi[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);
    phi_predictor[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);
    phi_predictor_n2[:, 1, n + 1] = phi_exact.(x1, x2[1], (n+1) * tau);

    # Space loop - Y direction
    for j = 2:1:Nx + 1

        # First order solution
        phi_first_order[:, j, n + 1] = ( phi_first_order[:, j, n] + d[j] * phi_first_order[:, j - 1, n + 1]  ) / ( 1 + d[j] );
        
        # Predictor
        phi_predictor[:, j, n + 1] = ( phi[:, j, n] + d[j] * phi_predictor[:, j - 1, n + 1] ) / ( 1 + d[j] );

        # Corrector
        r_downwind_n_minus = phi_predictor[:, j, n] - phi_predictor[:, j - 1, n + 1];
        r_upwind_n_minus = - phi[:, j - 1, n] + phi_old[:, j];

        r_downwind_n = - phi_predictor[:, j, n + 1] + phi_predictor_n2[:, j - 1, n + 1];
        r_upwind_n = - phi[:, j, n] + phi[:, j - 1, n + 1];

        r_upwind_n = r_upwind_n_minus + r_upwind_n;
        r_downwind_n = r_downwind_n_minus + r_downwind_n;

        # ENO parameter
        abs.(r_downwind_n) <= abs.(r_upwind_n) ? py[j, n + 1] = 0 : py[j, n + 1] = 0;

        # Second order solution
        phi[:, j, n + 1] = ( phi[:, j, n] + 0.5/d[j] * (d[j] - d[j-1]) * phi[:, j, n] + d[j] * phi[:, j - 1, n + 1] 
                                          - 0.5 * py[j, n + 1] .* r_upwind_n
                                          - 0.5 * ( 1 - py[j, n + 1] ) .* r_downwind_n )  / ( 1 + d[j] + 0.5 / d[j] * (d[j] - d[j-1]) );

        # Predictor for next time step
        phi_predictor_n2[:, j, n + 1] = ( phi[:, j, n + 1] + d[j] * phi_predictor_n2[:, j - 1, n + 1] ) / ( 1 + d[j] );


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

plot_phi = plot([trace1, trace2], layout)

# plot_error = plot(surface(x = x1, y = x2, z = abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))* h^2))

