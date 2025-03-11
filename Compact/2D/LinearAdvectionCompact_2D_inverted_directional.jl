# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS

include("../../Utils/InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

# Courant number
c = 5;
d = c;

# Grid settings - 2D regular grid
x1L = -pi/2
x1R = pi
x2L = -pi/2
x2R = pi
Nx = 100 * 2^level
h = (x1R - x1L) / Nx

# Velocity
angle = 45 # angle in degrees
U = [cosd(angle), sind(angle)]

# Time
tau = cosd(angle) * c * h / maximum(abs.(U))
Ntau = Int(Nx / 10)

# Initial condition
phi_0(x1, x2) = x1.^2 + x2.^2;

# Exact solution
phi_exact(x1, x2, t) = phi_0.(x1 - U[1] * t, x2 - U[2] * t);

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

# Ghost point on the time -1
ghost_point_time = phi_exact.(X1, X2, -tau);

# ENO parameters
p = zeros(Nx + 1, Nx + 1, Ntau + 1); 

eps = 1e-8;

# Time Loop
for n = 1:Ntau

    # Boundary conditions
    phi_first_order[1, :, n + 1] = phi_exact.(x1[1], x2, n * tau);
    phi_first_order[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);

    phi[1, :, n + 1] = phi_exact.(x1[1], x2, n * tau);
    phi[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);

    phi_predictor[1, :, n + 1] = phi_exact.(x1[1], x2, n * tau);
    phi_predictor[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);

    phi_predictor_n2[1, :, n + 1] = phi_exact.(x1[1], x2, (n + 1) * tau);
    phi_predictor_n2[:, 1, n + 1] = phi_exact.(x1, x2[1], (n + 1) * tau);

    if n > 1
        phi_old = phi[:, :, n - 1];
    else
        phi_old = ghost_point_time;
    end

    # Space Loop Sweep 1 =====================================================================================================
    for i = 2:1:Nx + 1
    for j = 2:1:Nx + 1

        A = c * ( U[1] - U[2] ) / 2;
        B = c * ( -U[1] + U[2] ) / 2;
        D = c * ( U[1] + U[2] ) / 2;

    # First order solution
        phi_first_order[i, j, n + 1] = ( phi_first_order[i, j, n] + A * phi_first_order[i-1, j, n + 1] + B * phi_first_order[i, j-1, n + 1] + D * phi_first_order[i-1, j-1, n + 1]  ) / ( 1 + A + B + D );
        
    # Predictor
        phi_predictor[i, j, n + 1] = ( phi[i, j, n] + A * phi_predictor[i-1, j, n + 1] + B * phi_predictor[i, j-1, n + 1] + D * phi_predictor[i-1, j-1, n + 1]  ) / ( 1 + A + B + D );

    # Corrector
        r_upwind_n = phi_old[i, j] - phi[i , j, n] + phi[i - 1, j - 1, n + 1] - phi[i - 1, j - 1, n];
        r_downwind_n = phi_predictor[i, j, n] - phi_predictor[i, j, n + 1] + phi_predictor_n2[i - 1, j - 1, n + 1] - phi_predictor[i - 1, j - 1, n + 1];
   
        # ENO parameter 
        abs(r_downwind_n) <= abs(r_upwind_n) ? p[i, j, n + 1] = 0 : p[i, j, n + 1] = 0;

    # Second order solution
        phi[i, j, n + 1] = ( phi[i, j, n] + A * phi[i-1, j, n + 1] + B * phi[i, j-1, n + 1] + D * phi[i-1, j-1, n + 1] 
                                          - 0.5 * ( ( 1 - p[i, j, n + 1] ) .* r_downwind_n + p[i, j, n + 1] .* r_upwind_n ) ) / ( 1 + A + B + D );
        
    # Predictor for next time step
        phi_predictor_n2[i, j, n + 1] = ( phi[i, j, n + 1]  + A * phi_predictor_n2[i-1, j, n + 1] + B * phi_predictor_n2[i, j-1, n + 1] + D * phi_predictor_n2[i-1, j-1, n + 1]  ) / ( 1 + A + B + D );

    end
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
trace3 = contour(x = x1, y = x2, z = phi_first_order[:, :, end], name = "First-order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1, line_dash="dash")
layout = Layout(title = "Linear advection equation", xaxis_title = "x1", yaxis_title = "x2", zaxis_title = "phi", colorbar = false)

plot_phi = plot([trace1, trace2, trace3], layout)

# plot_error = plot(surface(x = x1, y = x2, z = abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))* h^2))
