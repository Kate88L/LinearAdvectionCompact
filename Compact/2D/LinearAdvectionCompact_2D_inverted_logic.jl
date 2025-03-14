# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS

include("../../Utils/InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

# Courant number
c = 2;
d = c;

# Grid settings - 2D regular grid
x1L = -pi/2
x1R = pi
x2L = -pi/2
x2R = pi
Nx = 100 * 2^level
h = (x1R - x1L) / Nx

# Velocity
U = [1.0, 1.0]

# Time
tau = sqrt(2) * c * h / maximum(abs.(U))
Ntau = Int(Nx / 10)
# Ntau = 1;

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

# Initial condition
phi[:, :, 1] = phi_0.(X1, X2);

# Ghost point on the time -1
ghost_point_time = phi_exact.(X1, X2, -tau);
future_ghost_point_time = phi_exact.(X1, X2, (Ntau + 1) * tau);

α = 1/2;

# Boundary conditions
for n = 0:Ntau
    phi[1, :, n + 1] = phi_exact.(x1[1], x2, n * tau);
    phi[:, 1, n + 1] = phi_exact.(x1, x2[1], n * tau);
end

# Space Loop Sweep 1 =====================================================================================================
for i = 2:1:Nx + 1
for j = 2:1:Nx + 1

    # Time Loop
    for n = 1:Ntau

        if n > 1
            phi_old = phi[i, j, n - 1];
        else
            phi_old = ghost_point_time[i , j];
        end

        if n < Ntau
            phi_new_x = phi[i - 1, j, n + 2];
            phi_new_y = phi[i, j - 1, n + 2];
        else
            phi_new_x = future_ghost_point_time[i - 1, j];
            phi_new_y = future_ghost_point_time[i, j - 1];
        end

    # Corrector
        r_upwind_n =  phi_old + phi[i - 1, j, n + 1] + phi[i, j - 1, n + 1] - phi[i - 1, j, n] - phi[i, j - 1, n];
        r_downwind_n = 2 * phi[i, j, n] - 0.5 * phi[i - 1, j, n] - 0.5 * phi[i, j - 1, n] + 0.5 * phi_new_x + 0.5 * phi_new_y;
   
    # Second order solution
        phi[i, j, n + 1] = ( phi[i, j, n] + sqrt(2) * c * phi[i - 1, j - 1, n + 1]- 0.5 * ( ( 1 - α ) .* r_downwind_n + α .* r_upwind_n ) ) / ( 1 + sqrt(2) * c + 0.5 * ( α - 2 ));
    end

end  
end

# Print the error
println("Error L2: ", sum(abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))) * h^2)
println("Error L_inf: ", norm(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau), Inf)* h^2)

# Plot of the result at the final time together with the exact solution
trace1 = contour(x = x1, y = x2, z = phi_exact.(X1, X2, Ntau * tau), name = "Exact solution", showscale=false, contours_coloring="lines", colorscale="Greys", line_width=2)
trace2 = contour(x = x1, y = x2, z = phi[:, :, end], name = "Compact", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1)
layout = Layout(title = "Linear advection equation", xaxis_title = "x1", yaxis_title = "x2", zaxis_title = "phi", colorbar = false)

plot_phi = plot([trace1, trace2], layout)

# plot_error = plot(surface(x = x1, y = x2, z = abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))* h^2))
