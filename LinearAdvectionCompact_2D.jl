# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS

include("InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

# Courant number
c = 1.5;

# Grid settings - 2D regular grid
x1L = -pi/2
x1R = pi
x2L = -pi/2
x2R = pi
Nx = 80 * 2^level
h = (x1R - x1L) / Nx

# Velocity
U = [1.0, 1.0]

# Time
tau = c * h / maximum(abs.(U))
Ntau = Int(Nx / 10)
# Ntau = 1;

# Initial condition
phi_0(x1, x2) = cos.(x1) .* cos.(x2);

# Exact solution
phi_exact(x1, x2, t) = phi_0.(x1 - U[1] .* t, x2 - U[2] .* t);

## Comptutation

# Grid initialization
x1 = range(x1L, x1R, length = Nx + 1)
x2 = range(x2L, x2R, length = Nx + 1)
t = range(0, Ntau * tau, length = Ntau + 1)

X1 = repeat(x1, 1, Nx+1)
X2 = repeat(x2', Nx+1, 1)
T = repeat(t, 1, Ntau+1)

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
sx = 0.5; 
sy = 0.5; 
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

    ghost_point_left = phi_exact.(x1L-h, x2, n * tau);
    ghost_point_right = phi_exact.(x1R+h, x2, (n-1) * tau);
    ghost_point_up = phi_exact.(x1, x2R+h, (n-1) * tau);
    ghost_point_down = phi_exact.(x1, x2L-h, n * tau);

    # Space Loop Sweep 1 =====================================================================================================
    for i = 2:Nx + 1
    for j = 2:Nx + 1

        if i < Nx + 1
            phi_i_plus = phi[i + 1, j, n];
        else
            phi_i_plus = ghost_point_right[j];
        end

        if i > 2
            phi_i_minus_n_plus = phi[i - 2, j, n + 1];
        else
            phi_i_minus_n_plus = ghost_point_left[j];
        end

        if j < Nx + 1
            phi_j_plus = phi[i, j + 1, n];
        else
            phi_j_plus = ghost_point_up[i];
        end

        if j > 2
            phi_j_minus_n_plus = phi[i, j - 2, n + 1];
        else
            phi_j_minus_n_plus = ghost_point_down[i];
        end

    # First order solution
        phi_first_order[i, j, n + 1] = ( phi_first_order[i, j, n] + c * phi_first_order[i - 1, j, n + 1] + c * phi_first_order[i, j - 1, n + 1] ) / ( 1 + 2*c );
        
    # Predictor
        phi_predictor[i, j, n + 1] = ( phi_predictor[i, j, n] + c * phi[i - 1, j, n + 1] + c * phi[i, j - 1, n + 1]) / ( 1 + 2*c );

    # Corrector
        r_downwind_i = - phi[i, j, n] + phi_predictor[i - 1, j, n + 1] - phi_predictor[i, j, n + 1] + phi_i_plus;
        r_upwind_i = - phi[i - 1, j, n] + phi_i_minus_n_plus + phi[i, j, n] - phi[i - 1, j, n + 1];

        r_downwind_j = - phi[i, j, n] + phi_predictor[i, j - 1, n + 1] - phi_predictor[i, j, n + 1] + phi_j_plus;
        r_upwind_j = - phi[i, j - 1, n] + phi_j_minus_n_plus + phi[i, j, n] - phi[i, j - 1, n + 1]; 

    # Second order solution
        # phi[i, j, n + 1] = ( phi[i, j, n] + c * ( phi[i - 1, j, n + 1] - 0.5 * sx[i - 1, j, n + 1] .* r_downwind_i_minus - 0.5 * sx[i, j, n + 1] .* r_downwind_i ) + 
        #                                     c * ( phi[i, j - 1, n + 1] - 0.5 * sy[i, j - 1, n + 1] .* r_downwind_j_minus - 0.5 * sy[i, j, n + 1] .* r_downwind_j ) ) / ( 1 + 2*c );
        phi[i, j, n + 1] = ( phi[i, j, n] + c * ( phi[i - 1, j, n + 1] - 0.5 * (1-sx) .* r_downwind_i - 0.5 * sx .* r_upwind_i ) + 
                                            c * ( phi[i, j - 1, n + 1] - 0.5 * (1-sy) .* r_downwind_j - 0.5 * sy .* r_upwind_j ) ) / ( 1 + 2*c );

    end
    end

end

# Print the error

println("Error L2: ", sum(abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau ))) * h^2)
# println("Error L2: ", sum(reduce(+, (abs.(phi[:, :, n] - phi_exact.(X1, X2, n * tau)) for n in 1:Ntau)) * tau * h^2))
println("Error L_inf: ", norm(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau), Inf)* h^2)

# Print first order error
# println("Error L2 first order: ", sum(reduce(+, (abs.(phi_first_order[:, :, n] - phi_exact.(X1, X2, n * tau)) for n in 1:Ntau)) * tau * h^2))
println("Error L2 first order: ", sum(abs.(phi_first_order[:, :, end] - phi_exact.(X1, X2, Ntau * tau))) * h^2)
println("Error L_inf firts order: ", norm(phi_first_order[:, :, end] - phi_exact.(X1, X2, Ntau * tau), Inf)* h^2)

# Plot of the result at the final time together with the exact solution

# trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Compact scheme solution")
trace2 = contour(x = x1, y = x2, z = phi_exact.(X1, X2, Ntau * tau), name = "Exact solution", showscale=false, contours_coloring="lines", colorscale = "Black" , line_width=2)
trace3 = contour(x = x1, y = x2, z = phi[:, :, end], name = "Compact", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1)
trace1 = contour(x = x1, y = x2, z = phi_first_order[:, :, end], name = "First order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=0.5)


layout = Layout(title = "Linear advection equation", xaxis_title = "x1", yaxis_title = "x2", zaxis_title = "phi", colorbar = false)

plot_phi = plot([trace1, trace2, trace3], layout)

# plot_error = plot(surface(x = x1, y = x2, z = abs.(phi[:, :, end] - phi_exact.(X1, X2, Ntau * tau))* h^2))
