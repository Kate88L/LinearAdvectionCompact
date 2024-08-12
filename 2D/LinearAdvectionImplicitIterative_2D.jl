#  iterative 2nd order fully implicit scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 4;

# Courant number
C = 1;

# Grid settings
xL = -1 #- 2 * π / 2
xR = 1 #3 * π / 2
yL = xL
yR = xR
N = 80 * 2^level
h = (xR - xL) / N # regular grid

# Velocity
u(x, y) = 1.0
v(x, y) = 1.0

# Initial condition
# phi_0(x, y) = x.^2 + y.^2
phi_0(x, y) = piecewiseLinear2D(x, y);

# Exact solution
phi_exact(x, y, t) = phi_0.(x - u(x,y) .* t, y - v(x,y) .* t);

## Comptutation

# Grid initialization
x = range(xL, xR + h, length = N + 2)
y = range(yL, yR + h, length = N + 2)

X = repeat(x, 1, N + 2)
Y = repeat(y', N + 2, 1)

# Time
Tfinal = 1 * π / sqrt(7) * 2 / 20
Ntau = 1 * 2^level
tau = C * h / maximum(max(u.(x, y), v.(x, y)))
Ntau = Int(round(Tfinal / tau))

t = range(0, (Ntau + 1) * tau, length = Ntau + 2) 

c = zeros(N+2, N+2) .+ u.(x, y) * tau / h
d = zeros(N+2, N+2) .+ v.(x, y) * tau / h

# Predictor and corrector (including the ghost points)
phi_star = zeros(N + 2, N + 2, Ntau + 2);
phi = zeros(N + 2, N + 2, Ntau + 2);
phi_first_order = zeros(N + 2, N + 2, Ntau + 2);

# Initial condition
phi[:, :, 1] = phi_0.(X, Y);
phi_first_order[:, :, 1] = phi_0.(X, Y);
phi_star[:, :, 1] = phi_0.(X, Y);

ghost_point_left = zeros(N + 2, Ntau + 2);
ghost_point_bottom = zeros(N + 2, Ntau + 2);

# Boundary condition
for n = 2 : Ntau + 2
    phi[1, :, n] = phi_exact.(xL, y, t[n]);
    phi[:, 1, n] = phi_exact.(x, yL, t[n] );
    phi_star[1, :, n] = phi_exact.(xL, y, t[n]);
    phi_star[:, 1, n] = phi_exact.(x, yL, t[n]);
    phi_first_order[1, :, n] = phi_exact.(xL, y, t[n]);
    phi_first_order[:, 1, n] = phi_exact.(x, yL, t[n] );

    # Ghost point on the inflow boundary
    ghost_point_left[:, n] = phi_exact.(xL - h, y, t[n]);
    ghost_point_bottom[:, n] = phi_exact.(x, yL - h, t[n]);
end

# Ghost point on the time -1
ghost_point_time = phi_exact.(X, Y, -tau);

## ENO
ω0 = 0
α0 = 0

ω_x = zeros(N + 1, N + 1) .+ ω0
ω_y = zeros(N + 1, N + 1) .+ ω0
α = zeros(N + 1, N + 1) .+ α0

## SOLUTION ----------------------------------------------------------------------------

# Time Loop
for n = 1 : Ntau
    
    # Initialize ghost point in time - point ϕ_i_j_n-2
    if n > 1 
        phi_old = phi[:, :, n - 1];
    else
        phi_old = ghost_point_time;
    end

    # Compute ALL first order predictors 
    for i = 2 : N + 2
    for j = 2 : N + 2
       
        phi_first_order[i, j, n + 1] = ( phi_first_order[i, j, n] + c[i, j] * phi_first_order[i - 1, j, n + 1] 
                                                                  + d[i, j] * phi_first_order[i, j - 1, n + 1] ) / (1 + c[i, j] + d[i, j]);

        phi_star[i, j, n] = ( phi_old[i, j] + c[i, j] * phi_star[i - 1, j, n] + d[i, j] * phi_star[i, j - 1, n] ) / (1 + c[i, j] + d[i, j]);
        phi_star[i, j, n + 1] = ( phi_star[i, j, n] + c[i, j] * phi_star[i - 1, j, n + 1] + d[i, j] * phi_star[i, j - 1, n + 1] ) / (1 + c[i, j] + d[i, j]);
        phi_star[i, j, n + 2] = ( phi_star[i, j, n + 1] + c[i, j] * phi_star[i - 1, j, n + 2] + d[i, j] * phi_star[i, j - 1, n + 2] ) / (1 + c[i, j] + d[i, j]);

        if n < Ntau
            phi_star[i, j, n + 3] = ( phi_star[i, j, n + 2] + c[i, j] * phi_star[i - 1, j, n + 3] + d[i, j] * phi_star[i, j - 1, n + 3] ) / (1 + c[i, j] + d[i, j]);
        end
    end
    end

    # First iteration - compute phi_i_j_n+1 and phi_i_j_n+2 if you can
    for i = 2 : N + 1
    for j = 2 : N + 1

        if i > 2
            phi_left = phi_star[i - 2, j, n + 1];
        else
            phi_left = ghost_point_left[j, n + 1];
        end

        if j > 2
            phi_bottom = phi_star[i, j - 2, n + 1];
        else
            phi_bottom = ghost_point_bottom[i, n + 1];
        end

        # ENO
        if abs(phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n]) <= abs(phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi_old[i, j])
            α[i, j] = 1;
        else
            α[i, j] = 0;
        end

        if abs(phi_star[i + 1, j, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i - 1, j, n + 1]) <= abs(phi_star[i, j, n + 1] - 2 * phi_star[i - 1, j, n + 1] + phi_left )
            ω_x[i, j] = 1;
        else
            ω_x[i, j] = 0;
        end

        if abs(phi_star[i, j + 1, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i, j - 1, n + 1]) <= abs(phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi_bottom )
            ω_y[i, j] = 1;
        else
            ω_y[i, j] = 0;
        end

        # Second order solution
        phi[i, j, n + 1] = ( phi[i, j, n] - α[i, j] / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n] ) 
                                - ( 1 - α[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi_old[i, j] )
                            + c[i, j] * ( phi[i - 1, j, n + 1] 
                                        - ω_x[i, j] / 2 * ( phi_star[i + 1, j, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i - 1, j, n + 1] )
                                - (1 - ω_x[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i - 1, j, n + 1] + phi_left ) ) 
                            + d[i, j] * ( phi[i, j - 1, n + 1] 
                                        - ω_y[i, j] / 2 * ( phi_star[i, j + 1, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i, j - 1, n + 1] )
                                - (1 - ω_y[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi_bottom ) ) ) / ( 1 + c[i,j] + d[i,j] );
    end
    end
    # Extrapolation
    phi[N + 2, :, n + 1] = 3 * phi[N + 1, :, n + 1] - 3 * phi[N, :, n + 1] + phi[N - 1, :, n + 1];
    phi[:, N + 2, n + 1] = 3 * phi[:, N + 1, n + 1] - 3 * phi[:, N, n + 1] + phi[:, N - 1, n + 1];

    if n < Ntau 

        for i = 2 : N + 1
        for j = 2 : N + 1

            if i > 2
                phi_left = phi_star[i - 2, j, n + 2];
            else
                phi_left = ghost_point_left[j, n + 2];
            end

            if j > 2
                phi_bottom = phi_star[i, j - 2, n + 2];
            else
                phi_bottom = ghost_point_bottom[i, n + 2];
            end

            # ENO
            if abs(phi_star[i, j, n + 3] - 2 * phi_star[i, j, n + 2] + phi_star[i, j, n + 1]) <= abs(phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n])
                α[i, j] = 1;
            else
                α[i, j] = 0;
            end

            if abs(phi_star[i + 1, j, n + 2] - 2 * phi_star[i, j, n + 2] + phi_star[i - 1, j, n + 2]) <= abs(phi_star[i, j, n + 2] - 2 * phi_star[i - 1, j, n + 2] + phi_left )
                ω_x[i, j] = 1;
            else
                ω_x[i, j] = 0;
            end

            if abs(phi_star[i, j + 1, n + 2] - 2 * phi_star[i, j, n + 2] + phi_star[i, j - 1, n + 2]) <= abs(phi_star[i, j, n + 2] - 2 * phi_star[i, j - 1, n + 2] + phi_bottom )
                ω_y[i, j] = 1;
            else
                ω_y[i, j] = 0;
            end

            # Second order solution
            phi[i, j, n + 2] = ( phi[i, j, n + 1] - α[i, j] / 2 * ( phi_star[i, j, n + 3] - 2 * phi_star[i, j, n + 2] + phi_star[i, j, n + 1] ) 
                                    - ( 1 - α[i, j] ) / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_old[i, j] )
                                + c[i, j] * ( phi[i - 1, j, n + 2] 
                                            - ω_x[i, j] / 2 * ( phi_star[i + 1, j, n + 2] - 2 * phi_star[i, j, n + 2] + phi_star[i - 1, j, n + 2] )
                                    - (1 - ω_x[i, j] ) / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i - 1, j, n + 2] + phi_left ) ) 
                                + d[i, j] * ( phi[i, j - 1, n + 2] 
                                            - ω_y[i, j] / 2 * ( phi_star[i, j + 1, n + 2] - 2 * phi_star[i, j, n + 2] + phi_star[i, j - 1, n + 2] )
                                    - (1 - ω_y[i, j] ) / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j - 1, n + 2] + phi_bottom ) ) ) / ( 1 + c[i,j] + d[i,j] );
        end
        end
        # Extrapolation
        phi[N + 2, :, n + 2] = 3 * phi[N + 1, :, n + 2] - 3 * phi[N, :, n + 2] + phi[N - 1, :, n + 2];
        phi[:, N + 2, n + 2] = 3 * phi[:, N + 1, n + 2] - 3 * phi[:, N, n + 2] + phi[:, N - 1, n + 2];

        # Second iteration - compute final phi_i_j_n+1
        phi_star[:, :, n] = phi[:, :, n];
        phi_star[:, :, n + 1] = phi[:, :, n + 1];
        phi_star[:, :, n + 2] = phi[:, :, n + 2];

        for i = 2 : N + 1
        for j = 2 : N + 1

            if i > 2
                phi_left = phi[i - 2, j, n + 1];
            else
                phi_left = ghost_point_left[j, n + 1];
            end

            if j > 2
                phi_bottom = phi[i, j - 2, n + 1];
            else
                phi_bottom = ghost_point_bottom[i, n + 1];
            end

            # ENO
            if abs(phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n]) <= abs(phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi_old[i, j])
                α[i, j] = 1;
            else
                α[i, j] = 0;
            end

            if abs(phi_star[i + 1, j, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i - 1, j, n + 1]) <= abs(phi_star[i, j, n + 1] - 2 * phi_star[i - 1, j, n + 1] + phi_left )
                ω_x[i, j] = 1;
            else
                ω_x[i, j] = 0;
            end

            if abs(phi_star[i, j + 1, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i, j - 1, n + 1]) <= abs(phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi_bottom )
                ω_y[i, j] = 1;
            else
                ω_y[i, j] = 0;
            end

            # Second order solution
            phi[i, j, n + 1] = ( phi[i, j, n] - α[i, j] / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n] ) 
                                    - ( 1 - α[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi_old[i, j] )
                                + c[i, j] * ( phi[i - 1, j, n + 1] 
                                            - ω_x[i, j] / 2 * ( phi_star[i + 1, j, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i - 1, j, n + 1] )
                                    - (1 - ω_x[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i - 1, j, n + 1] + phi_left ) ) 
                                + d[i, j] * ( phi[i, j - 1, n + 1] 
                                            - ω_y[i, j] / 2 * ( phi_star[i, j + 1, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i, j - 1, n + 1] )
                                    - (1 - ω_y[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi_bottom ) ) ) / ( 1 + c[i,j] + d[i,j] );


        end
        end
    end

    # Extrapolation
    phi[N + 2, :, n + 1] = 3 * phi[N + 1, :, n + 1] - 3 * phi[N, :, n + 1] + phi[N - 1, :, n + 1];
    phi[:, N + 2, n + 1] = 3 * phi[:, N + 1, n + 1] - 3 * phi[:, N, n + 1] + phi[:, N - 1, n + 1];

end

# Error
Error_t_h = tau * h^2 * sum(sum(abs.(phi[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 1:Ntau+1)
println("Error t*h: ", Error_t_h)

# error first order
Error_t_h_first_order = tau * h^2 * sum(sum(abs.(phi_first_order[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 1:Ntau+1)
println("Error t*h first order: ", Error_t_h_first_order)

# Compute gradient of the solution in the last time step
d_phi_x = zeros(N + 1, N + 1);
d_phi_y = zeros(N + 1, N + 1);

for i = 1 : N + 1
for j = 1 : N + 1
   d_phi_x[i, j] = (phi[i + 1, j, end-1] - phi[i, j, end-1]) / h;
   d_phi_y[i, j] = (phi[i, j + 1, end-1] - phi[i, j, end-1]) / h;
end
end

# Plot of the result at the final time together with the exact solution
trace1 = contour(x = x, y = y, z = phi_exact.(X, Y, Ntau * tau), name = "Exact solution", showscale=false, contours_coloring="lines", colorscale="Greys", line_width=2)
trace2 = contour(x = x, y = y, z = phi[:, :, end - 1], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1)
trace3 = contour(x = x, y = y, z = phi_first_order[:, :, end - 1], name = "First order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_dash="dash", line_width=1)
layout = Layout(title = "Linear advection equation", xaxis_title = "x", yaxis_title = "y", zaxis_title = "phi", colorbar = false)

plot_phi = plot([trace1, trace3, trace2], layout)

# Plot derivative
trace1_d_x = surface(x = x, y = y, z = d_phi_x[:, :], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=2)

trace1_d_y = surface(x = x, y = y, z = d_phi_y[:, :], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=2)

plot_phi_d_x = plot([trace1_d_x])
plot_phi_d_y = plot([trace1_d_y])

p = [plot_phi; plot_phi_d_x plot_phi_d_y]
relayout!(p, width=800, height=1000)
p