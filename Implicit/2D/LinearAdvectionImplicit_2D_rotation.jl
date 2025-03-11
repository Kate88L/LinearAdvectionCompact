#  iterative 2nd order fully implicit scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames

include("../../Utils/InitialFunctions.jl")
include("../../Utils/ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 3;

K = 4; # Number of iterations for the second order correction


# Courant number
C = 3;

# Grid settings
xL = 0 #- 2 * π / 2
xR = 0.5 #3 * π / 2
yL = xL
yR = xR
N = 100 * 2^level
h = (xR - xL) / N # regular grid

# Velocity
u(x, y) = -y
v(x, y) = x

## Comptutation

# Grid initialization
x = range(xL, xR + h, length = N + 2)
y = range(yL - h, yR, length = N + 2)

X = repeat(x, 1, N + 2)
Y = repeat(y', N + 2, 1)

# Initial condition
phi_0(x, y) = nonSmoothRotation(x, y, 0)

# Exact solution
phi_exact(x, y, t) = nonSmoothRotation(x, y, t)

# Time
Tfinal = 0.1
Ntau = 1 * 2^level
tau = C * h / maximum(max(abs.(u.(x, y)), abs.(v.(x, y))))
Ntau = Int(round(Tfinal / tau))

# Ntau = 1

t = range(0, Ntau * tau + tau, length = Ntau + 2) 

c = zeros(N+2, N+2) .+ u.(x, y)' * tau / h
d = zeros(N+2, N+2) .+ v.(x, y) * tau / h

# Predictor and corrector (including the ghost points)
phi_first_order = zeros(N + 2, N + 2, Ntau + 1);
phi_star = zeros(N + 2, N + 2, Ntau + 3);
phi = zeros(N + 2, N + 2, Ntau + 3);

# Initial condition
phi_first_order[:, :, 1] = phi_0.(X, Y); 
phi_star[:, :, 2] = phi_0.(X, Y); 
phi[:, :, 2] = phi_0.(X, Y);

phi_star[:, :, 1] = phi_exact.(X, Y, -tau); # ghost points in time
phi[:, :, 1] = phi_exact.(X, Y, -tau); # ghost points in time

# Boundary condition
for n = 2 : Ntau + 1
    phi_first_order[:, 1, n] = phi_exact.(x, yL - h, t[n] );
    phi_first_order[:, 2, n] = phi_exact.(x, yL, t[n] );
    phi_first_order[end - 1, :, n] = phi_exact.(xR, y, t[n]);
    phi_first_order[end, :, n] = phi_exact.(xR + h, y, t[n]);

    phi_star[:, 1, n + 1] = phi_exact.(x, yL - h, t[n] );
    phi_star[:, 2, n + 1] = phi_exact.(x, yL, t[n] );
    phi_star[end - 1, :, n + 1] = phi_exact.(xR, y, t[n]);
    phi_star[end, :, n + 1] = phi_exact.(xR + h, y, t[n]);

    phi[:, 1, n + 1] = phi_exact.(x, yL - h, t[n] );
    phi[:, 2, n + 1] = phi_exact.(x, yL, t[n] );
    phi[end - 1, :, n + 1] = phi_exact.(xR, y, t[n]);
    phi[end, :, n + 1] = phi_exact.(xR + h, y, t[n]);
end

phi_star[:, 1, end] = phi_exact.(x, yL - h, t[end] );
phi_star[:, 2, end] = phi_exact.(x, yL, t[end] );
phi_star[end - 1, :, end] = phi_exact.(xR, y, t[end]);
phi_star[end, :, end] = phi_exact.(xR + h, y, t[end]);

phi[:, 1, end] = phi_exact.(x, yL - h, t[end] );
phi[:, 2, end] = phi_exact.(x, yL, t[end] );
phi[end - 1, :, end] = phi_exact.(xR, y, t[end]);
phi[end, :, end] = phi_exact.(xR + h, y, t[end]);

## ENO
ω0 = 0
α0 = 0

ω_x = zeros(N + 2, N + 2) .+ ω0
ω_y = zeros(N + 2, N + 2) .+ ω0
α = zeros(N + 2, N + 2) .+ α0

## SOLUTION ----------------------------------------------------------------------------
@time begin
# Time Loop
for n = 2 : Ntau + 1
    
    # Compute ALL PREDICTORS 
    for i = N: - 1 : 1
        for j = 3 : N + 2     
            phi_first_order[i, j, n] = ( phi_first_order[i, j, n - 1] + abs(c[i, j]) * phi_first_order[i + 1, j, n] 
                                                                      + abs(d[i, j]) * phi_first_order[i, j - 1, n]) / (1 + abs(c[i, j]) + abs(d[i, j]) );
            
            # First order predictors
            phi_star[i, j, n] = ( phi[i, j, n - 1] + abs(c[i, j]) * phi_star[i + 1, j, n] 
                                                   + abs(d[i, j]) * phi_star[i, j - 1, n]) / (1 + abs(c[i, j]) + abs(d[i, j]) );
            phi_star[i, j, n + 1] = ( phi_star[i, j, n] + abs(c[i, j]) * phi_star[i + 1, j, n + 1] 
                                                   + abs(d[i, j]) * phi_star[i, j - 1, n + 1]) / (1 + abs(c[i, j]) + abs(d[i, j]) );
            phi_star[i, j, n + 2] = ( phi_star[i, j, n + 1] + abs(c[i, j]) * phi_star[i + 1, j, n + 2] 
                                                   + abs(d[i, j]) * phi_star[i, j - 1, n + 2]) / (1 + abs(c[i, j]) + abs(d[i, j]) );
            
            # Second order predictors
            phi[i, j, n + 1] = ( phi[i, j, n] - 1 / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi[i, j, n - 1] )
                                        + abs(c[i, j]) * ( phi[i + 1, j, n + 1] 
                                              - 1 / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i + 1, j, n + 1] + phi[i + 2, j, n + 1] )  )
                                        + abs(d[i, j]) * ( phi[i, j - 1, n + 1]
                                              - 1 / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi[i, j - 2, n + 1] ) ) ) / (1 + abs(c[i, j]) + abs(d[i, j]));
            phi[i, j, n + 2] = ( phi[i, j, n + 1] - 1 / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n] )
                                        + abs(c[i, j]) * ( phi[i + 1, j, n + 2] 
                                              - 1 / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i + 1, j, n + 2] + phi[i + 2, j, n + 2] )  )
                                        + abs(d[i, j]) * ( phi[i, j - 1, n + 2]
                                              - 1 / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j - 1, n + 2] + phi[i, j - 2, n + 2] ) ) ) / (1 + abs(c[i, j]) + abs(d[i, j]));
        end
    end

    # Extrapolation
    phi[N + 1, :, n + 1] = 3 * phi[N, :, n + 1] - 3 * phi[N - 1, :, n + 1] + phi[N - 2, :, n + 1];
    phi[:, 2, n + 1] = 3 * phi[:, 3, n + 1] - 3 * phi[:, 4, n + 1] + phi[:, 5, n + 1];
    phi[N + 1, :, n + 2] = 3 * phi[N, :, n + 2] - 3 * phi[N - 1, :, n + 2] + phi[N - 2, :, n + 2];
    phi[:, 2, n + 2] = 3 * phi[:, 3, n + 2] - 3 * phi[:, 4, n + 2] + phi[:, 5, n + 2];

    # CORRECTION
    for k = 1:K

        phi_star[:, :, n] = phi[:, :, n];
        phi_star[:, :, n + 1] = phi[:, :, n + 1];
        phi_star[:, :, n + 2] = phi[:, :, n + 2];

        for i = N: - 1 : 2
            for j = 3 : N + 1      

                # ENO parameter (0 - upwnd, 1 - central)
                ω_x[i, j] = ifelse( abs(phi_star[i, j, n + 1] - 2 * phi_star[i + 1, j, n + 1] + phi[i + 2, j, n + 1]) <= abs(phi_star[i + 1, j, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i - 1, j, n + 1]), 0, 1)
                ω_y[i, j] = ifelse( abs(phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi[i, j - 2, n + 1]) <= abs(phi_star[i, j + 1, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i, j - 1, n + 1]), 0, 1)
                α[i, j] = ifelse( abs(phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi[i, j, n - 1]) <= abs(phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n]), 0, 1)
           
                phi[i, j, n + 1] = ( phi[i, j, n] - α[i, j] / 2 * ( phi_star[i, j, n + 2] - 2 * phi_star[i, j, n + 1] + phi_star[i, j, n] ) 
                                                - ( 1 - α[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j, n] + phi[i, j, n - 1] )
                                    + abs(c[i, j]) * ( phi[i + 1, j, n + 1] 
                                                - ω_x[i, j] / 2 * ( phi_star[i + 1, j, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i - 1, j, n + 1] )
                                                - (1 - ω_x[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i + 1, j, n + 1] + phi[i + 2, j, n + 1] ) ) 
                                    + abs(d[i, j]) * ( phi[i, j - 1, n + 1] 
                                                - ω_y[i, j] / 2 * ( phi_star[i, j + 1, n + 1] - 2 * phi_star[i, j, n + 1] + phi_star[i, j - 1, n + 1] )
                                                - (1 - ω_y[i, j] ) / 2 * ( phi_star[i, j, n + 1] - 2 * phi_star[i, j - 1, n + 1] + phi[i, j - 2, n + 1] ) ) ) / ( 1 + abs(c[i,j]) + abs(d[i,j]) );
           
                phi[i, j, n + 2] = ( phi[i, j, n + 1] + abs(c[i,j]) * phi[i + 1, j, n + 2] + abs(d[i,j]) * phi[i, j - 1, n + 2] ) / ( 1 + abs(c[i,j]) + abs(d[i,j]) );
            end
        end

        # Extrapolation
        # phi[N + 2, :, n + 1] = 3 * phi[N, :, n + 1] - 3 * phi[N - 1, :, n + 1] + phi[N - 2, :, n + 1];
        # phi[:, 1, n + 1] = 3 * phi[:, 3, n + 1] - 3 * phi[:, 4, n + 1] + phi[:, 5, n + 1];
        # phi[N + 2, :, n + 2] = 3 * phi[N, :, n + 2] - 3 * phi[N - 1, :, n + 2] + phi[N - 2, :, n + 2];
        # phi[:, 1, n + 2] = 3 * phi[:, 3, n + 2] - 3 * phi[:, 4, n + 2] + phi[:, 5, n + 2];
    end
end
end

# Error
Error_t_h = tau * h^2 * sum(sum(abs.(phi[:, :, n + 1] - phi_exact.(X, Y, t[n]))) for n in 1:Ntau+1)
println("Error t*h: ", Error_t_h)

# error first order
Error_t_h_first_order = tau * h^2 * sum(sum(abs.(phi_first_order[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 2:Ntau + 1)
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

println("minimum derivative x: ", minimum(d_phi_x))
println("maximum derivative x: ", maximum(d_phi_x))
println("minimum derivative y: ", minimum(d_phi_y))
println("maximum derivative y: ", maximum(d_phi_y))

# Plot of the result at the final time together with the exact solution
trace1 = contour(x = x, y = y, z = phi_exact.(X, Y, t[end - 1])', name = "Exact solution", showscale=false, contours_coloring="lines", colorscale="Greys", line_width=2)
trace2 = contour(x = x, y = y, z = phi_first_order[:, :, end]',name = "First order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_dash="dash", line_width=1)
trace3 = contour(x = x, y = y, z = phi[:, :, end - 1]',name = "Second order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1)
layout = Layout(title = "Linear advection equation", xaxis_title = "x", yaxis_title = "y", zaxis_title = "phi", colorbar = false)

plot_phi = plot([trace1, trace2, trace3], layout)

# Plot derivative
trace1_d_x = contour(x = x, y = y, z = d_phi_x[:, :], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=2)

trace1_d_y = contour(x = x, y = y, z = d_phi_y[:, :], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=2)

plot_phi_d_x = plot([trace1_d_x])
plot_phi_d_y = plot([trace1_d_y])

p = [plot_phi; plot_phi_d_x plot_phi_d_y]
relayout!(p, width=800, height=1000)
p

