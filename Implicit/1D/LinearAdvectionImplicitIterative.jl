#  iterative 2nd order fully implicit scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames

include("../../Utils/InitialFunctions.jl")
include("../../Utils/ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

K = 2; # Number of iterations for the second order correction

# Courant number
C = 5;

# Grid settings
xL = 0 # - 1 * π / 2
xR = 1.5 # 3 * π / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
# u(x) = 1 + 3/4 * cos(x)
# u(x) = 2 + 3/2 * cos(x)
u(x) = 1

# Initial condition
# phi_0(x) = asin( sin(x + π/2) ) * 2 / π;
# phi_0(x) = cos.(x);
# phi_0(x) = exp.(-10*(x+0)^2);
phi_0(x) = piecewiseLinear(x);

# Exact solution
# phi_exact(x, t) = cosVelocityNonSmooth(x, t); 
# phi_exact(x, t) = cosVelocitySmooth(x, t);
phi_exact(x, t) = phi_0.(x - t);             

## Comptutation

# Grid initialization with ghost point on the left and right
x = range(xL, xR+h, length = Nx + 2)

# Time
T = 1 * π / sqrt(7) * 2 / 10
# T = 2 *π / sqrt(3)
# tau = C * h / u
Ntau = 1 * 2^level
tau = T / Ntau
tau = C * h / maximum(u.(x))
Ntau = Int(round(T / tau))

# Ntau = 1

c = zeros(Nx+2) .+ u.(x) * tau / h

# predictor
phi1 = zeros(Nx + 2, Ntau + 3);
# corrector and the final solution
phi2 = zeros(Nx + 2, Ntau + 2);
# first order
phi_first_order = zeros(Nx + 2, Ntau + 2)

# Initial condition
phi1[:, 1] = phi_0.(x);
phi2[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi1[1, :] = phi_exact.(x[1], range(0, (Ntau+2) * tau, length = Ntau + 3));
phi2[1, :] = phi_exact.(x[1], range(0, (Ntau+1) * tau, length = Ntau + 2));
phi_first_order[1, :] = phi_exact.(x[1], range(0, (Ntau+1) * tau, length = Ntau + 2));

# Ghost point on the left side 
ghost_point_left = phi_exact.(xL - h, range(0, (Ntau+1) * tau, length = Ntau + 2));
# Ghost point on the time -1
ghost_point_time = phi_exact.(x, -tau);

# WENO parameters
ϵ = 1e-16;
ω0 = 2/3;
α0 = 2/3;

ω = zeros(Nx + 1) .+ ω0;
α = zeros(Nx + 1) .+ α0;

l = zeros(Nx + 1)
s = zeros(Nx + 1)

# Initial old solution
phi_old = ghost_point_time

@time begin

# Time Loop
for n = 1:Ntau

    global phi_old;

    # First iteration, second order correction at n+1
    phi_left = ghost_point_left[n + 1];
    phi_left_future = ghost_point_left[n + 2];

    for i = 2:1:Nx + 2

        # First order predictors
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + abs(c[i]) * (c[i] > 0) * phi_first_order[i - 1, n + 1] ) / ( 1 + abs(c[i]) );

        # Nutne pre 2. rad:
        phi1[i, n + 0] = ( phi_old[i] + abs(c[i]) * (c[i] > 0) * phi1[i - 1, n] ) / ( 1 + abs(c[i]) );
        phi1[i, n + 1] = ( phi1[i, n] + abs(c[i]) * (c[i] > 0) * phi1[i - 1, n + 1] ) / ( 1 + abs(c[i]) );
        phi1[i, n + 2] = ( phi1[i, n + 1] + abs(c[i]) * (c[i] > 0) * phi1[i - 1, n + 2] ) / ( 1 + abs(c[i]) );

        # First iteration, it should be already 2nd order accurate:
        phi2[i, n + 1] =  ( phi2[i, n] 
            - 1 / 2 * ( phi1[i, n + 1] - 2 * phi1[i, n] + phi_old[i] ) 
            + c[i] * ( phi2[i-1, n + 1] 
            - 1 / 2 * ( phi1[i, n + 1] - 2 * phi1[i - 1, n + 1] + phi_left) ) ) / (1 + c[i]);

        # Preparation for the 2nd iteration, second order correction at n+2  
        phi2[i, n + 2] =  ( phi2[i, n + 1] 
            - 1 / 2 * ( phi1[i, n + 2] - 2 * phi1[i, n + 1] + phi1[i, n] ) 
            + c[i] * ( phi2[i-1, n + 2] 
            - 1 / 2 * ( phi1[i, n + 2] - 2 * phi1[i - 1, n + 2] + phi_left_future) ) ) / (1 + c[i]);

        phi_left = phi1[i - 1, n + 1];
        phi_left_future = phi1[i - 1, n + 2];    
    end

    phi2[Nx + 2, n + 1] = 3*phi2[Nx + 1, n + 1] - 3*phi2[Nx + 0, n + 1] + phi2[Nx - 1, n + 1];
    phi2[Nx + 2, n + 2] = 3*phi2[Nx + 1, n + 2] - 3*phi2[Nx + 0, n + 2] + phi2[Nx - 1, n + 2];

    for k = 1:K # Multiple correction iterations
        # Second and the final iteration
        phi1[:, n] = phi2[:, n];
        phi1[:, n + 1] = phi2[:, n + 1];
        phi1[:, n + 2] = phi2[:, n + 2];

        phi_left = ghost_point_left[n + 1];

        for i = 2:1:Nx + 1

            ω[i] = ifelse( abs(phi1[i, n + 1] - 2 * phi1[i - 1, n + 1] + phi_left) <= abs(phi1[i + 1, n + 1] - 2*phi1[i, n + 1] + phi1[i - 1, n + 1]), 0, 1)
            α[i] = ifelse( abs(phi1[i, n + 1] - 2 * phi1[i, n] + phi_old[i]) <= abs( phi1[i, n + 2] - 2*phi1[i, n + 1] + phi1[i, n]), 0, 1)

            # U = ω0 * ( 1 / ( ϵ + (phi1[i, n + 1] - 2 * phi1[i - 1, n + 1] + phi_left)^2 )^2 );
            # D = ( 1 - ω0 ) * ( 1 / ( ϵ + (phi1[i + 1, n + 1] - 2*phi1[i, n + 1] + phi1[i - 1, n + 1])^2 )^2 );
            # ω[i] = U / ( U + D );

            # U = α0 * ( 1 / ( ϵ + (phi1[i, n + 1] - 2 * phi1[i, n] + phi_old[i])^2 )^2 );
            # D = ( 1 - α0 ) * ( 1 / ( ϵ + ( phi1[i, n + 2] - 2*phi1[i, n + 1] + phi1[i, n])^2 )^2 );
            # α[i] = U / ( U + D );

            phi2[i, n + 1] =  ( phi2[i, n] 
                - α[i] / 2 * ( phi1[i, n + 2] - 2*phi1[i, n + 1] + phi1[i, n] ) 
                - ( 1 - α[i] ) / 2 * ( phi1[i, n + 1] - 2 * phi1[i, n] + phi_old[i] ) 
                + c[i] * ( phi2[i - 1, n + 1] 
                - ( ω[i] / 2 )* (phi1[i + 1, n + 1] - 2*phi1[i, n + 1] + phi1[i - 1, n + 1]) 
            - ( (1 - ω[i]) / 2 )* ( phi1[i, n + 1] - 2 * phi1[i - 1, n + 1] + phi_left) ) ) / (1 + c[i]);

            phi2[i, n + 2] = ( phi2[i, n + 1] + abs(c[i]) * (c[i] > 0) * phi2[i - 1, n + 2] ) / ( 1 + abs(c[i]) );

            phi_left = phi2[i - 1, n + 1];
        end
        phi2[Nx + 2, n + 1] = 3*phi2[Nx + 1, n + 1] - 3*phi2[Nx + 0, n + 1] + phi2[Nx - 1, n + 1];

    end
    # Update old solution
    phi_old = phi2[:, n];
end

end

# Print error
Error_t_h = tau * h * sum(abs(phi2[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)
println("=============================")

# Compute TVD property of the derivative
phi_d = zeros(Nx + 1, Ntau + 1)
phi_dd = zeros(Nx + 1, Ntau + 1)
TVD = zeros(Ntau + 1)
for n = 1:Ntau + 1
    phi_d[:, n] = [ diff(phi2[1:Nx+1, n]) / h ; ( phi2[2, n] - phi2[end, n] ) / h ];
    phi_dd[:, n] = [ diff(phi_d[1:Nx+1, n]) / h ; ( phi_d[2, n] - phi_d[end, n] ) / h ]; 
    TVD[n] = sum(abs.(phi_dd[:,n]));
end

phi_d = zeros(Nx + 1, Ntau + 1)
phi_dd = zeros(Nx + 1, Ntau + 1)
TVD_1 = zeros(Ntau + 1)
for n = 1:Ntau + 1
    phi_d[:, n] = [ diff(phi_first_order[1:Nx+1, n]) / h ; ( phi_first_order[2, n] - phi_first_order[end, n] ) / h ];
    phi_dd[:, n] = [ diff(phi_d[1:Nx+1, n]) / h ; ( phi_d[2, n] - phi_d[end, n] ) / h ]; 
    TVD_1[n] = sum(abs.(phi_dd[:,n]));
end

CSV.write("phi.csv", DataFrame(phi2, :auto))

# Plot of the result at the final time together with the exact solution
# trace1 = scatter(x = x, y = phi_8[:,end], mode = "lines", name = "Compact TVD C = 8", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x[1:Nx+1], y = phi_exact.(x[1:Nx+1], Ntau*tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace1 = scatter(x = x, y = phi_0.(x), mode = "lines", name = "Initial Condition", line=attr(color="black", width=1, dash = "dash") )
# trace4 = scatter(x = x, y = phi_1[:,end], mode = "lines", name = "Compact TVD C = 1", line=attr(color="green", width=2))
# trace5 = scatter(x = x, y = phi_16[:,end], mode = "lines", name = "Compact TVD C = 1.6", line=attr(color="orange", width=2))
trace3 = scatter(x = x[1:Nx+1], y = phi2[1:Nx+1,end-1], mode = "lines", name = "Compact TVD", line=attr(color="firebrick", width=2))
trace4 = scatter(x = x[1:Nx+1], y = phi_first_order[1:Nx+1,end-1], mode = "lines", name = "First order", line=attr(color="green", width=2))


layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
# plot_phi = plot([ trace2, trace1,trace5, trace4, trace3], layout)
plot_phi = plot([ trace2, trace3, trace4], layout)

plot_phi

CSV.write("phi.csv", DataFrame(phi2,:auto))

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x[1:Nx+1], y = diff(phi2[1:Nx+1, end-1]) / h, mode = "lines", name = "Compact sol. gradient")
trace2_d = scatter(x = x[1:Nx+1], y = diff(phi_exact.(x[1:Nx+1], Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = diff(phi_first_order[:, end-1]) / h, mode = "lines", name = "First order sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace1_d, trace2_d], layout_d)

# Plot omega and alpha
trace_ω = scatter(x = x[1:Nx+1], y = ω[1:Nx+1], mode = "lines", name = "ω")
trace_α = scatter(x = x[1:Nx+1], y = α[1:Nx+1], mode = "lines", name = "α")
layout_ωα = Layout(title = "ω and α")
plot_ωα = plot([trace_ω, trace_α], layout_ωα)

p = [plot_phi; plod_d_phi; plot_ωα]
relayout!(p, width = 1000, height = 500)
p