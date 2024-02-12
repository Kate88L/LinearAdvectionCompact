# Inverted linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS

include("InitialFunctions.jl")
include("ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 2;

# Courant number
C = 0.5;

# Level of correction
p = 2;

# Grid settings
xL = -0.5 #- 1 * π / 2
xR = 2.5 #3 * π / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
# u(x) = 1 + 3/4 * cos(x)
u(x) = 1

# Initial condition
# phi_0(x) = cos(x);
phi_0(x) = piecewiseLinear(x);

# Exact solution
phi_exact(x, t) = phi_0(x - t);

## Comptutation
x = range(xL, xR, length = Nx + 1)

# Time
# T = 8 * π / sqrt(7)
T = 1
# tau = C * h / u
Ntau = 100 * 2^level
tau = T / Ntau
tau = C * h / maximum(u.(x))
Ntau = Int(round(T / tau))

c = zeros(Nx+1,1) .+ u.(x) * tau / h

# Grid initialization
phi = zeros(Nx + 1, Ntau + 1);
phi_predictor = zeros(Nx + 1, Ntau + 1); # predictor in time n+1
phi_first_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));

# Ghost point on the time -1
ghost_point_time = phi_exact.(x, -tau);

# WENO parameters
ω = zeros(Nx + 1, Ntau + 1);
ϵ = 1e-16;

# Space loop
for i = 2:1:Nx+1

    # Time loop
    for n = 1:Ntau

        if n > 1
            phi_old = phi[i, n - 1];
        else
            phi_old = ghost_point_time[i];
        end

        if n > Ntau - 1
            phi_future = phi_exact.(x[i - 1], (n+2) * tau);
        else
            phi_future = phi[i - 1, n + 2];
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c[i] * phi_first_order[i - 1, n + 1] ) / ( 1 + c[i] );

        # Predictor - first order
        phi_predictor[i, n + 1] = ( phi[i, n] + c[i] * phi_predictor[i - 1, n + 1] ) / ( 1 + c[i] );

        # Predictor - second order
        # phi_predictor[i, n + 1] = ( phi[i, n] + 0.5 / c[i] * ( c[i] - c[i-1] ) * phi[i, n] + c[i] * phi[i - 1, n + 1]
        # - 0.5 * ( ( 1 ) * (phi[i, n] - phi[i - 1, n + 1] + phi_future) ) ) / ( 1 + c[i] + 0.5 / c[i] * (c[i] - c[i-1]) - (1) / 2 );

        # Corrector
        for j = 1:p

            if j < 2
                phi_hat = phi_predictor[i, n + 1]
            else
                phi_hat = phi[i, n + 1]
            end
        
            r_downwind = phi[i, n] - phi[i - 1, n + 1] - phi_hat + phi_future;
            r_upwind = - phi[i - 1, n] + phi_old + phi[i - 1, n + 1] - phi[i, n];

            # WENO parameter 
            r = ( ϵ + r_upwind.^2 ) ./ ( ϵ + r_downwind.^2 );
            ω[i, n] = 1 / ( 1 + 2 * r.^2 );

            A = 1;
            if ( (r_downwind) * (r_upwind) < 0 ) 
                A = 0;
            end

            phi[i, n + 1] = ( phi[i, n] + 0.5 / c[i] * ( c[i] - c[i-1] ) * phi[i, n] + c[i] * phi[i - 1, n + 1]
            - 0.5 * A * ( ( ω[i, n] ) * r_upwind + ( 1 - ω[i, n] ) * r_downwind) ) / ( 1 + c[i] + 0.5 / c[i] * (c[i] - c[i-1]) );
        end

    end
end


# Print error
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)
println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)

# Print first order error
println("Error L2 first order: ", norm(phi_first_order[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf firts order: ", norm(phi_first_order[:, end] - phi_exact.(x, Ntau * tau), Inf)* h)

println("=============================")


# Plot of the result at the final time together with the exact solution
trace2 = scatter(x = x, y = phi_exact.(x, Ntau*tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace1 = scatter(x = x, y = phi_0.(x), mode = "lines", name = "Initial Condition", line=attr(color="black", width=1, dash = "dash") )
trace3 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Compact TVD", line=attr(color="firebrick", width=2))


layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([ trace1, trace2, trace3], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[:, end]) / h, mode = "lines", name = "Compact sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x, Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = diff(phi_predictor[:, end]) / h, mode = "lines", name = "First order sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace1_d, trace2_d, trace3_d], layout_d)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p