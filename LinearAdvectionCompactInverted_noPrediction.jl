# Inverted linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS
using CSV
using DataFrames

include("InitialFunctions.jl")
include("ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 4;

# Courant number
C = 8;

# Level of correction
p = 2;

# Level of predictor accuracy
pA = 2;

# Grid settings
xL = -0.5  #- 1 * π / 2
xR = 2.5  #3 * π / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
# u(x) = 1 + 3/4 * cos(x)
u(x) = 1

# Initial condition
# phi_0(x) = cos(x);
phi_0(x) = piecewiseLinear(x);
# phi_0(x) = makePeriodic(nonSmooth,-1,1)(x - 0.5);

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
# Ntau = 1

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
ω0 = 1/3;
ω0 = (C*(2*C+3)+1)/(6*C*(C+1));
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
            phi_future = phi_exact.(x[i - 1], (n+1) * tau);
        else
            phi_future = phi[i - 1, n + 2];
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c[i] * phi_first_order[i - 1, n + 1] ) / ( 1 + c[i] );

        # Predictor 
        if ( pA < 2 )
            # Predictor - first order
            phi_predictor[i, n + 1] = ( phi[i, n] + c[i] * phi_predictor[i - 1, n + 1] ) / ( 1 + c[i] );
        else
            # Predictor - second order
            phi_predictor[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] 
                                - 0.5 * ( ( 1 - ω0 ) * ( phi[i, n] - phi[i - 1, n + 1] + phi_future ) + ω0 * ( -phi[i - 1, n] + phi_old + phi[i - 1, n + 1] - phi[i, n] ) ) ) / ( 1 + c[i] - 0.5 * ( 1 - ω0 ) );
        end

        # Corrector
        for j = 1:p

            if j < 2
                phi_hat = phi_predictor[i, n + 1]
            else
                phi_hat = phi[i, n + 1]
            end
        
            r_downwind = phi[i, n] - phi[i - 1, n + 1] - phi_hat + phi_future;
            r_upwind = - phi[i - 1, n] + phi_old + phi[i - 1, n + 1] - phi[i, n];

            # WENO SHU
            U = ω0 * ( 1 / ( ϵ + r_upwind )^2 );
            D = ( 1 - ω0 ) * ( 1 / ( ϵ + r_downwind )^2 );
            ω[i, n] = U / ( U + D );
            r = ( r_upwind + ϵ ) / ( r_downwind + ϵ )
            local s = 1 - ω[i, n] + ω[i, n] * r;

            # if (abs(r_downwind) + ϵ) < (abs(r_upwind) + ϵ)
            #     ω[i, n] = 0;
            # else
            #     ω[i, n] = 1;
            # end

            # abs(r_upwind) < ϵ ? A = 0 : A = 1;
            (r_downwind) * (r_upwind) < 0 ? A = 0 : A = 1;

            # ENO version
            # phi[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1]
            # - 0.5 * A * ( ( ω[i, n] ) * r_upwind + ( 1 - ω[i, n] ) * r_downwind) ) / ( 1 + c[i] );

            # WENO shu version
            phi[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] - 0.5 * A * s * ( r_downwind + ϵ ) ) / ( 1 + c[i] );

            # Compute the solution with no predictor
            # phi[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] - 0.5 * A * ( ( ω[i, n] ) * r_upwind + ( 1 - ω[i, n] ) * ( phi[i, n] - phi[i - 1, n + 1] + phi_future ) ) ) / ( 1 + c[i] - 0.5 * A * (1 - ω[i, n]) );
        end

    end
end

CSV.write("phi.csv", DataFrame(phi, :auto))

# df = CSV.File("phi_normal.csv") |> DataFrame
# phi_1 = Matrix(df)

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
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Inverted scheme", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
# trace3 = scatter(x = x, y = phi_1[:, end], mode = "lines", name = "Classical scheme", line=attr(color="royalblue", width=2))


layout = Layout(title = "Linear advection equation", xaxis_title = "x")
plot_phi = plot([trace1, trace2], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[:, end]) / h, mode = "lines", name = "Inverted sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x, Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
# trace3_d = scatter(x = x, y = diff(phi_1[:, end]) / h, mode = "lines", name = "classical sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x")

plod_d_phi = plot([trace1_d, trace2_d], layout_d)

# Plot ω values in the last time step
# trace_ω = scatter(x = x, y = ω[:, end-1], mode = "lines", name = "ω", line=attr(color="firebrick", width=2))

# layout_ω = Layout(title = "Linear advection equation - WENO parameter", xaxis_title = "x", yaxis_title = "ω")
# plot_ω = plot([trace_ω], layout_ω)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p