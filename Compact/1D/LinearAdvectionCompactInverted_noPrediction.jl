# Inverted linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS
using CSV
using DataFrames
using Statistics
using ForwardDiff

include("../../Utils/InitialFunctions.jl")
include("../../Utils/ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 6
# Courant number
C = 3;

# Level of correction
p = 1;

# Grid settings
xL = -1 # -1 * π / 2
xR = 1 # 3 * π / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
# u(x) = 1 + 3/4 * cos(x)
u(x) = 1

# Initial condition
# phi_0(x) = cos(x);
# phi_0(x) = piecewiseLinear(x);
phi_0(x) = makePeriodic(nonSmooth,-1,1)(x - 0.5);

# Exact solution
phi_exact(x, t) = phi_0(x - t);

# Exact solution derivative
phi_exact_derivative(x, t) = phi_derivative_x(phi_exact, x, t);

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
ω = zeros(Ntau + 1);
ω0 = 1/3;
ϵ = 1e-16;

s = zeros(Nx + 1, Ntau + 1)

# Space loop
for i = 2:1:Nx+1

    # Time loop
    for n = 1:Ntau

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c[i] * phi_first_order[i - 1, n + 1] ) / ( 1 + c[i] );

        # Predictor 
        phi_predictor[i, n + 1] = ( phi[i, n] + c[i] * phi_predictor[i - 1, n + 1] ) / ( 1 + c[i] );


        # Corrector
        for j = 1:p

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

            if j < 2
                r_downwind = phi_predictor[i, n] - phi[i - 1, n + 1] - phi_predictor[i, n + 1] + phi_future;
            else
                r_downwind = phi[i, n] - phi[i - 1, n + 1] - phi[i, n + 1] + phi_future;
            end
        
            r_upwind = - phi[i - 1, n] + phi_old + phi[i - 1, n + 1] - phi[i, n];

            # WENO SHU
            U = ω0 * ( 1 / ( ϵ + r_upwind^2 )^2 );
            D = ( 1 - ω0 ) * ( 1 / ( ϵ + r_downwind^2 )^2 );
            ω[n] = U / ( U + D );

            # Space - Time limiter
            # ω[n] = (2 + ( 1 / c[i] )) / 6;
            # r = ( r_upwind + ϵ ) / ( r_downwind + ϵ )
            # local s[i, n+1] = 1 - ω[i] + ω[i] * r;
            # s[i, n+1] = maximum([-1, minimum([s[i, n+1], 2])])
            # s[i, n+1] = maximum([-1, minimum([s[i, n+1], r * (2 / abs(1/c[i]) + s[i-1, n+1])])])

            # if (abs(r_downwind) + ϵ) <= (abs(r_upwind) + ϵ)
            #     ω[n]  = 0;
            # else
            #     ω[n]  = 1;
            # end

            # WENO / ENO no predictors
            phi[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] - 0.5 * ( 1 - ω[n] ) * ( phi[i, n] - phi[i - 1, n + 1] + phi_future ) 
                                                - 0.5 * ω[n] * r_upwind ) / ( 1 + c[i] - 0.5 * ( 1 - ω[n] ) );

            # Space-Time correction no predictors
            # phi[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] - 0.5 *  s[i, n + 1] * ( phi[i, n] - phi[i - 1, n + 1] + phi_future ) ) / ( 1 + c[i] - 0.5 * s[i, n + 1] );
        end

    end
end

# CSV.write("phi.csv", DataFrame(phi, :auto))

# df = CSV.File("phi_normal.csv") |> DataFrame
# phi_1 = Matrix(df)

# Compute TVD property of the derivative
phi_d = zeros(Nx + 1, Ntau + 1)
phi_dd = zeros(Nx + 1, Ntau + 1)
TVD = zeros(Ntau + 1)
for n = 1:Ntau + 1
    phi_d[:, n] = [ ( phi[2, n] - phi[1,n] ) / h; [ ( phi[i+1,n] - phi[i-1,n] ) / (2*h) for i in 2:Nx ]  ;( phi[end, n] - phi[end-1, n] ) / h ];
    phi_dd[:, n] = [ diff(phi_d[:, n]) / h ; ( phi_d[2, n] - phi_d[end, n] ) / h ]; 
    TVD[n] = sum(abs.(phi_dd[:,n]));
end

# Print error
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)
println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
# println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) )
println("Error L_inf: ", maximum(abs(phi[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1) )


# Print first order error
println("Error L2 first order: ", norm(phi_first_order[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf firts order: ", norm(phi_first_order[:, end] - phi_exact.(x, Ntau * tau), Inf))

matrixM = zeros(Nx + 1, 1)
matrixM[:] = abs.(phi_d[:, end] - phi_exact_derivative.(x, Ntau*tau));
for i = 1:Nx + 1
    if isnan(matrixM[i])
        matrixM[i] = 0;
    end
    if (abs(x[i] - 0.5) < (2 * 0.02) || 
       abs(x[i] + 0.5) < (2 * 0.02) ||
       abs(x[i] + 0.83) < (2 * 0.02) ||
       abs(x[i] + 0.16) < (2 * 0.02) )
        matrixM[i] = 0;
    end
end

# Compute T*h error for the solution derivative
Error_t_h = tau * h * sum(abs(phi_d[i, n] - phi_exact_derivative(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1 if !isnan(phi_d[i, n]) && !isnan(phi_exact_derivative(x[i], (n-1)*tau)))
MaxError = maximum(matrixM)
max_indexes = argmax(matrixM)
println("Error t*h derivative: ", Error_t_h)    
println("Error L_inf derivative: ", MaxError)

testPhi = phi_d[:, end]
testPhiExact = phi_exact_derivative.(x, Ntau * tau)
test = abs.(testPhi - testPhiExact)

println("=============================")

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Inverted scheme", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
# trace3 = scatter(x = x, y = phi_1[:, end], mode = "lines", name = "Classical scheme", line=attr(color="royalblue", width=2))


layout = Layout(title = "Linear advection equation", xaxis_title = "x")
plot_phi = plot([trace1, trace2], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = phi_d[:, end], mode = "lines", name = "Inverted sol. gradient")
trace2_d = scatter(x = x, y = phi_exact_derivative.(x, Ntau * tau), mode = "lines", name = "Exact sol. gradient")
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