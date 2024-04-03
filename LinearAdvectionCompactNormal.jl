# Normal linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS
using CSV
using DataFrames

include("Utils\\InitialFunctions.jl")
include("Utils\\ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 3;

# Courant number
C = 8

# Grid settings
xL = - 1 * π / 2
xR = 3 * π / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
u(x) = 1 + 3/4 * cos(x)
# u(x) = 1

# Initial condition
phi_0(x) = asin( sin(x + π/2) ) * 2 / π;
# phi_0(x) = cos.(x);
# phi_0(x) = makePeriodic(nonSmooth,-1,1)(x - 0.5);
# phi_0(x) = piecewiseLinear(x);
# phi_0(x) = exp(-x.^2 *1000)

# Exact solution
phi_exact(x, t) = cosVelocityNonSmooth(x, t);
# phi_exact(x, t) = phi_0.(x - t);

# Grid initialization
x = range(xL, xR, length = Nx + 1)

# Time settings
T = 8 * π / sqrt(7)
# T = 1
Ntau = 100 * 2^level
tau = T / Ntau
tau = C * h / maximum(u.(x))
Ntau = Int(round(T / tau))
# Ntau = 1

c = zeros(Nx+1,1) .+ u.(x) * tau / h

## Comptutation
phi = zeros(Nx + 1, Ntau + 1);
phi_predictor = zeros(Nx + 1, Ntau + 1);
phi_first_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));

phi[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));

# Ghost point on the left side 
ghost_point_left = phi_exact.(xL - h, range(tau, (Ntau+1) * tau, length = Ntau + 1));

# ENO parameters
s = zeros(Nx + 1, Ntau + 1);
eps = 1e-8;

# Time loop
for n = 1:Ntau
    # Space loop
    for  i = 2:1:Nx + 1

        if i > 2
            phi_left = phi[i - 2, n + 1];
        else
            phi_left = ghost_point_left[n];
        end

        if i < Nx + 1
            phi_right = phi_predictor[i + 1, n];
        else
            phi_right = phi_exact(xR + h, (n-1) * tau) ;
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c[i] *  phi_first_order[i - 1, n + 1] ) / ( 1 + c[i] );
        
        # Predictor
        phi_predictor[i, n + 1] = ( phi[i, n] + c[i] *  phi[i - 1, n + 1] ) / ( 1 + c[i] );

        # Corrector
        r_downwind_i_minus = - phi_predictor[i, n] + phi_predictor[i - 1, n + 1];
        r_upwind_i_minus = - phi[i - 1, n] + phi_left;

        r_downwind_i = - phi_predictor[i, n + 1] + phi_right;
        r_upwind_i = phi[i, n] - phi[i - 1, n + 1];

        # ENO parameter
        if abs(r_downwind_i_minus + r_downwind_i) <= abs(r_upwind_i_minus + r_upwind_i)
            s[i,n+1] = 1
        else
            s[i,n+1] = 0
        end

        # Second order solution
        phi[i, n + 1] = ( phi[i, n] + c[i] * ( phi[i - 1, n + 1] - 0.5 * s[i, n + 1] * (r_downwind_i_minus + r_downwind_i) - 0.5 * (1-s[i, n + 1]) * (r_upwind_i + r_upwind_i_minus ) ) ) / ( 1 + c[i] );
    end

end

Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)

# Compute TVD property of the derivative
phi_d = zeros(Nx + 1, Ntau + 1)
phi_dd = zeros(Nx + 1, Ntau + 1)
TVD = zeros(Ntau + 1)
for n = 1:Ntau + 1
    phi_d[:, n] = [ diff(phi[:, n]) / h ; ( phi[2, n] - phi[end, n] ) / h ];
    phi_dd[:, n] = [ diff(phi_d[:, n]) / h ; ( phi_d[2, n] - phi_d[end, n] ) / h ]; 
    TVD[n] = sum(abs.(phi_dd[:,n]));
end

phi_d = zeros(Nx + 1, Ntau + 1)
phi_dd = zeros(Nx + 1
, Ntau + 1)
TVD_1 = zeros(Ntau + 1)
for n = 1:Ntau + 1
    phi_d[:, n] = [ diff(phi_first_order[:, n]) / h ; ( phi_first_order[2, n] - phi_first_order[end, n] ) / h ];
    phi_dd[:, n] = [ diff(phi_d[:, n]) / h ; ( phi_d[2, n] - phi_d[end, n] ) / h ]; 
    TVD_1[n] = sum(abs.(phi_dd[:,n]));
end


df = CSV.File("phi.csv") |> DataFrame
phi_1 = Matrix(df)

println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)

# Print first order error
println("Error L2 first order: ", norm(phi_first_order[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf firts order: ", norm(phi_first_order[:, end] - phi_exact.(x, Ntau * tau), Inf)* h)

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Normal scheme", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace3 = scatter(x = x, y = phi_1[:, end], mode = "lines", name = "Inverted scheme", line=attr(color="royalblue", width=2))


layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([trace1, trace2, trace3], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[:, end]) / h, mode = "lines", name = "Normal sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x, Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = diff(phi_1[:, end]) / h, mode = "lines", name = "Inverted sol. gradient")

layout_d = Layout(plot_bgcolor="white", 
xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))

plod_d_phi = plot([trace1_d, trace2_d, trace3_d], layout_d)

# Plot TVD
trace_tvd = scatter(x = range(0, Ntau, length = Ntau + 1), y = TVD, mode = "lines", name = "TVD", line=attr(color="firebrick", width=2))
trace_tvd_1 = scatter(x = range(0, Ntau, length = Ntau + 1), y = TVD_1, mode = "lines", name = "TVD first order", line=attr(color="royalblue", width=2))
layout_tvd = Layout(title = "TVD")
plot_tvd = plot([trace_tvd; trace_tvd_1], layout_tvd)


p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p