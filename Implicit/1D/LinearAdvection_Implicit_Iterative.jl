#  iterative 2nd order fully implicit scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames
using JSON

include("../../Utils/InitialFunctions.jl")
include("../../Utils/ExactSolutions.jl")
include("../../Utils/Utils.jl")

## Definition of basic parameters

third_order = true;

# Level of refinement
level = 0;

K = 1; # Number of iterations for the second order correction

# Courant number
C = 3;

# Grid settings
xL = - 1 * π / 2
xR =  π / 2
Nx = 40 * 2^level
h = (xR - xL) / Nx

# Velocity
# u(x) = 1 + 3/4 * cos(x)
u(x) = 2 + 3/2 * cos(x)
# u(x) = 1

# Initial condition
# phi_0(x) = asin( sin(x + π/2) ) * 2 / π;
phi_0(x) = cos.(x);
# phi_0(x) = 0. + (x - 4.)^2
# phi_0(x) = exp.(-10*(x+0)^2);
# phi_0(x) = piecewiseLinear(x);

# Exact solution
# phi_exact(x, t) = cosVelocityNonSmooth(x, t); 
phi_exact(x, t) = cosVelocitySmooth(x, t);
# phi_exact(x, t) = phi_0.(x - t * u.(x));             

## Comptutation

# Grid initialization with ghost point on the left and right
x = range(xL-h, xR+h, length = Nx + 3)

# Time
# T = 1 * π / sqrt(7) * 2 / 10
T = 8 * π / sqrt(7) * 0.5
# tau = C * h / u
Ntau = 1 * 2^level
# Ntau = Int(round(Nx / 15 / C + 1))
tau = T / Ntau
tau = C * h / maximum(u.(x))
Ntau = Int(round(T / tau))

# Ntau = 1

t = range(0, (Ntau + 2) * tau, length = Ntau + 3)

c = zeros(Nx+3) .+ u.(x) * tau / h

# predictor
phi1 = zeros(Nx + 3, Ntau + 3);
# corrector 
phi2 = zeros(Nx + 3, Ntau + 3);
# final solution
phi = zeros(Nx + 3, Ntau + 3);
# first order
phi_first_order = zeros(Nx + 3, Ntau + 3)

# Initial condition
phi1[:, 1] = phi_0.(x); # Ghost points
phi2[:, 1] = phi_0.(x);
phi[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);
phi1[:, 2] = phi_exact.(x, tau); # Initial condition
phi2[:, 2] = phi_exact.(x, tau);
phi[:, 2] = phi_exact.(x, tau);
phi_first_order[:, 2] = phi_exact.(x, tau);

# Boundary conditions
phi1[1, :] = phi_exact.(x[1], t);
phi2[1, :] = phi_exact.(x[1], t);
phi[1, :] = phi_exact.(x[1], t);
phi_first_order[1, :] = phi_exact.(x[1], t);
phi1[2, :] = phi_exact.(x[2], t);
phi2[2, :] = phi_exact.(x[2], t);
phi[2, :] = phi_exact.(x[2], t);
phi_first_order[2, :] = phi_exact.(x[2], t);

# phi_i_predictor = phi1[:, 2];

# WENO parameters
ϵ = 1e-16;
ω0 = 0;
α0 = 0;

ω1 = zeros(Nx + 3) .+ ω0;
ω2 = zeros(Nx + 3) .+ ω0;
α1 = zeros(Nx + 3) .+ α0;
α2 = zeros(Nx + 3) .+ α0;

l = zeros(Nx + 3)
s = zeros(Nx + 3)

@time begin

# Precompute predictors for the initial time
for i = 3:1:Nx + 2
    phi1[i, 2] = ( phi[i, 1] + c[i] * phi[i - 1, 2] ) / ( 1 + c[i] );
    phi1[i - 1, 3] = ( phi[i - 1, 2] + c[i - 1] * phi2[i - 2, 3] ) / ( 1 + c[i - 1] );

    phi2[i, 3] = ( phi[i, 2] - 1/2 * ( -phi1[i, 2] + phi[i , 1] ) + c[i] * ( phi2[i - 1, 3] - 1/2 * ( -phi1[i - 1, 3] + phi2[i - 2, 3] ) ) ) / ( 1 + c[i] );
end

# Time Loop
for n = 2:Ntau + 1

    phi1[3, n] = ( phi[3, n - 1] + c[3] * phi[2, n] ) / ( 1 + c[3] );

    global phi_i_predictor = ( phi[3, n] - 1/2 * ( -phi1[3, n] + phi[3 , n - 1] ) + c[3] * ( phi2[2, n + 1] - 1/2 * ( -phi1[2, n + 1] + phi[1, n + 1] ) ) ) / ( 1 + c[3] );
    # global phi_i_predictor = phi2[3, n + 1];

    for i = 3:1:Nx + 2

        if (third_order && u(x[i]) < 0) 
            continue
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c[i] * phi_first_order[i - 1, n + 1] ) / ( 1 + c[i] );

        phi2_i_old = phi2[i, n + 1];
        global phi2_i_old_p = phi_i_predictor;

        # FIRST ITERATION 
        phi1[i, n] = ( phi[i, n - 1] + c[i] * phi[i - 1, n] ) / ( 1 + c[i] );
        phi1[i - 1, n + 1] = ( phi[i - 1, n] + c[i - 1] * phi[i - 2, n + 1] ) / ( 1 + c[i - 1] );
        phi1[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] ) / ( 1 + c[i] );

        phi[i, n + 1] =  ( phi[i, n] 
            - 1 / 2 * ( phi1[i, n + 1] - phi[i, n] - phi1[i, n] + phi[i, n - 1] ) 
            + c[i] * ( phi[i - 1, n + 1] 
            - 1 / 2 * ( phi1[i, n + 1] - phi[i - 1, n + 1] - phi1[i - 1, n + 1] + phi[i - 2, n + 1]) ) ) / (1 + c[i]);
            
        phi2[i, n + 1] = phi[i, n + 1];

        # Compute second order predictor for i + 1
        phi1[i + 1, n] = ( phi[i + 1, n - 1] + c[i + 1] * phi[i, n] ) / ( 1 + c[i + 1] );
        phi1[i + 1, n + 1] = ( phi[i + 1, n] + c[i + 1] * phi[i, n + 1] ) / ( 1 + c[i + 1] );

        global phi_i_predictor = ( phi[i + 1, n] 
            - 1 / 2 * ( phi1[i + 1, n + 1] - phi[i + 1, n] - phi1[i + 1, n] + phi[i + 1, n - 1] ) 
            + c[i + 1] * ( phi2[i, n + 1] 
            - 1 / 2 * ( phi1[i + 1, n + 1] - phi2[i, n + 1] - phi1[i, n + 1] + phi[i - 1, n + 1]) ) ) / (1 + c[i + 1]);

        # Compute second order predictor for n + 2
        phi1[i - 1, n + 2] = ( phi[i - 1, n + 1] + c[i - 1] * phi2[i - 2, n + 2] ) / ( 1 + c[i - 1] );
        phi1[i, n + 2] = ( phi2[i, n + 1] + c[i] * phi2[i - 1, n + 2] ) / ( 1 + c[i] );

        phi2[i, n + 2] = ( phi2[i, n + 1] 
            - 1 / 2 * ( phi1[i, n + 2] - phi2[i, n + 1] - phi1[i, n + 1] + phi[i, n] ) 
            + c[i] * ( phi2[i - 1, n + 2] 
            - 1 / 2 * ( phi1[i, n + 2] - phi2[i - 1, n + 2] - phi1[i - 1, n + 2] + phi2[i - 2, n + 2]) ) ) / (1 + c[i]);
        
        for k = 1:K # Multiple correction iterations

            # SECOND ITERATION
            rd_n = phi2[i, n + 2] - phi2[i, n + 1] - phi2_i_old + phi[i, n];
            ru_n = phi2[i, n + 1] - phi[i, n] - phi2[i, n] + phi[i, n - 1] 

            rd_i = phi_i_predictor - phi2[i, n + 1] - phi2_i_old_p + phi[i - 1, n + 1];
            ru_i = phi2[i, n + 1] - phi[i - 1, n + 1] - phi2[i - 1, n + 1] + phi[i - 2, n + 1];

            ω1[i] = ifelse( abs(ru_i) <= abs(rd_i), 1, 0) * Int(!third_order) * ifelse( ru_i * rd_i > 0, 1, 0)
            α1[i] = ifelse( abs(ru_n) <= abs(rd_n), 1, 0) * Int(!third_order) * ifelse( ru_n * rd_n > 0, 1, 0)

            ω2[i] = ( 1 - ω1[i] )
            α2[i] = ( 1 - α1[i] )     

            phi[i, n + 1] =  ( phi[i, n] 
                - α1[i] / 2 * ru_n - α2[i] / 2 * rd_n
                + c[i] * ( phi[i - 1, n + 1] - ω1[i] / 2 *  ru_i - ω2[i] / 2 * rd_i ) ) / (1 + c[i]);

            phi2_i_old = phi2[i, n + 1];
            phi2_i_old_p = phi2[i, n + 1];
            phi2[i, n + 1] = phi[i, n + 1];
        end
    end
    phi[Nx + 3, n + 1] = 3*phi[Nx + 2, n + 1] - 3*phi[Nx + 1, n + 1] + phi[Nx, n + 1];
    phi1[Nx + 3, n + 1] = 3*phi1[Nx + 2, n + 1] - 3*phi1[Nx + 1, n + 1] + phi1[Nx, n + 1];
    phi_first_order[Nx + 3, n + 1] = 3*phi_first_order[Nx + 2, n + 1] - 3*phi_first_order[Nx + 1, n + 1] + phi_first_order[Nx, n + 1];
    phi2[Nx + 3, n + 1] = 3*phi2[Nx + 2, n + 1] - 3*phi2[Nx + 1, n + 1] + phi2[Nx, n + 1];

    for i = Nx + 2:-1:3

        if (third_order && u(x[i]) > 0) 
            continue
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + c[i] * phi_first_order[i - 1, n + 1] ) / ( 1 + c[i] );

        phi2_i_old = phi2[i, n + 1];
        global phi2_i_old_p = phi2[i, n + 1];

        # FIRST ITERATION 
        phi1[i, n] = ( phi[i, n - 1] + c[i] * phi[i - 1, n] ) / ( 1 + c[i] );
        phi1[i - 1, n + 1] = ( phi[i - 1, n] + c[i - 1] * phi[i - 2, n + 1] ) / ( 1 + c[i - 1] );
        phi1[i, n + 1] = ( phi[i, n] + c[i] * phi[i - 1, n + 1] ) / ( 1 + c[i] );

        phi[i, n + 1] =  ( phi[i, n] 
            - 1 / 2 * ( phi1[i, n + 1] - phi[i, n] - phi1[i, n] + phi[i, n - 1] ) 
            + c[i] * ( phi[i - 1, n + 1] 
            - 1 / 2 * ( phi1[i, n + 1] - phi[i - 1, n + 1] - phi1[i - 1, n + 1] + phi[i - 2, n + 1]) ) ) / (1 + c[i]);
            
        phi2[i, n + 1] = phi[i, n + 1];

        # Compute second order predictor for i + 1
        phi1[i + 1, n] = ( phi[i + 1, n - 1] + c[i + 1] * phi[i, n] ) / ( 1 + c[i + 1] );
        phi1[i + 1, n + 1] = ( phi[i + 1, n] + c[i + 1] * phi[i, n + 1] ) / ( 1 + c[i + 1] );

        global phi_i_predictor = ( phi[i + 1, n] 
            - 1 / 2 * ( phi1[i + 1, n + 1] - phi[i + 1, n] - phi1[i + 1, n] + phi[i + 1, n - 1] ) 
            + c[i + 1] * ( phi2[i, n + 1] 
            - 1 / 2 * ( phi1[i + 1, n + 1] - phi2[i, n + 1] - phi1[i, n + 1] + phi[i - 1, n + 1]) ) ) / (1 + c[i + 1]);

        phi1[i - 1, n] = ( phi[i - 1, n - 1] + c[i - 1] * phi[i - 2, n] ) / ( 1 + c[i - 1] );
        phi1[i - 1, n + 1] = ( phi[i - 1, n] + c[i - 1] * phi[i - 2, n + 1] ) / ( 1 + c[i - 1] );

        # global phi_i_predictor = ( phi[i - 1, n] 
            # - 1 / 2 * ( phi1[i - 1, n + 1] - phi[i - 1, n] - phi1[i - 1, n] + phi[i - 1, n - 1] ) 
            # + c[i - 1] * ( phi2[i - 2, n + 1] 
            # - 1 / 2 * ( phi1[i, n + 1] - phi2[i - 1, n + 1] - phi1[i - 1, n + 1] + phi[i - 2, n + 1]) ) ) / (1 + c[i - 1]);

        # Compute second order predictor for n + 2
        phi1[i - 1, n + 2] = ( phi[i - 1, n + 1] + c[i - 1] * phi2[i - 2, n + 2] ) / ( 1 + c[i - 1] );
        phi1[i, n + 2] = ( phi2[i, n + 1] + c[i] * phi2[i - 1, n + 2] ) / ( 1 + c[i] );

        phi2[i, n + 2] = ( phi2[i, n + 1] 
            - 1 / 2 * ( phi1[i, n + 2] - phi2[i, n + 1] - phi1[i, n + 1] + phi[i, n] ) 
            + c[i] * ( phi2[i - 1, n + 2] 
            - 1 / 2 * ( phi1[i, n + 2] - phi2[i - 1, n + 2] - phi1[i - 1, n + 2] + phi2[i - 2, n + 2]) ) ) / (1 + c[i]);
        
        for k = 1:K # Multiple correction iterations

            # SECOND ITERATION
            rd_n = phi2[i, n + 2] - phi2[i, n + 1] - phi2_i_old + phi[i, n];
            ru_n = phi2[i, n + 1] - phi[i, n] - phi2[i, n] + phi[i, n - 1] 

            rd_i = phi_i_predictor - phi2[i, n + 1] - phi2_i_old_p + phi[i + 1, n + 1];
            ru_i = phi2[i, n + 1] - phi[i - 1, n + 1] - phi2[i - 1, n + 1] + phi[i - 2, n + 1];

            ω1[i] = ifelse( abs(ru_i) <= abs(rd_i), 1, 0) * Int(!third_order) * ifelse( ru_i * rd_i > 0, 1, 0)
            α1[i] = ifelse( abs(ru_n) <= abs(rd_n), 1, 0) * Int(!third_order) * ifelse( ru_n * rd_n > 0, 1, 0)

            ω2[i] = ( 1 - ω1[i] )
            α2[i] = ( 1 - α1[i] )   

            phi[i, n + 1] =  ( phi[i, n] 
                - α1[i] / 2 * ru_n - α2[i] / 2 * rd_n
                + c[i] * ( phi[i - 1, n + 1] - ω1[i] / 2 *  ru_i - ω2[i] / 2 * rd_i ) ) / (1 + c[i]);

            phi2_i_old = phi2[i, n + 1];
            phi2_i_old_p = phi2[i, n + 1];
            phi2[i, n + 1] = phi[i, n + 1];
        end
    end
    # phi[Nx + 3, n + 1] = 3*phi[Nx + 2, n + 1] - 3*phi[Nx + 1, n + 1] + phi[Nx, n + 1];
    # phi1[Nx + 3, n + 1] = 3*phi1[Nx + 2, n + 1] - 3*phi1[Nx + 1, n + 1] + phi1[Nx, n + 1];
    # phi_first_order[Nx + 3, n + 1] = 3*phi_first_order[Nx + 2, n + 1] - 3*phi_first_order[Nx + 1, n + 1] + phi_first_order[Nx, n + 1];
    # phi2[Nx + 3, n + 1] = 3*phi2[Nx + 2, n + 1] - 3*phi2[Nx + 1, n + 1] + phi2[Nx, n + 1];
end

end

# Print error
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], t[n])) for n in 2:Ntau+2 for i in 2:Nx+2)
println("Error t*h: ", Error_t_h)
Error_t_h_1 = tau * h * sum(abs(phi_first_order[i, n] - phi_exact.(x[i], t[n])) for n in 2:Ntau+2 for i in 2:Nx+2)
println("Error t*h first order: ", Error_t_h_1)

# Error_t_h = h * sum(abs(phi[i, end-1] - phi_exact.(x[i], t[end-1])) for i in 2:Nx+2)
# println("Error h: ", Error_t_h)

# Print maximum and minimum of the derivative
println("Max derivative: ", maximum(diff(phi[:, end-1]) / h))
println("Min derivative: ", minimum(diff(phi[:, end-1]) / h))

# Load the last error
last_error = load_last_error()
if last_error != nothing
    println("Order: ", log(2, last_error / Error_t_h))
end 
# Save the last error
save_last_error(Error_t_h)
println("=============================")

CSV.write("phi.csv", DataFrame(phi2, :auto))

# Plot of the result at the final time together with the exact solution
trace2 = scatter(x = x, y = phi_exact.(x, t[end-1]), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace1 = scatter(x = x, y = phi_0.(x), mode = "lines", name = "Initial Condition", line=attr(color="black", width=1, dash = "dash") )
trace3 = scatter(x = x, y = phi[:,end-1], mode = "lines", name = "Compact TVD", line=attr(color="firebrick", width=2))
trace4 = scatter(x = x, y = phi_first_order[:,end-1], mode = "lines", name = "First order", line=attr(color="green", width=2))

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
# plot_phi = plot([ trace2, trace1,trace5, trace4, trace3], layout)
plot_phi = plot([ trace2, trace3, trace4], layout)

plot_phi

CSV.write("phi.csv", DataFrame(phi2,:auto))

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[:, end-1]) / h, mode = "lines", name = "Compact sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x, t[end-1])) / h, mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = diff(phi_first_order[:, end-1]) / h, mode = "lines", name = "First order sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace3_d, trace2_d, trace1_d], layout_d)

# Plot omega and alpha
# trace_ω = scatter(x = x, y = ω, mode = "lines", name = "ω")
# trace_α = scatter(x = x, y = α, mode = "lines", name = "α")
# layout_ωα = Layout(title = "ω and α")
# plot_ωα = plot([trace_ω, trace_α], layout_ωα)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p