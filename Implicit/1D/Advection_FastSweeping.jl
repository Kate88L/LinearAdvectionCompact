# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames
using JSON
using DataStructures

include("../../Utils/InitialFunctions.jl")
include("../../Utils/ExactSolutions.jl")
include("../../Utils/Utils.jl")

## Definition of basic parameters

# Level of refinement
level = 6;

# Courant number
C = 5;

# Grid settings
xL = - 1 * π / 2
xR = 3 * π / 2
Nx = 40 * 2^level
h = (xR - xL) / Nx

# Constant velocity example
# u(x) = 1
# phi_0(x) = cos.(x);
# phi_exact(x, t) = phi_0.(x - t);    

# Variable velocity example
offset = 0.5
u(x) = sin.(x - offset)
phi_0(x) = sin.(x - offset);
phi_exact(x, t) = sin.( 2 * atan.( exp.(-t) .* tan.( (x-offset) ./ 2) ) )

## Comptutation

# Grid initialization
x = range(xL - h, xR + h, length = Nx + 3)

# Time
T = 1.2
tau = C * h / maximum(u.(x))
Ntau = Int(round(T / tau))

# Ntau = 1

t = range(0, (Ntau + 2) * tau, length = Ntau + 3)


c = zeros(Nx+3) .+ u.(x) * tau / h
cp = zeros(Nx+3)
cm = zeros(Nx+3)

phi = zeros(Nx + 3, Ntau + 3);
phi1 = zeros(Nx + 3, Ntau + 3);
phi2 = zeros(Nx + 3, Ntau + 3);
phi_k = zeros(Nx + 3, Ntau + 3); # Sweep solution
phi_first_order = zeros(Nx + 3, Ntau + 3);

# Initial condition
phi[:, 1] = phi_0.(x);
phi[:, 2] = phi_exact.(x, tau);

# Boundary conditions right side - inflow
phi[end-1, :] = phi_exact.(x[end-1], t);
phi[end, :] = phi_exact.(x[end], t);

phi1 = copy(phi)
phi2 = copy(phi)
phi_first_order = copy(phi)
phi_k = copy(phi)

# ENO parameters
ω = zeros(Nx + 1);
α = zeros(Nx + 1);

# Fast sweeping
sweep_cases = Dict(1 => (3:1:Nx+1), 2 => (Nx+1:-1:3))
ii = 0

@time begin

# Precompute the predictors for the initial time 
# phi1[3:end-2, 3] = phi[3:end-2, 1] # initialize with previous time step
# phi2[3:end-2, 3] = phi[3:end-2, 2] # initialize with previous time step
# for sweep = 1:2
#     sw = sweep_cases[sweep]
#     # Set c plus and c minus
#     global cp = ifelse(sweep == 1, max.(c, 0), zeros(Nx+3))
#     global cm = ifelse(sweep == 2, min.(c, 0), zeros(Nx+3))
#     for i = sw
#         phi1[i, 2] = ( phi1[i, 2] + cp[i] * phi[i - 1, 2] - cm[i] * phi[i + 1, 2] ) / ( 1 + cp[i] - cm[i] );
#         phi1[i - 1, 3] = ( phi[i - 1, 2] + cp[i - 1] * phi2[i - 2, 3] ) / ( 1 + cp[i - 1] );
#         phi1[i + 1, 3] = ( phi[i + 1, 2] - cm[i + 1] * phi2[i + 2, 3] ) / ( 1 - cm[i + 1] );
    
#         phi2[i, 3] = ( phi2[i, 3] - 1/2 * ( -phi1[i, 2] + phi[i , 1] ) 
#             + cp[i] * ( phi2[i - 1, 3] - 1/2 * ( -phi1[i - 1, 3] + phi2[i - 2, 3] ) )
#             - cm[i] * ( phi2[i + 1, 3] - 1/2 * ( -phi1[i + 1, 3] + phi2[i + 2, 3] ) ) ) / ( 1 + cp[i] - cm[i] );
#     end
#     # Outlfow boundary condition on the left side
#     phi1[2, 3] = 3 * phi1[3, 3] - 3 * phi1[4, 3] + phi1[5, 3]
#     phi1[1, 3] = 3 * phi1[2, 3] - 3 * phi1[3, 3] + phi1[4, 3]
#     phi2[2, 3] = 3 * phi2[3, 3] - 3 * phi2[4, 3] + phi2[5, 3]
#     phi2[1, 3] = 3 * phi2[2, 3] - 3 * phi2[3, 3] + phi2[4, 3]
#     phi1[2, 2] = 3 * phi1[3, 2] - 3 * phi1[4, 2] + phi1[5, 2]
#     phi1[1, 2] = 3 * phi1[2, 2] - 3 * phi1[3, 2] + phi1[4, 2]
# end

# Time Loop
for n = 2:Ntau + 1

    # Initialize with previous time step
    phi1[3:end-2, n + 1] = phi1[3:end-2, n]
    phi_k[3:end-2, n] = phi[3:end-2, n] 
    phi_first_order[3:end-2, n + 1] = phi_first_order[3:end-2, n]

    for sweep in 1:2
        sw = sweep_cases[sweep]
        # Set c plus and c minus
        global cp = ifelse(sweep == 1, max.(c, 0), zeros(Nx+3))
        global cm = ifelse(sweep == 2, min.(c, 0), zeros(Nx+3))

        for i = sw

            # Check, if the velocity has changed the sign - rarefaction wave
            if (c[i] > 0 && c[i - 1] < 0 && sweep == 1)
                # solve the subsistem 
                A = [1 - c[i - 1]  c[i - 1]; 
                      -c[i]        1 + c[i]]
           
                b = [phi[i - 1, n]; phi[i, n]]
                local x = A \ b  # Solves the system A * x = b
                global ii = i
                phi[i - 1, n + 1] = x[1] 
                phi1[i, n + 1] = x[2]
                phi[i, n + 1] = x[2]
                phi_k[i, n] = x[2] # remember the solution in previous sweep
                b = [phi_first_order[i - 1, n]; phi_first_order[i, n]]
                local x = A \ b  # Solves the system A * x = b
                phi_first_order[i, n + 1] = x[2]
                continue
            end
            # skip already computed points in rearfaction wave in the second sweep
            # if (c[i] < 0 && c[i + 1] > 0 && sweep == 2)
            #     continue
            # end

            phi_first_order[i, n + 1] = ( phi_first_order[i, n + 1] + cp[i] * phi_first_order[i - 1, n + 1] - cm[i] * phi_first_order[i + 1, n + 1] ) / ( 1 + cp[i] - cm[i] );
            
            phi1[i, n + 1] = ( phi1[i, n + 1] + cp[i] * phi[i - 1, n + 1] - cm[i] * phi[i + 1, n + 1] ) / ( 1 + cp[i] - cm[i] );

            phi[i, n + 1] =  ( phi_k[i, n] - 1 / 2 * ( phi1[i, n + 1] - phi[i, n] - phi1[i, n] + phi[i, n - 1] ) * abs(sign(cp[i]-cm[i]))
            + cp[i] * ( phi[i - 1, n + 1] - 1 / 2 * ( phi1[i, n + 1] - phi[i - 1, n + 1] - phi1[i - 1, n + 1] + phi[i - 2, n + 1]) )
            - cm[i] * ( phi[i + 1, n + 1] - 1 / 2 * ( phi1[i, n + 1] - phi[i + 1, n + 1] - phi1[i + 1, n + 1] + phi[i + 2, n + 1]) ) ) / ( 1 + cp[i] - cm[i] );
        
            phi_k[i, n] = phi[i, n + 1]
        end

        # Outlfow boundary condition on the left side
        phi[2, n + 1] = 3 * phi[3, n + 1] - 3 * phi[4, n + 1] + phi[5, n + 1]
        phi1[2, n + 1] = 3 * phi1[3, n + 1] - 3 * phi1[4, n + 1] + phi1[5, n + 1]
        phi_first_order[2, n + 1] = 3 * phi_first_order[3, n + 1] - 3 * phi_first_order[4, n + 1] + phi_first_order[5, n + 1]

        phi[1, n + 1] = 3 * phi[2, n + 1] - 3 * phi[3, n + 1] + phi[4, n + 1]
        phi1[1, n + 1] = 3 * phi1[2, n + 1] - 3 * phi1[3, n + 1] + phi1[4, n + 1]
        phi_first_order[1, n + 1] = 3 * phi_first_order[2, n + 1] - 3 * phi_first_order[3, n + 1] + phi_first_order[4, n + 1]

        phi_k[:, n + 1] = copy(phi[:, n + 1])
    end
end
end
# Print error
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], t[n])) for n in 2:Ntau+2 for i in 2:Nx+2)
println("Error t*h: ", Error_t_h)
Error_t_h_1 = tau * h * sum(abs(phi_first_order[i, n] - phi_exact.(x[i], t[n])) for n in 2:Ntau+2 for i in 2:Nx+2)
println("Error t*h first order: ", Error_t_h_1)

# Error near rearfaction wave
# println("Error near rearfaction wave: ", abs(phi[ii, end-1] - phi_exact.(x[ii], t[end-1])))
println("Max derivative: ", maximum(diff(phi[:, end-1]) / h))
println("Min derivative: ", minimum(diff(phi[:, end-1]) / h))

# Load the last error
last_error = load_last_error()
if last_error !== nothing
    println("Order: ", log(2, last_error / Error_t_h))
end 
# Save the last error
save_last_error(Error_t_h)
println("=============================")

# Plot of the result at the final time together with the exact solution
trace2 = scatter(x = x, y = phi_exact.(x[2:end-1], t[end-1]), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace1 = scatter(x = x, y = phi_0.(x[2:end-1]), mode = "lines", name = "Initial Condition", line=attr(color="black", width=1, dash = "dash") )
trace3 = scatter(x = x, y = phi[2:end-1,end-1], mode = "lines", name = "Solution", line=attr(color="firebrick", width=2))
trace4 = scatter(x = x, y = phi_first_order[2:end-1,end-1], mode = "lines", name = "First order", line=attr(color="green", width=2))

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))

plot_phi = plot([ trace2, trace3, trace1, trace4], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[2:end-1, end-1]) / h, mode = "lines", name = "Sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x[2:end-1], t[end-1])) / h, mode = "lines", name = "Exact sol. gradient")
# trace3_d = scatter(x = x, y = diff(phi_first_order[:, end-1]) / h, mode = "lines", name = "First order sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace2_d, trace1_d], layout_d)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p