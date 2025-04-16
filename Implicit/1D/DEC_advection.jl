# Deferred correction scheme for linear advection equation

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
level = 0;

# Number of DEC iterations (accuracy is k + 1)
k = 2

# Courant number
C = 1.5;

ωk = 1/3;
αk = 1/3;

# Grid settings
xL = - 1 * π / 2 * 0
xR = 3 * π / 2 * 0 + 1.5
Nx = 40 * 2^level
h = (xR - xL) / Nx

# Constant velocity example
u(x) = -1.0
# phi_0(x) = cos.(x);
phi_0(x) = exp(-40 * (x-1.5).^2)
phi_exact(x, t) = phi_0.(x - u.(x) * t);    

# Variable velocity example
# offset = 0.5
# u(x) = sin.(x - offset)
# phi_0(x) = sin.(x - offset);
# phi_exact(x, t) = sin.( 2 * atan.( exp.(-t) .* tan.( (x-offset) ./ 2) ) )

## Comptutation

# Grid initialization
x = range(xL - h, xR + h, length = Nx + 3)

# Time
T = 1.2 /2
tau = C * h / maximum(abs.(u.(x))) 
Ntau = Int(round(T / tau))

# Ntau = 1

t = range(0, (Ntau + 2) * tau, length = Ntau + 3)

c = zeros(Nx+3) .+ u.(x) * tau / h

# Set c plus and c minus
cp = max.(c, 0)
cm = min.(c, 0)

phi = zeros(Nx + 3, Ntau + 3);
phi_p = zeros(Nx + 3, Ntau + 3);
phi_p2 = zeros(Nx + 3, Ntau + 3);
phi_first_order = zeros(Nx + 3, Ntau + 3);

# Initial condition
phi[:, 1] = phi_0.(x);
phi[:, 2] = phi_exact.(x, tau);

# Boundary conditions - inflow
phi[end, :] = phi_exact.(x[end], t);
phi[end-1, :] = phi_exact.(x[end-1], t);

# phi[2, :] = phi_exact.(x[2], t);
# phi[1, :] = phi_exact.(x[1], t);

phi_p = copy(phi)
phi_p2 = copy(phi)
phi_first_order = copy(phi)

# Right hand side
function rightHandSide(f, n)
    b = zeros(Nx + 3)
    b[:] = f[:, n] # Initial condition
    # Boundary - exact solution
    b[end] = phi_exact.(x[end], t[n + 1])
    b[end-1] = phi_exact.(x[end-1], t[n + 1])
    # b[2] = phi_exact.(x[2], t[n + 1])
    # b[1] = phi_exact.(x[1], t[n + 1])
    return b
end

# System matrix - FIRST ORDER -----------------------------------
function L1() 
    d = zeros(Nx + 3); # Diagonal
    l_1 = zeros(Nx + 2); # Lower diagonal 1
    u_1 = zeros(Nx + 2); # Upper diagonal 1

    d = d .+ 1 .+ ( cp - cm )
    l_1 = l_1 + ( - cp[2:end] )
    u_1 = u_1 + ( cm[1:end-1] )
    
    A = Tridiagonal(l_1, d, u_1)

    # Boundary conditions - inflow
    A[end-1,:] .= 0.0
    A[end, :] .= 0.0
    A[end, end] = 1.0
    A[end-1, end-1] = 1.0

    # A[1, :] .= 0.0
    # A[2, :] .= 0.0  
    # A[1, 1] = 1.0
    # A[2, 2] = 1.0

    return A
end


@time begin

# Time Loop
for n = 2:Ntau + 1

    function L2(f, n, α = αk, ω = ωk)
        b = zeros(Nx + 3)
    
        # Simplified notation for easier reading
        i = 3:Nx + 1
        ip = 4:Nx + 2
        ipp = 5:Nx + 3
        im = 2:Nx
        imm = 1:Nx - 1
    
        if n > Ntau + 1
            b[i] = b[i] + 0.5 * ( f[i, n] - f[i, n - 1] - f[i, n - 1] + f[i, n - 2] )
        else
            b[i] = b[i] + 0.5 * ( α .* ( f[i, n] - f[i, n - 1] - f[i, n - 1] + f[i, n - 2] ) +
                                ( 1 - α ) .* ( f[i, n + 1] - f[i, n] - f[i, n] + f[i, n - 1] ) )
        end
    
        b[i] = b[i] + (  0.5 * ( + cp[i] .* ( ω .* ( f[i, n] - 2 * f[im, n] + f[imm, n] ) +
                                            ( 1 - ω ) .* ( f[ip, n] - 2 * f[i, n] + f[im, n] ) )
                                  - cm[i] .* ( ω .* ( f[i, n] - 2 * f[ip, n] + f[ipp, n] )  +
                                            ( 1 - ω ) .* ( f[ip, n] - 2 * f[i, n] + f[im, n] ) ) ) )             
        return b
    end

    # First order solution
    phi_first_order[:, n + 1] = L1() \ rightHandSide(phi_first_order, n) # Solves the system A * x = b 

    # Iteration 0 : first order solution of the system
    phi_p[:, n + 1] = L1() \ rightHandSide(phi_p, n) # Solves the system L1p = b

    # Iteration 0+ : first order predictor in the future
    phi_p[:, n + 2] = L1() \ rightHandSide(phi_p, n + 1) # Solves the system L1p = b

    # Iteration 1 : second order solution of the system
    phi_p2[:, n + 1] = L1() \ ( rightHandSide(phi_p2, n) - L2(phi_p, n + 1, 1, 1) ) # Solves the system L1 = L1p - L2p

    phi[:, n + 1] = phi_p2[:, n + 1] # Assign second order solution to the main variable

    if k > 1
        # Iteration 1+ : second order solution of the system in the future
        phi_p2[:, n + 2] = L1() \ ( rightHandSide(phi_p2, n + 1) - L2(phi_p, n + 2, 1, 1) ) # Solves the system L1 = L1p - L2p 
        # Iteration 2 : third order solution of the system
        phi[:, n + 1] = L1() \ ( rightHandSide(phi, n) - L2(phi_p2, n + 1, 1/3, 1/3) ) # Solves the system L1 = L1p - L2p
    end
    # Outlfow boundary condition 
    phi[2, n + 1] = 3 * phi[3, n + 1] - 3 * phi[4, n + 1] + phi[5, n + 1]
    phi[1, n + 1] = 3 * phi[2, n + 1] - 3 * phi[3, n + 1] + phi[4, n + 1]

    # phi[end-1, n + 1] = 3 * phi[end-2, n + 1] - 3 * phi[end-3, n + 1] + phi[end-4, n + 1]
    # phi[end, n + 1] = 3 * phi[end-1, n + 1] - 3 * phi[end-2, n + 1] + phi[end-3, n + 1]
    
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
trace3 = scatter(x = x, y = phi[2:end-1,end - 1], mode = "lines", name = "Solution", line=attr(color="firebrick", width=2))
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