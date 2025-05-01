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
level = 4;

# Courant number
C = 2;

ωk = 1/3;
αk = 1/3;

# Grid settings
xL = - 1 * π / 2 * 0
xR = 3 * π / 2 * 0 + 1.5
Nx = 40 * 2^level
h = (xR - xL) / Nx

# Constant velocity example
u(x) = +1.0
# phi_0(x) = cos.(x);
phi_0(x) = exp(-40 * (x-0.5*1.5).^2)
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
T = 1.2 /10
tau = C * h / maximum(abs.(u.(x))) 
Ntau = Int(round(T / tau))

# Ntau = 2

t = range(0, (Ntau + 2) * tau, length = Ntau + 3)

N = (Nx + 3) * (Ntau + 3);

c = zeros(Nx+3) .+ u.(x) * tau / h

phi = zeros(Nx + 3, Ntau + 3);
phi[:, 1] = phi_0.(x)

function idx(i, n) 
    return i + (n-1) * (Nx + 3)
end

# System matrix - FIRST ORDER -----------------------------------
function A1() 
    A = zeros(N, N)

    for n in 1:Ntau+3
        for i in 1:Nx+3
            # Boundary conditions
            if (i == 1) || (i == 2) || (n == 1) || (n == 2) 
                A[idx(i, n), idx(i, n)] = 1
            # Interior points
            else
                A[idx(i, n), idx(i, n)] = 1 + c[i] 
                A[idx(i, n), idx(i-1, n)] = - c[i]
                A[idx(i, n), idx(i, n - 1)] = -1
            end
        end
    end

    return A
end

# System matrix - SECOND/THIRD ORDER -----------------------------------
function A2() 
    A = zeros(N, N)

    for n in 1:Ntau+3
        for i in 1:Nx+3
            # Boundary conditions
            if (i == 1) || (i == 2) || (n == 1) || (n == 2) 
                A[idx(i, n), idx(i, n)] = 1
            elseif  (i == Nx + 3) && (n == Ntau + 3)
                A[idx(i, n), idx(i, n)] = 1 + 0.5 * 1 + c[i] + 0.5 * 1 * c[i]
                A[idx(i, n), idx(i - 1, n)] = - 2 * c[i] 
                A[idx(i, n), idx(i - 2, n)] = c[i] * 0.5
                A[idx(i, n), idx(i, n - 1)] = -2
                A[idx(i, n), idx(i, n - 2)] = 0.5
            elseif (i == Nx + 3) && (n < Ntau + 3)
                A[idx(i, n), idx(i, n)] = 1 + 0.5 * αk - (1 - αk) + c[i] + 0.5 * 1 * c[i]
                A[idx(i, n), idx(i - 1, n)] = - c[i] - 1 * c[i] 
                A[idx(i, n), idx(i - 2, n)] = c[i] * 1 * 0.5
                A[idx(i, n), idx(i, n - 1)] = -1 - αk + 0.5 * (1 - αk) 
                A[idx(i, n), idx(i, n - 2)] = αk * 0.5
                A[idx(i, n), idx(i, n + 1)] = ( 1 - αk ) * 0.5
            elseif (i < Nx + 3) && (n == Ntau + 3)
                A[idx(i, n), idx(i, n)] = 1 + 0.5 * 1 + c[i] + 0.5 * ωk * c[i] - (1 - ωk) * c[i]
                A[idx(i, n), idx(i - 1, n)] = - c[i] - ωk * c[i] + 0.5 * (1 - ωk) * c[i]
                A[idx(i, n), idx(i - 2, n)] = c[i] * ωk * 0.5
                A[idx(i, n), idx(i + 1, n)] = c[i] * (1 - ωk) * 0.5
                A[idx(i, n), idx(i, n - 1)] = -1 - 1 
                A[idx(i, n), idx(i, n - 2)] = 1 * 0.5
            # Interior points
            else
                A[idx(i, n), idx(i, n)] = 1 + 0.5 * αk - (1 - αk) + c[i] + 0.5 * ωk * c[i] - (1 - ωk) * c[i]
                A[idx(i, n), idx(i - 1, n)] = - c[i] - ωk * c[i] + 0.5 * (1 - ωk) * c[i]
                A[idx(i, n), idx(i - 2, n)] = c[i] * ωk * 0.5
                A[idx(i, n), idx(i + 1, n)] = c[i] * (1 - ωk) * 0.5
                A[idx(i, n), idx(i, n - 1)] = -1 - αk + 0.5 * (1 - αk) 
                A[idx(i, n), idx(i, n - 2)] = αk * 0.5
                A[idx(i, n), idx(i, n + 1)] = ( 1 - αk ) * 0.5
            end
        end
    end

    return A
end

function rightHandSide()
    b = zeros(N)
    
    for n in 1:Ntau+3
        for i in 1:Nx+3
            # Boundary conditions
            if (i == 1) || (i == 2) || (n == 1) || (n == 2)
                b[idx(i, n)] = phi_exact.(x[i], t[n])
            # Interior points
            else
                b[idx(i, n)] = 0 # Initial condition
            end
        end
    end

    return b
end


@time begin

matrix = A2()
b = rightHandSide()

Phi = matrix \ b

phi = reshape(Phi, (Nx + 3, Ntau + 3))
    
end
# Print error
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], t[n])) for n in 2:Ntau+2 for i in 2:Nx+2)
println("Error t*h: ", Error_t_h)

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

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))

plot_phi = plot([ trace2, trace3, trace1], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[2:end-1, end-1]) / h, mode = "lines", name = "Sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x[2:end-1], t[end-1])) / h, mode = "lines", name = "Exact sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace2_d, trace1_d], layout_d)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p