# Program to solve linear advection using the extended backward differentiation formula proposed by J.R. Cash in 1980.

using LinearAlgebra
using PlotlyJS

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")
include("../Utils/Solvers.jl")

@time begin
## Level of refinement
level = 3
order_x = 2

## Courant number
C = 8

## Model
u(x) = 1 # velocity
phi_0(x) = cos(x) # initial condition
phi_exact(x, t) = phi_0(x - t) # exact solution

## Grid settings
xL = - 1 * π / 2
xR = 3 * π / 2

Nx = 100 * 2^level
h = (xR - xL) / Nx

x = range(xL, xR, length = Nx + 1)

## Time settings
T = 8 * π / sqrt(7)

tau = C * h / maximum(u.(x))
Ntau = Int(round(T / tau))

c = zeros(Nx+1,1) .+ u.(x) * tau / h

## Solution initialization
phi = zeros(Nx + 1, Ntau + 1)
phi_hat = zeros(Nx + 1, Ntau + 2)
f_hat = zeros(Nx + 1, Ntau + 2)

A = zeros(Nx + 1, Nx + 1)
b = zeros(Nx + 1) # right-hand side

# Initial condition
phi[:,1] = phi_0.(x)

# Boundary conditions
phi[1,:] = phi_exact.(xL, tau*(0:Ntau))

## Computation
ω = zeros(Nx + 1) .+ 1/3
ω[2] = 0
ω[Nx+1] = 1

# Time loop
for n = 1:Ntau

    # [1] Predict phi_hat in time n + 1 -------------------
    global A, b
    # Inflow boundary condition
    A[1,1] = 1

    b[1] = phi_exact(xL, tau*n) 

    # Interior points
    for i = 2:Nx+1
        # second order
        if (order_x == 2)
            try A[i, i - 2] = c[i] * ω[i] / 2 catch end
            A[i, i - 1] = - c[i] + c[i] * (1 - ω[i]) / 2 - c[i] * ω[i] 
            A[i, i] = 1 + c[i] - c[i] * (1 - ω[i]) + c[i] * ω[i] / 2 
            try  A[i, i + 1] = c[i] * (1 - ω[i]) / 2 catch end
        end
        # first order
        if (order_x == 1)
            A[i, i - 1] = -c[i]
            A[i, i] = 1 + c[i]
        end
        b[i] = phi[i, n]
    end

    # Solve the system
    # phi_hat[:, n + 1] = A \ b
    phi_hat[:, n + 1] = modifiedThomasAlgorithm(A, b)

    # [2] Compute the prediction in time n + 2 ------------------------------------
    b = phi_hat[:, n + 1]
    b[1] = phi_exact(xL, tau*(n+1))  
    # phi_hat[:, n + 2] = A \ b
    phi_hat[:, n + 2] = modifiedThomasAlgorithm(A, b)

    # [3] Compute the fluxes in time n + 2 ----------------------------------------
    if (order_x == 2 )
        f_hat[1, n + 2] = c[1] * (phi_hat[2, n + 2] - phi_hat[1, n + 2]) / tau
        for i = 2:Nx+1
            if (i == Nx + 1)
                f_hat[i, n + 2] = c[i] * (phi_hat[i, n + 2] - phi_hat[i - 1, n + 2]) / tau +
                c[i] * 0.5 * ω[i] * (phi_hat[i, n + 2] - 2 * phi_hat[i - 1, n + 2] + phi_hat[i - 2, n + 2]) / tau
            elseif ( i == 2)
                f_hat[i, n + 2] = c[i] * (phi_hat[i, n + 2] - phi_hat[i - 1, n + 2]) / tau +
                c[i] * 0.5 * (1 - ω[i]) * (phi_hat[i + 1, n + 2] - 2 * phi_hat[i, n + 2] + phi_hat[i - 1, n + 2]) / tau 
            else 
                f_hat[i, n + 2] = c[i] * (phi_hat[i, n + 2] - phi_hat[i - 1, n + 2]) / tau +
                c[i] * 0.5 * (1 - ω[i]) * (phi_hat[i + 1, n + 2] - 2 * phi_hat[i, n + 2] + phi_hat[i - 1, n + 2]) / tau +
                c[i] * 0.5 * ω[i] * (phi_hat[i, n + 2] - 2 * phi_hat[i - 1, n + 2] + phi_hat[i - 2, n + 2]) / tau
            end
        end
    end
    if (order_x == 1)
        f_hat[1, n + 2] = c[1] * (phi_hat[2, n + 2] - phi_hat[1, n + 2]) / tau
        for i = 2:Nx+1
            f_hat[i, n + 2] = c[i] * (phi_hat[i, n + 2] - phi_hat[i - 1, n + 2]) / tau
        end
    end

    # [4] Compute the final solution using the proposed formula -------------------
    b[1] = phi_exact(xL, tau*n) 

    # Interior points
    for i = 2:Nx + 1
        # second order
        if (order_x == 2)
            try A[i, i - 2] = 3/2 * c[i] * ω[i] / 2 catch end
            A[i, i - 1] = 3/2* ( - c[i] + c[i] * (1 - ω[i]) / 2 - c[i] * ω[i] )
            A[i, i] = 1 + 3/2* ( c[i] - c[i] * (1 - ω[i]) + c[i] * ω[i] / 2 )
            try A[i, i + 1] = 3/2 * c[i] * (1 - ω[i]) / 2 catch end
        end
        # first order
        if (order_x == 1)
            A[i, i - 1] = -3/2 * c[i]
            A[i, i] = 1 + 3/2 * c[i]
        end
        b[i] = phi[i, n] + tau * 0.5 * f_hat[i, n + 2]
    end

    # Solve the system
    # phi[:, n + 1] = A \ b
    phi[:, n + 1] = modifiedThomasAlgorithm(A, b)

end
end

## Error computation
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)
println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)


## Plot
trace1 =  scatter(x = x, y = phi_exact.(x, Ntau*tau), mode = "lines", name = "Exact solution", line=attr(color="black", width=2) )
trace2 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Solution at time T")

layout = Layout(title = "Linear advection using the extended backward differentiation formula",
                xaxis_title = "x",
                yaxis_title = "phi")

plot([trace1, trace2], layout)






