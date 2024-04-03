# Program to solve linear advection using the extended backward differentiation formula proposed by J.R. Cash in 1980.

using LinearAlgebra
using PlotlyJS

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")


## Level of refinement
level = 1

order_x = 1

## Courant number
C = 1

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
ω = 1

# Time loop
for n = 1:Ntau

    # [1] Predict phi_hat in time n + 1 -------------------
    global A, b
    # Boundary conditions
    A[1,1] = 1
    A[2,2] = 1
    A[Nx + 1, Nx + 1] = 1

    b[1] = phi_exact(xL, tau*n)
    b[2] = phi_exact(xL + h, tau*n)   
    b[Nx + 1] = phi_exact(xR, tau*n)

    # Interior points
    for i = 3:Nx
        # second order
        if (order_x == 2)
            A[i, i - 2] = c[i] * ω / 2
            A[i, i - 1] =  c[i] * 0.5 * (1 - ω) - c[i] * ω - c[i]
            A[i, i] = 1 - c[i] * (1 - ω) + c[i] * ω / 2 + c[i]
            A[i, i + 1] = c[i] * 0.5 * (1 - ω)
        end
        # first order
        if (order_x == 1)
            A[i, i - 1] = -c[i]
            A[i, i] = 1 + c[i]
        end
        b[i] = phi[i, n]
    end

    # Solve the system
    phi_hat[:, n + 1] = A \ b

    # [2] Compute the prediction in time n + 2 ------------------------------------
    b = phi_hat[:, n + 1]
    b[1] = phi_exact(xL, tau*(n+1))
    b[2] = phi_exact(xL + h, tau*(n+1))   
    b[Nx + 1] = phi_exact(xR, tau*(n+1))
    phi_hat[:, n + 2] = A \ b

    # [3] Compute the fluxes in time n + 2 ----------------------------------------
    if (order_x == 2)
        f_hat[1, n + 2] = c[1] * (phi_hat[2, n + 2] - phi_hat[1, n + 2]) / tau
        f_hat[2, n + 2] = c[2] * (phi_hat[3, n + 2] - phi_hat[2, n + 2]) / tau
        for i = 3:Nx+1
            f_hat[i, n + 2] = c[i] * (phi_hat[i, n + 2] - phi_hat[i - 1, n + 2]) / tau +  c[i] * 0.5 * (phi_hat[i, n + 2] - 2 * phi_hat[i - 1, n + 2] + phi_hat[i - 2, n + 2]) / tau
        end
    end
    if (order_x == 1)
        f_hat[1, n + 2] = c[1] * (phi_hat[2, n + 2] - phi_hat[1, n + 2]) / tau
        f_hat[2, n + 2] = c[2] * (phi_hat[3, n + 2] - phi_hat[2, n + 2]) / tau
        for i = 3:Nx+1
            f_hat[i, n + 2] = c[i] * (phi_hat[i, n + 2] - phi_hat[i - 1, n + 2]) / tau
        end
    end

    # [4] Compute the final solution using the proposed formula -------------------
    b[1] = phi_exact(xL, tau*n) 
    b[2] = phi_exact(xL + h, tau*n) 
    b[Nx + 1] = phi_exact(xR, tau*n) 

    # Interior points
    for i = 3:Nx
        # second order
        if (order_x == 2)
            A[i, i - 2] = 3/2*(c[i] * ω / 2)
            A[i, i - 1] =  3/2 * (c[i] * 0.5 * (1 - ω) - c[i] * ω - c[i])
            A[i, i] = 1 + 3/2*(-c[i] * (1 - ω) + c[i] * ω / 2 + c[i])
            A[i, i + 1] = 3/2*(c[i] * 0.5 * (1 - ω))
        end
        # first order
        if (order_x == 1)
            A[i, i - 1] = -3/2 * c[i]
            A[i, i] = 1 + 3/2 * c[i]
        end
        b[i] = phi[i, n] + tau * 0.5 * f_hat[i, n + 2]
    end

    # Solve the system
    phi[:, n + 1] = A \ b

end

## Error computation
println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)


## Plot
trace1 =  scatter(x = x, y = phi_exact.(x, Ntau*tau), mode = "lines", name = "Exact solution", line=attr(color="black", width=2) )
trace2 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Solution at time T")

layout = Layout(title = "Linear advection using the extended backward differentiation formula",
                xaxis_title = "x",
                yaxis_title = "phi")

plot([trace1, trace2], layout)






