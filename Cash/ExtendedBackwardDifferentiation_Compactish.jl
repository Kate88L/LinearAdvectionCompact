# Program to solve linear advection using the extended backward differentiation formula proposed by J.R. Cash in 1980.

using LinearAlgebra
using PlotlyJS

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")

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

phi_predictor = zeros(Nx + 1, Ntau + 2); # predictor in time n+1
phi_predictor_n2 = zeros(Nx + 1, Ntau + 2); # predictor in time n+2

A = zeros(Nx + 1, Nx + 1)
b = zeros(Nx + 1) # right-hand side

# Initial condition
phi[:,1] = phi_0.(x)
phi_hat[:,1] = phi_0.(x)
phi_predictor[:, 1] = phi_0.(x);
phi_predictor_n2[:, 1] = phi_0.(x);

# Boundary conditions
phi[1,:] = phi_exact.(xL, tau*(0:Ntau))
phi_hat[1,:] = phi_exact.(xL, tau*(0:Ntau+1))
phi_predictor[1,:] = phi_exact.(xL, tau*(0:Ntau+1))
phi_predictor_n2[1,:] = phi_exact.(xL, tau*(1:Ntau+2))

# ENO parameters
s = zeros(Nx + 1, Ntau + 1) .+ 1/3;

# Ghost point on the time -1
ghost_point_time = phi_exact.(x, -tau);

## Computation
ω = zeros(Nx + 1) .+ 1/3
ω[2] = 0
ω[Nx+1] = 1

# Time loop
for n = 1:Ntau

    # [1] [2] Predict phi_hat in time n + 1  and n + 2 -------------------
    for j = 0:1
        if n > 1 
            phi_old = phi_hat[:, n-1+j];
        else
            phi_old = ghost_point_time;
        end

        for i = 2:Nx + 1
            # Predictor
            phi_predictor[i, n + 1 + j] = ( phi_hat[i, n + j] + c[i] * phi_predictor[i - 1, n + 1 + j] ) / ( 1 + c[i] );

            # Corrector
            r_downwind_n_old = phi_predictor[i, n + j] - phi_predictor[i - 1, n + 1 + j];
            r_upwind_n_old = - phi_hat[i - 1, n + j] + phi_old[i];

            r_downwind_n_new = - phi_predictor[i, n + 1 + j] + phi_predictor_n2[i - 1, n + 1 + j];
            r_upwind_n_new = phi_hat[i - 1, n + 1 + j] - phi_hat[i, n + j];

            phi_hat[i, n + 1 + j] = ( phi_hat[i, n + j] + c[i] * phi_hat[i - 1, n + 1 + j] - 0.5 * ( (1-s[i,n + 1]) * (r_downwind_n_old + r_downwind_n_new) 
            + (s[i,n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + c[i] );

            # Predictor for next time step
            phi_predictor_n2[i, n + 1 + j] = ( phi_hat[i, n + 1 + j] + c[i] * phi_predictor_n2[i - 1, n + 1 + j] ) / ( 1 + c[i] );
        end
    end

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
    global A, b
    b[1] = phi_exact(xL, tau*n) 
    A[1,1] = 1

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

    # phi[:, n + 1] = phi_hat[:, n + 1]

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






