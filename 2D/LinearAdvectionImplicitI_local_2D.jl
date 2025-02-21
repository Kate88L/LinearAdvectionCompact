#  iterative 2nd order fully implicit scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames
using JSON
using DataStructures

include("../Utils/InitialFunctions.jl")
include("../Utils/ExactSolutions.jl")
include("../Utils/Utils.jl")

## Definition of basic parameters

third_order = true;

# Level of refinement
level = 0;

K = 1 # Number of iterations for the second order correction

# Courant number
C = 1.5;

# Grid settings
xL = -3
xR = 3
yL = xL
yR = xR
N = 40 * 2^level
h = (xR - xL) / N

# Velocity
u(x, y) = 1.0
v(x, y) = 1.0

# Initial condition
# phi_0(x, y) = x.^2 + y.^2
# phi_0(x, y) = piecewiseLinear2D(x, y);
# phi_0(x, y) = exp.(-10 * (x.^2 + y.^2));
phi_0(x,y) = 0.8*exp.(-(x.^2 + y.^2)/1.7);

# Exact solution
phi_exact(x, y, t) = phi_0.(x - u(x,y) .* t, y - v(x,y) .* t);          

## INITIALIZATION ===============================================================================================

# Grid initialization with ghost point on the left and right
x = range(xL-h, xR+h, length = N + 3)
y = range(yL-h, yR+h, length = N + 3)

X = repeat(x, 1, length(y))
Y = repeat(y, 1, length(x))'

# Time
Tfinal = 1 * π / sqrt(7) * 1 / 5
Ntau = 1 * 2^level
tau = C * h / maximum(max(abs.(u.(x, y)), abs.(v.(x, y))))
Ntau = Int(round(Tfinal / tau))

# Ntau = 0

t = range(0, (Ntau + 2) * tau, length = Ntau + 3)

c = zeros(N+3, N+3) .+ u.(x, y)' * tau / h
d = zeros(N+3, N+3) .+ v.(x, y) * tau / h

c_p = max.(c, 0) # Positive part
c_m = min.(c, 0) # Negative part
d_p = max.(d, 0) # Positive part 
d_m = min.(d, 0) # Negative part

# predictor
phi1 = zeros(N + 3, N + 3, Ntau + 3);
# corrector 
phi2 = zeros(N + 3, N + 3, Ntau + 3);
# final solution
phi = zeros(N + 3, N + 3, Ntau + 3);
# first order
phi_first_order = zeros(N + 3, N + 3, Ntau + 3)

# Initial condition
phi1[:, :, 1] = phi_0.(X, Y); # Ghost points
phi2[:, :, 1] = phi_0.(X, Y);
phi[:, :, 1] = phi_0.(X, Y);
phi_first_order[:, :, 1] = phi_0.(X, Y);

phi1[:, :, 2] = phi_exact.(X, Y, tau); # Initial condition
phi2[:, :, 2] = phi_exact.(X, Y, tau);
phi[:, :, 2] = phi_exact.(X, Y, tau);
phi_first_order[:, :, 2] = phi_exact.(X, Y, tau);

phi_predictor_i = zeros(N + 3) # vector values needed to predict in space 
phi_predictor_j = [CircularBuffer{Float64}(2) for _ in 1:N + 3] # buffer  of last 3 values needed to predict in space

# WENO parameters
ϵ = 1e-16;
ω0 = 0;
α0 = 0;

ω1_i = zeros(N + 3, N + 3) .+ ω0;
ω2_i = zeros(N + 3, N + 3) .+ ω0;
ω1_j = zeros(N + 3, N + 3) .+ ω0;
ω2_j = zeros(N + 3, N + 3) .+ ω0;
α1 = zeros(N + 3, N + 3) .+ α0;
α2 = zeros(N + 3, N + 3) .+ α0;

# Definition of 4 sweep cases for the fast sweeping method
sweep_cases = Dict(
    1 => (3:1:N+1, 3:1:N+1),
    2 => (3:1:N+1, N+1:-1:3),
    3 => (N+1:-1:3, 3:1:N+1),
    4 => (N+1:-1:3, N+1:-1:3)
)

sweep_cases_mask = Dict(
    1 => (1:1:N+3, 1:1:N+3),
    2 => (1:1:N+3, N+3:-1:1),
    3 => (N+3:-1:1, 1:1:N+3),
    4 => (N+3:-1:1, N+3:-1:1)
)

# Boundary conditions
for n = 2:Ntau + 3
    phi1[1, :, n] = phi_exact.(x[1], y, t[n]);
    phi2[1, :, n] = phi_exact.(x[1], y, t[n]);
    phi[1, :, n] = phi_exact.(x[1], y, t[n]);
    phi_first_order[1, :, n] = phi_exact.(x[1], y, t[n]);
    phi1[:, 1, n] = phi_exact.(x, y[1], t[n]);
    phi2[:, 1, n] = phi_exact.(x, y[1], t[n]);
    phi[:, 1, n] = phi_exact.(x, y[1], t[n]);
    phi_first_order[:, 1, n] = phi_exact.(x, y[1], t[n]);

    phi1[2, :, n] = phi_exact.(x[2], y, t[n]);
    phi2[2, :, n] = phi_exact.(x[2], y, t[n]);
    phi[2, :, n] = phi_exact.(x[2], y, t[n]);
    phi_first_order[2, :, n] = phi_exact.(x[2], y, t[n]);
    phi1[:, 2, n] = phi_exact.(x, y[2], t[n]);
    phi2[:, 2, n] = phi_exact.(x, y[2], t[n]);
    phi[:, 2, n] = phi_exact.(x, y[2], t[n]);
    phi_first_order[:, 2, n] = phi_exact.(x, y[2], t[n]);

    phi1[N + 2, :, n] = phi_exact.(x[N + 2], y, t[n]);
    phi2[N + 2, :, n] = phi_exact.(x[N + 2], y, t[n]);
    phi[N + 2, :, n] = phi_exact.(x[N + 2], y, t[n]);
    phi_first_order[N + 2, :, n] = phi_exact.(x[N + 2], y, t[n]);
    phi1[:, N + 2, n] = phi_exact.(x, y[N + 2], t[n]);
    phi2[:, N + 2, n] = phi_exact.(x, y[N + 2], t[n]);
    phi[:, N + 2, n] = phi_exact.(x, y[N + 2], t[n]);
    phi_first_order[:, N + 2, n] = phi_exact.(x, y[N + 2], t[n]);

    phi1[N + 3, :, n] = phi_exact.(x[N + 3], y, t[n]);
    phi2[N + 3, :, n] = phi_exact.(x[N + 3], y, t[n]);
    phi[N + 3, :, n] = phi_exact.(x[N + 3], y, t[n]);
    phi_first_order[N + 3, :, n] = phi_exact.(x[N + 3], y, t[n]);
    phi1[:, N + 3, n] = phi_exact.(x, y[N + 3], t[n]);
    phi2[:, N + 3, n] = phi_exact.(x, y[N + 3], t[n]);
    phi[:, N + 3, n] = phi_exact.(x, y[N + 3], t[n]);
    phi_first_order[:, N + 3, n] = phi_exact.(x, y[N + 3], t[n]);
end

## MAIN LOOP ====================================================================================================

@time begin

# Precompute predictors for the initial time step
for sweep in 1:4
    swI, swJ = sweep_cases[sweep]

    for i = swI
    for j = swJ
        phi1[i, j, 2] = ( phi[i, j, 1] + c_p[i, j] * phi[i - 1, j, 2]
                                       + d_p[i, j] * phi[i, j - 1, 2] 
                                       - c_m[i, j] * phi[i + 1, j, 2] 
                                       - d_m[i, j] * phi[i, j + 1, 2] ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );
        phi1[i - 1, j, 3] = ( phi[i - 1, j, 2] + c_p[i - 1, j] * phi2[i - 2, j, 3]
                                               + d_p[i - 1, j] * phi2[i - 1, j - 1, 3]
                                               - d_m[i - 1, j] * phi2[i - 1, j + 1, 3] ) / ( 1 + c_p[i - 1, j] + d_p[i - 1, j] - d_m[i, j] );
        phi1[i + 1, j, 3] = ( phi[i + 1, j, 2] - c_m[i + 1, j] * phi2[i + 2, j, 3]
                                               + d_p[i + 1, j] * phi2[i + 1, j - 1, 3]
                                               - d_m[i + 1, j] * phi2[i + 1, j + 1, 3] ) / ( 1 - c_m[i + 1, j] + d_p[i + 1, j] - d_m[i, j] );
        phi1[i, j - 1, 3] = ( phi[i, j - 1, 2] + c_p[i, j - 1] * phi2[i - 1, j - 1, 3]
                                               - c_m[i, j - 1] * phi2[i + 1, j - 1, 3]
                                               + d_p[i, j - 1] * phi2[i, j - 2, 3] ) / ( 1 + c_p[i, j - 1] - c_m[i, j - 1] + d_p[i, j - 1] );
        phi1[i, j + 1, 3] = ( phi[i, j + 1, 2] + c_p[i, j + 1] * phi2[i - 1, j + 1, 3]
                                               - c_m[i, j + 1] * phi2[i + 1, j + 1, 3]
                                               - d_m[i, j + 1] * phi2[i, j + 2, 3] ) / ( 1 + c_p[i, j + 1] - c_m[i, j + 1] - d_m[i, j + 1] );    
        phi2[i, j, 3] = ( phi[i, j, 2] - 0.5 * ( -phi1[i, j, 2] + phi[i, j, 1] ) + c_p[i, j] * ( phi2[i - 1, j, 3] - 0.5 * ( -phi1[i - 1, j, 3] + phi2[i - 2, j, 3] ) ) 
                                                                                 + d_p[i, j] * ( phi2[i, j - 1, 3] - 0.5 * ( -phi1[i, j - 1, 3] + phi2[i, j - 2, 3] ) )
                                                                                 - c_m[i, j] * ( phi2[i + 1, j, 3] - 0.5 * ( -phi1[i + 1, j, 3] + phi2[i + 2, j, 3] ) )
                                                                                 - d_m[i, j] * ( phi2[i, j + 1, 3] - 0.5 * ( -phi1[i, j + 1, 3] + phi2[i, j + 2, 3] ) ) ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );
    end
    end

end

# Time Loop
for n = 2:Ntau + 1

    phi2_old_ = copy(phi2[:, :, n + 1]);

    for sweep in 1:4
        swI, swJ = sweep_cases[sweep]
        swI_m, swJ_m = sweep_cases_mask[sweep]

        phi_predictor_i[1:end] = phi2_old_[swI[1], 1:end];
        if sweep % 2 == 1
            for j = swJ_m[2:end-1]
                empty!(phi_predictor_j[j]);
                push!(phi_predictor_j[j], phi2_old_[swI_m[1], j + 1]);
                push!(phi_predictor_j[j], phi2_old_[swI_m[2], j + 1]);
            end
        else
            for j = swJ_m[2:end-1]
                empty!(phi_predictor_j[j]);
                push!(phi_predictor_j[j], phi2_old_[swI_m[1], j - 1]);
                push!(phi_predictor_j[j], phi2_old_[swI_m[2], j - 1]);
            end
        end

        for i = swI
            phi_predictor_i[swJ_m[1:2]] = ifelse(sweep < 3, phi2_old_[i + 1, swJ_m[1:2]], phi2_old_[i - 1, swJ_m[1:2]]);  
            phi_predictor_j[swJ_m[2]][2] = phi2_old_[i, swJ[1]]
            for j = swJ

                # First order solution
                phi_first_order[i, j, n + 1] = ( phi_first_order[i, j, n] + c_p[i, j] * phi_first_order[i - 1, j, n + 1] 
                                                                          + d_p[i, j] * phi_first_order[i, j - 1, n + 1] 
                                                                          - c_m[i, j] * phi_first_order[i + 1, j, n + 1]
                                                                          - d_m[i, j] * phi_first_order[i, j + 1, n + 1] ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );

                phi2_old = phi2_old_[i, j];
                phi2_i_old_p = phi_predictor_i[j];
                phi2_j_old_p = ifelse(sweep % 2 == 1, phi_predictor_j[j-1][2], phi_predictor_j[j+1][2]);

                # FIRST ITERATION 
                phi1[i, j, n] = ( phi[i, j, n - 1] + c_p[i, j] * phi[i - 1, j, n] 
                                                   + d_p[i, j] * phi[i, j - 1, n]
                                                   - c_m[i, j] * phi[i + 1, j, n]
                                                   - d_m[i, j] * phi[i, j + 1, n] ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );
                phi1[i - 1, j, n + 1] = ( phi[i - 1, j, n] + c_p[i - 1, j] * phi[i - 2, j, n + 1] 
                                                           + d_p[i - 1, j] * phi[i - 1, j - 1, n + 1]
                                                           - d_m[i - 1, j] * phi[i - 1, j + 1, n + 1] ) / ( 1 + c_p[i - 1, j] + d_p[i - 1, j] - d_m[i, j] );
                phi1[i + 1, j, n + 1] = ( phi[i + 1, j, n] - c_m[i + 1, j] * phi[i + 2, j, n + 1] 
                                                           + d_p[i + 1, j] * phi[i + 1, j - 1, n + 1]
                                                           - d_m[i + 1, j] * phi[i + 1, j + 1, n + 1] ) / ( 1 - c_m[i + 1, j] + d_p[i + 1, j] - d_m[i, j] );
                phi1[i, j - 1, n + 1] = ( phi[i, j - 1, n] + c_p[i, j - 1] * phi[i - 1, j - 1, n + 1] 
                                                           - c_m[i, j - 1] * phi[i + 1, j - 1, n + 1]
                                                           + d_p[i, j - 1] * phi[i, j - 2, n + 1] ) / ( 1 + c_p[i, j - 1] - c_m[i, j] + d_p[i, j - 1] );
                phi1[i, j + 1, n + 1] = ( phi[i, j + 1, n] + c_p[i, j + 1] * phi[i - 1, j + 1, n + 1] 
                                                           - c_m[i, j + 1] * phi[i + 1, j + 1, n + 1]
                                                           - d_m[i, j + 1] * phi[i, j + 2, n + 1] ) / ( 1 + c_p[i, j + 1] - c_m[i, j + 1] - d_m[i, j + 1] );
                phi1[i, j, n + 1] = ( phi[i, j, n] + c_p[i, j] * phi[i - 1, j, n + 1] 
                                                   + d_p[i, j] * phi[i, j - 1, n + 1] 
                                                   - c_m[i, j] * phi[i + 1, j, n + 1]
                                                   - d_m[i, j] * phi[i, j + 1, n + 1] ) / ( 1 + c_p[i, i] + d_p[i, j] - c_m[i, j] - d_m[i, j] );

                phi[i, j, n + 1] = ( phi[i, j, n] - 0.5 * ( phi1[i, j, n + 1] - phi[i, j, n] - phi1[i, j, n] + phi[i, j, n - 1] ) 
                    + c_p[i, j] * ( phi[i - 1, j, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i - 1, j, n + 1] - phi1[i - 1, j, n + 1] + phi[i - 2, j, n + 1] ) )
                    - c_m[i, j] * ( phi[i + 1, j, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i + 1, j, n + 1] - phi1[i + 1, j, n + 1] + phi[i + 2, j, n + 1] ) )
                    + d_p[i, j] * ( phi[i, j - 1, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i, j - 1, n + 1] - phi1[i, j - 1, n + 1] + phi[i, j - 2, n + 1] ) )
                    - d_m[i, j] * ( phi[i, j + 1, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i, j + 1, n + 1] - phi1[i, j + 1, n + 1] + phi[i, j + 2, n + 1] ) ) ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );
                    
                phi2[i, j, n + 1] = phi[i, j, n + 1];

                # Compute first order predictor in x direction
                if sweep < 3 # moving from left to right
                    phi1[i + 1, j, n] = ( phi[i + 1, j, n - 1] + c_p[i + 1, j] * phi[i, j, n] 
                                                               + d_p[i + 1, j] * phi[i + 1, j - 1, n] 
                                                               - d_m[i + 1, j] * phi[i + 1, j + 1, n]) / ( 1 + c_p[i + 1, j] + d_p[i + 1, j] - d_m[i + 1, j] );
                    phi1[i + 1, j, n + 1] = ( phi[i + 1, j, n] + c_p[i + 1, j] * phi[i, j, n + 1]
                                                               + d_p[i + 1, j] * phi_predictor_i[j - 1]
                                                               - d_m[i + 1, j] * phi_predictor_i[j + 1] ) / ( 1 + c_p[i + 1, j] + d_p[i + 1, j] - d_m[i + 1, j]);
                    phi_predictor_i[j] = ( phi[i + 1, j, n] - 0.5 * ( phi1[i + 1, j, n + 1] - phi[i + 1, j, n] - phi1[i + 1, j, n] + phi[i + 1, j, n - 1] ) 
                        + c_p[i + 1, j] * ( phi2[i, j, n + 1] - 0.5 * ( phi1[i + 1, j, n + 1] - phi2[i, j, n + 1] - phi1[i, j, n + 1] + phi[i - 1, j, n + 1]) ) 
                        + d_p[i + 1, j] * ( phi_predictor_i[j - 1] - 0.5 * ( phi1[i + 1, j, n + 1] - phi_predictor_i[j - 1] - phi1[i + 1, j - 1, n + 1] + phi_predictor_i[j - 2] ) ) 
                        - d_m[i + 1, j] * ( phi_predictor_i[j + 1] - 0.5 * ( phi1[i + 1, j, n + 1] - phi_predictor_i[j + 1] - phi1[i + 1, j + 1, n + 1] + phi_predictor_i[j + 2] ) ) ) / ( 1 + c_p[i + 1, j] + d_p[i + 1, j] - d_m[i + 1, j] );
                else # moving from right to left
                    phi1[i - 1, j, n] = ( phi[i - 1, j, n - 1] - c_m[i - 1, j] * phi[i, j, n] 
                                                               + d_p[i - 1, j] * phi[i - 1, j - 1, n] 
                                                               - d_m[i - 1, j] * phi[i - 1, j + 1, n]) / ( 1 - c_m[i - 1, j] + d_p[i - 1, j] - d_m[i - 1, j] );
                    phi1[i - 1, j, n + 1] = ( phi[i - 1, j, n] - c_m[i - 1, j] * phi[i, j, n + 1]
                                                               + d_p[i - 1, j] * phi_predictor_i[j - 1]
                                                               - d_m[i - 1, j] * phi_predictor_i[j + 1] ) / ( 1 - c_m[i - 1, j] + d_p[i - 1, j] - d_m[i - 1, j]);
                    phi_predictor_i[j] = ( phi[i - 1, j, n] - 0.5 * ( phi1[i - 1, j, n + 1] - phi[i - 1, j, n] - phi1[i - 1, j, n] + phi[i - 1, j, n - 1] ) 
                        - c_m[i - 1, j] * ( phi2[i, j, n + 1] - 0.5 * ( phi1[i - 1, j, n + 1] - phi2[i, j, n + 1] - phi1[i, j, n + 1] + phi[i + 1, j, n + 1]) ) 
                        + d_p[i - 1, j] * ( phi_predictor_i[j - 1] - 0.5 * ( phi1[i - 1, j, n + 1] - phi_predictor_i[j - 1] - phi1[i - 1, j - 1, n + 1] + phi_predictor_i[j - 2] ) ) 
                        - d_m[i - 1, j] * ( phi_predictor_i[j + 1] - 0.5 * ( phi1[i - 1, j, n + 1] - phi_predictor_i[j + 1] - phi1[i - 1, j + 1, n + 1] + phi_predictor_i[j + 2] ) ) ) / ( 1 - c_m[i - 1, j] + d_p[i - 1, j] - d_m[i - 1, j] );
                end

                # Compute first order predictor in y direction
                if sweep % 2 == 1 # moving from bottom to top
                    phi1[i, j + 1, n] = ( phi[i, j + 1, n - 1] + c_p[i, j + 1] * phi[i - 1, j + 1, n] 
                                                               - c_m[i, j + 1] * phi[i + 1, j + 1, n]
                                                               + d_p[i, j + 1] * phi[i, j, n] ) / ( 1 + c_p[i, j + 1] - c_m[i, j + 1] + d_p[i, j + 1] );
                    phi1[i, j + 1, n + 1] = ( phi[i, j + 1, n] + c_p[i, j + 1] * phi[i - 1, j + 1, n + 1] 
                                                               - c_m[i, j + 1] * phi[i + 1, j + 1, n + 1]
                                                               + d_p[i, j + 1] * phi[i, j, n + 1] ) / ( 1 + c_p[i, j + 1] - c_m[i, j + 1] + d_p[i, j + 1] );
                    phi_j = ( phi[i, j + 1, n] - 1 / 2 * ( phi1[i, j + 1, n + 1] - phi[i, j + 1, n] - phi1[i, j + 1, n] + phi[i, j + 1, n - 1] )
                                                            + c_p[i, j + 1] * ( phi_predictor_j[j][2] - 1 / 2 * ( phi1[i, j + 1, n + 1] - phi_predictor_j[j][2] - phi1[i - 1, j + 1, n + 1] + phi_predictor_j[j][1] ) )
                                                            - c_m[i, j + 1] * ( phi_predictor_j[j][2] - 1 / 2 * ( phi1[i, j + 1, n + 1] - phi_predictor_j[j][2] - phi1[i + 1, j + 1, n + 1] + phi_predictor_j[j][1] ) )
                                                            + d_p[i, j + 1] * ( phi2[i, j, n + 1]  - 1 / 2 * ( phi1[i, j + 1, n + 1] - phi2[i, j, n + 1] - phi1[i, j, n + 1] + phi[i, j - 1, n + 1] ) ) ) / (1 + c_p[i, j + 1] - c_m[i, j + 1] + d_p[i, j + 1] );
                else # moving from top to bottom
                    phi1[i, j - 1, n] = ( phi[i, j - 1, n - 1] + c_p[i, j - 1] * phi[i - 1, j - 1, n] 
                                                            - c_m[i, j - 1] * phi[i + 1, j - 1, n]
                                                            - d_m[i, j - 1] * phi[i, j, n] ) / ( 1 + c_p[i, j - 1] - c_m[i, j - 1] - d_m[i, j - 1] );
                    phi1[i, j - 1, n + 1] = ( phi[i, j - 1, n] + c_p[i, j - 1] * phi[i - 1, j - 1, n + 1]
                                                            - c_m[i, j - 1] * phi[i + 1, j - 1, n + 1]
                                                            - d_m[i, j - 1] * phi[i, j, n + 1] ) / ( 1 + c_p[i, j - 1] - c_m[i, j - 1] - d_m[i, j - 1] ); 
                    phi_j = ( phi[i, j - 1, n] - 1 / 2 * ( phi1[i, j - 1, n + 1] - phi[i, j - 1, n] - phi1[i, j - 1, n] + phi[i, j - 1, n - 1] )
                                                            + c_p[i, j - 1] * ( phi_predictor_j[j][2] - 1 / 2 * ( phi1[i, j - 1, n + 1] - phi_predictor_j[j][2] - phi1[i - 1, j - 1, n + 1] + phi_predictor_j[j][1] ) )
                                                            - c_m[i, j - 1] * ( phi_predictor_j[j][2] - 1 / 2 * ( phi1[i, j - 1, n + 1] - phi_predictor_j[j][2] - phi1[i + 1, j - 1, n + 1] + phi_predictor_j[j][1] ) )
                                                            - d_m[i, j - 1] * ( phi2[i, j, n + 1]  - 1 / 2 * ( phi1[i, j - 1, n + 1] - phi2[i, j, n + 1] - phi1[i, j, n + 1] + phi[i, j + 1, n + 1] ) ) ) / (1 + c_p[i, j - 1] - c_m[i, j - 1] - d_m[i, j - 1] );
                end

                push!(phi_predictor_j[j], phi_j);
                    
                # Compute second order predictor for n + 2
                phi1[i - 1, j, n + 2] = ( phi[i - 1, j, n + 1] + c_p[i - 1, j] * phi2[i - 2, j, n + 2] 
                                                                + d_p[i - 1, j] * phi2[i - 1, j - 1, n + 2]
                                                                - d_m[i - 1, j] * phi2[i - 1, j + 1, n + 2] ) / ( 1 + c_p[i - 1, j] + d_p[i - 1, j] - d_m[i - 1, j] );
                phi1[i + 1, j, n + 2] = ( phi[i + 1, j, n + 1] - c_m[i + 1, j] * phi2[i + 2, j, n + 2] 
                                                                + d_p[i + 1, j] * phi2[i + 1, j - 1, n + 2]
                                                                - d_m[i + 1, j] * phi2[i + 1, j + 1, n + 2] ) / ( 1 - c_m[i + 1, j] + d_p[i + 1, j] - d_m[i + 1, j] );
                phi1[i, j - 1, n + 2] = ( phi[i, j - 1, n + 1] + c_p[i, j - 1] * phi2[i - 1, j - 1, n + 2] 
                                                                - c_m[i, j - 1] * phi2[i + 1, j - 1, n + 2]
                                                                + d_p[i, j - 1] * phi2[i, j - 2, n + 2] ) / ( 1 + c_p[i, j - 1] - c_m[i, j - 1] + d_p[i, j - 1] );
                phi1[i, j + 1, n + 2] = ( phi[i, j + 1, n + 1] + c_p[i, j + 1] * phi2[i - 1, j + 1, n + 2] 
                                                                - c_m[i, j + 1] * phi2[i + 1, j + 1, n + 2]
                                                                - d_m[i, j + 1] * phi2[i, j + 2, n + 2] ) / ( 1 + c_p[i, j + 1] - c_m[i, j + 1] - d_m[i, j + 1] );
                phi1[i, j, n + 2] = ( phi2[i, j, n + 1] + c_p[i, j] * phi2[i - 1, j, n + 2] 
                                                        + d_p[i, j] * phi2[i, j - 1, n + 2] 
                                                        - c_m[i, j] * phi2[i + 1, j, n + 2]
                                                        - d_m[i, j] * phi2[i, j + 1, n + 2] ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );

                phi2[i, j, n + 2] =  ( phi2[i, j, n + 1] - 0.5 * ( phi1[i, j, n + 2] - phi2[i, j, n + 1] - phi1[i, j, n + 1] + phi[i, j, n] ) 
                                    + c_p[i, j] * ( phi2[i - 1, j, n + 2] - 0.5 * ( phi1[i, j, n + 2] - phi2[i - 1, j, n + 2] - phi1[i - 1, j, n + 2] + phi2[i - 2, j, n + 2] ) )
                                    - c_m[i, j] * ( phi2[i + 1, j, n + 2] - 0.5 * ( phi1[i, j, n + 2] - phi2[i + 1, j, n + 2] - phi1[i + 1, j, n + 2] + phi2[i + 2, j, n + 2] ) )
                                    + d_p[i, j] * ( phi2[i, j - 1, n + 2] - 0.5 * ( phi1[i, j, n + 2] - phi2[i, j - 1, n + 2] - phi1[i, j - 1, n + 2] + phi2[i, j - 2, n + 2] ) )
                                    - d_m[i, j] * ( phi2[i, j + 1, n + 2] - 0.5 * ( phi1[i, j, n + 2] - phi2[i, j + 1, n + 2] - phi1[i, j + 1, n + 2] + phi2[i, j + 2, n + 2] ) ) ) / ( 1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j] );

                for k = 1:K # Multiple correction iterations

                    # phi2_i_old_p = phi2[i, j, n + 1];
                    # phi2_j_old_p = phi2[i, j, n + 1];

                    # SECOND ITERATION
                    rd_n = phi2[i, j, n + 2] - phi2[i, j, n + 1] - phi2_old + phi[i, j, n];
                    ru_n = phi2[i, j, n + 1] - phi[i, j, n] - phi2[i, j, n] + phi[i, j, n - 1] 

                    rd_i_p = phi_predictor_i[j] - phi2[i, j, n + 1] - phi2_i_old_p + phi[i - 1, j, n + 1];
                    ru_i_p = phi2[i, j, n + 1] - phi[i - 1, j, n + 1] - phi2[i - 1, j, n + 1] + phi[i - 2, j, n + 1];
                    rd_i_m = phi_predictor_i[j] - phi2[i, j, n + 1] - phi2_i_old_p + phi[i + 1, j, n + 1];
                    ru_i_m = phi2[i, j, n + 1] - phi[i + 1, j, n + 1] - phi2[i + 1, j, n + 1] + phi[i + 2, j, n + 1];

                    rd_j_p = phi_predictor_j[j][2] - phi2[i, j, n + 1] - phi2_j_old_p + phi[i, j - 1, n + 1];
                    ru_j_p = phi2[i, j, n + 1] - phi[i, j - 1, n + 1] - phi2[i, j - 1, n + 1] + phi[i, j - 2, n + 1];
                    rd_j_m = phi_predictor_j[j][2] - phi2[i, j, n + 1] - phi2_j_old_p + phi[i, j + 1, n + 1];
                    ru_j_m = phi2[i, j, n + 1] - phi[i, j + 1, n + 1] - phi2[i, j + 1, n + 1] + phi[i, j + 2, n + 1];

                    ru_i = ifelse( c_p[i, j] > 0, ru_i_p, ru_i_m);
                    rd_i = ifelse( c_p[i, j] > 0, rd_i_p, rd_i_m);
                    ru_j = ifelse( d_p[i, j] > 0, ru_j_p, ru_j_m);
                    rd_j = ifelse( d_p[i, j] > 0, rd_j_p, rd_j_m);

                    if (sweep > 2 && c_p[i, j] > 0) || (sweep % 2 == 0 && d_p[i, j] > 0)
                        continue
                    end

                    ω1_i[i, j] = ifelse( abs(ru_i) <= abs(rd_i), 0, 0)# * Int(!third_order)# * ifelse( ru_i * rd_i > 0, 1, 0)
                    ω1_j[i, j] = ifelse( abs(ru_j) <= abs(rd_j), 0, 0)# * Int(!third_order)# * ifelse( ru_j * rd_j > 0, 1, 0)
                    α1[i, j] = ifelse( abs(ru_n) <= abs(rd_n), 0, 0)# * Int(!third_order)# * ifelse( ru_n * rd_n > 0, 1, 0)

                    ω2_i[i, j] = ( 1 - ω1_i[i, j] )# * ifelse( ru_i * rd_i > 0, 1, 0);
                    ω2_j[i, j] = ( 1 - ω1_j[i, j] )# * ifelse( ru_j * rd_j > 0, 1, 0);
                    α2[i, j] = ( 1 - α1[i, j] )# * ifelse( ru_n * rd_n > 0, 1, 0);     

                    phi[i, j, n + 1] = ( phi[i, j, n] - α1[i, j] / 2 * ru_n - α2[i, j] / 2 * rd_n
                        + c_p[i, j] * ( phi[i - 1, j, n + 1] - ω1_i[i, j] / 2 *  ru_i_p - ω2_i[i, j] / 2 * rd_i_p ) 
                        - c_m[i, j] * ( phi[i + 1, j, n + 1] - ω1_i[i, j] / 2 *  ru_i_m - ω2_i[i, j] / 2 * rd_i_m )
                        + d_p[i, j] * ( phi[i, j - 1, n + 1] - ω1_j[i, j] / 2 *  ru_j_p - ω2_j[i, j] / 2 * rd_j_p )
                        - d_m[i, j] * ( phi[i, j + 1, n + 1] - ω1_j[i, j] / 2 *  ru_j_m - ω2_j[i, j] / 2 * rd_j_m ) ) / (1 + c_p[i, j] + d_p[i, j] - c_m[i, j] - d_m[i, j]);

                    phi2_old = phi2[i, j, n + 1];
                    phi2_j_old_p = phi2[i, j, n + 1];
                    phi2_i_old_p = phi2[i, j, n + 1];
                    phi2[i, j, n + 1] = phi[i, j, n + 1];

                end
            #     phi[N + 3, j, n + 1] = 3 * phi[N + 2, j, n + 1] - 3 * phi[N + 1, j, n + 1] + phi[N, j, n + 1];
            #     phi2[N + 3, j, n + 1] = 3 * phi2[N + 2, j, n + 1] - 3 * phi2[N + 1, j, n + 1] + phi2[N, j, n + 1];
            #     phi1[N + 3, j, n + 1] = 3 * phi1[N + 2, j, n + 1] - 3 * phi1[N + 1, j, n + 1] + phi1[N, j, n + 1];
            end
            # phi[i, N + 3, n + 1] = 3 * phi[i, N + 2, n + 1] - 3 * phi[i, N + 1, n + 1] + phi[i, N, n + 1];
            # phi2[i, N + 3, n + 1] = 3 * phi2[i, N + 2, n + 1] - 3 * phi2[i, N + 1, n + 1] + phi2[i, N, n + 1];
            # phi1[i, N + 3, n + 1] = 3 * phi1[i, N + 2, n + 1] - 3 * phi1[i, N + 1, n + 1] + phi1[i, N, n + 1];
        end
        # phi_first_order[N + 3, :, n + 1] = 3 * phi_first_order[N + 2, :, n + 1] - 3 * phi_first_order[N + 1, :, n + 1] + phi_first_order[N, :, n + 1];
        # phi_first_order[:, N + 3, n + 1] = 3 * phi_first_order[:, N + 2, n + 1] - 3 * phi_first_order[:, N + 1, n + 1] + phi_first_order[:, N, n + 1];
    end
end

end

# Print error
Error_t_h = tau * h^2 * sum(sum(abs.(phi[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 2:Ntau+2)
println("Error t*h: ", Error_t_h)
Error_t_h_1 = tau * h^2 * sum(sum(abs.(phi_first_order[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 2:Ntau+2)
println("Error t*h first order: ", Error_t_h_1)

# Error_t_h = h * sum(abs(phi[i, end-1] - phi_exact.(x[i], t[end-1])) for i in 2:Nx+2)
# println("Error h: ", Error_t_h)

# Load the last error
last_error = load_last_error()
if last_error !== nothing
    println("Order: ", log(2, last_error / Error_t_h))
end 
# Save the last error
save_last_error(Error_t_h)
println("=============================")

# CSV.write("phi.csv", DataFrame(phi2, :auto))

# Compute gradient of the solution in the last time step
d_phi_x = zeros(N + 3, N + 3);
d_phi_y = zeros(N + 3, N + 3);

for i = 5:length(x)-5
for j = 5:length(y)-5
   d_phi_x[i, j] = (phi[i + 1, j, end-1] - phi[i, j, end-1]) / h;
   d_phi_y[i, j] = (phi[i, j + 1, end-1] - phi[i, j, end-1]) / h;
end
end

println("minimum derivative x: ", minimum(d_phi_x))
println("maximum derivative x: ", maximum(d_phi_x))
println("minimum derivative y: ", minimum(d_phi_y))
println("maximum derivative y: ", maximum(d_phi_y))

# Plot of the result at the final time together with the exact solution
trace3 = contour(x = x, y = y, z = phi_exact.(X, Y, t[end - 1])', mode = "lines", name = "Exact", showscale=false, contours_coloring="lines", colorscale="Greys", line_width=2)
trace0 = contour(x = x, y = y, z = phi_0.(X, Y)', mode = "lines", name = "Initial Condition", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1 )
trace2 = contour(x = x, y = y, z = phi[:, :, end - 1]', mode = "lines", name = "First order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=1)
trace1 = contour(x = x, y = y, z = phi_first_order[:, :, end - 1]', mode = "lines", name = "First order", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_dash="dash", line_width=1)

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
# plot_phi = plot([ trace2, trace1,trace5, trace4, trace3], layout)
plot_phi = plot([ trace3, trace2, trace1 ], layout)

plot_phi

# For plot purposes replace values that are delta far way from 1 and -1 by 0
# d_phi_x = map(x -> abs(x) < 0.99 ? 0 : x, d_phi_x)
# d_phi_y = map(x -> abs(x) < 0.99 ? 0 : x, d_phi_y)

# Plot derivative
trace1_d_x = contour(x = x, y = y, z = d_phi_x[:, :], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=2, ncontours = 100)
trace1_d_y = contour(x = x, y = y, z = d_phi_y[:, :], name = "Implicit", showscale=false, colorscale = "Plasma", contours_coloring="lines", line_width=2, ncontours = 100)

plot_phi_d_x = plot([trace1_d_x])
plot_phi_d_y = plot([trace1_d_y])

p = [plot_phi; plot_phi_d_x plot_phi_d_y]
relayout!(p, width=800, height=1000)
p