# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS

include("InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 4;

# Courant number
C = 4;

# Grid settings
xL = -1 * pi/2
xR = 7 * pi / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
# u = 1.0
u(x) = sin(x) 

# Initial condition
# phi_0(x) = makePeriodic(allInOne,-1,1)(x);
phi_0(x) = sin(x);

# Exact solution
# phi_exact(x, t) = phi_0(x - u * t);
phi_exact(x, t) = sin.(2 * atan.( tan.(x / 2) * exp.(-t) ));

## Comptutation

# Grid initialization
x = range(xL, xR, length = Nx + 1)

# Time
T = 2.0;
tau = C * h / maximum(abs.(u.(x)))
Ntau = Int(Nx / 100)
tau = T / Ntau
# Ntau = 1

c = u.(x) * tau / h

phi = zeros(Nx + 1, Ntau + 1);
phi_predictor_i = zeros(Nx + 1, Ntau + 1); # predictor in time n+1
phi_predictor_n = zeros(Nx + 1, Ntau + 1); # predictor in time n
phi_predictor_n2 = zeros(Nx + 1, Ntau + 1); # predictor in time n+2
phi_first_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor_i[:, 1] = phi_0.(x);
phi_predictor_n[:, 1] = phi_0.(x);
phi_predictor_n2[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_i[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n2[1, :] = phi_exact.(x[1], range(tau, (Ntau + 1) * tau, length = Ntau + 1));

phi[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_i[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n2[end, :] = phi_exact.(x[end], range(tau, (Ntau + 1) * tau, length = Ntau + 1));


# Ghost point on the right side 
ghost_point_right = phi_exact.(xR + h, range(tau, (Ntau+1) * tau, length = Ntau + 1));
# Ghost point on the left side 
ghost_point_left = phi_exact.(xL - h, range(tau, (Ntau+1) * tau, length = Ntau + 1));
# Ghost point on the time -1
ghost_point_time = phi_exact.(x, -tau);

# ENO parameters
s = zeros(Nx + 1, Ntau + 1); # normal logic
p = zeros(Nx + 1, Ntau + 1); # inverted logic
eps = 1e-8;

# Time Loop
for n = 1:Ntau

    if n > 1
        phi_old = phi[:, n-1];
    else
        phi_old = ghost_point_time;
    end

    # Space loop
    for i = 2:1:Nx

        if i < Nx
            phi_right = phi[i + 2, n + 1];
        else
            phi_right = ghost_point_right[n];
        end

        if i > 2
            phi_left = phi[i - 2, n + 1];
        else
            phi_left = ghost_point_left[n];
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + abs(c[i]) * (c[i] > 0) * phi_first_order[i - 1, n + 1] 
                                                            + abs(c[i]) * (c[i] < 0) * phi_first_order[i + 1, n + 1] ) / ( 1 + abs(c[i]) );
        
        # Predictor
        phi_predictor_i[i, n + 1] = ( phi_predictor_i[i, n] + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] 
                                                            + abs(c[i]) * (c[i] < 0) * phi[i + 1, n + 1] ) / ( 1 + abs(c[i]) );
        phi_predictor_n[i, n + 1] =  ( phi[i, n] + abs(c[i]) * (c[i] > 0) * phi_predictor_n[i - 1, n + 1]
                                                 + abs(c[i]) * (c[i] < 0) * phi_predictor_n[i + 1, n + 1] ) / ( 1 + abs(c[i]) );

        # Corrector
        r_downwind_i_minus = - phi[i, n] + (c[i] > 0) * phi[i - 1, n + 1] + (c[i] < 0) * phi[i + 1, n + 1];
        r_upwind_i_minus = - (c[i] > 0) * phi[i - 1, n] - (c[i] < 0) * phi[i + 1, n] + (c[i] > 0) * phi_left + (c[i] < 0) * phi_right;

        r_downwind_i = - phi_predictor_i[i, n + 1] + (c[i] > 0) * phi[i + 1, n] + (c[i] < 0) * phi[i - 1, n];
        r_upwind_i = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1] - (c[i] < 0) * phi[i + 1, n + 1];

        r_downwind_n_old = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1] - (c[i] < 0) * phi[i + 1, n + 1];
        r_upwind_n_old = - (c[i] > 0) *  phi[i - 1, n] - (c[i] < 0) * phi[i + 1, n] + phi_old[i];

        r_downwind_n_new = - phi_predictor_n[i, n + 1] + (c[i] > 0) * phi_predictor_n2[i - 1, n + 1] + (c[i] < 0) * phi_predictor_n2[i + 1, n + 1];
        r_upwind_n_new = (c[i] > 0) * phi[i - 1, n + 1] + (c[i] < 0) * phi[i + 1, n + 1] - phi[i, n];


        # ENO parameters
        if abs(r_downwind_i_minus + r_downwind_i) <= abs(r_upwind_i_minus + r_upwind_i)
            s[i,n+1] = 1
        else
            s[i,n+1] = 0
        end

        if abs(r_downwind_n_new + r_downwind_n_old) <= abs(r_upwind_n_new + r_upwind_n_old)
            p[i,n+1] = 1
        else
            p[i,n+1] = 0
        end
        # Second order solution
        phi[i, n + 1] = ( phi[i, n] + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] + abs(c[i]) * (c[i] < 0) * phi[i + 1, n + 1] 
                                    - 0.5 * abs(c[i]) * min( 1, max( 0, 3/2 - c[i] ) ) * ( s[i, n + 1] * (r_downwind_i_minus + r_downwind_i) + (1 - s[i, n + 1]) * (r_upwind_i + r_upwind_i_minus ) ) 
                                           - 0.5 * max( 0, min( 1, c[i] - 1/2 ) ) * ( p[i, n + 1] * (r_downwind_n_old + r_downwind_n_new) + (1 - p[i, n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + abs(c[i]) );


        # Predictor for next time step
        phi_predictor_n2[i, n + 1] =  ( phi[i, n + 1] + abs(c[i]) * (c[i] > 0) * phi_predictor_n2[i - 1, n + 1] 
                                                      + abs(c[i]) * (c[i] < 0) * phi_predictor_n2[i + 1, n + 1] ) / ( 1 + abs(c[i]) );

    end

    for i = Nx:-1:2

        if i < Nx
            phi_right = phi[i + 2, n + 1];
        else
            phi_right = ghost_point_right[n];
        end

        if i > 2
            phi_left = phi[i - 2, n + 1];
        else
            phi_left = ghost_point_left[n];
        end

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + abs(c[i]) * (c[i] > 0) * phi_first_order[i - 1, n + 1] 
                                                            + abs(c[i]) * (c[i] < 0) * phi_first_order[i + 1, n + 1] ) / ( 1 + abs(c[i]) );
        
        # Predictor
        phi_predictor_i[i, n + 1] = ( phi_predictor_i[i, n] + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] 
                                                          + abs(c[i]) * (c[i] < 0) * phi[i + 1, n + 1] ) / ( 1 + abs(c[i]) );
        phi_predictor_n[i, n + 1] =  ( phi[i, n] + abs(c[i]) * (c[i] > 0) * phi_predictor_n[i - 1, n + 1]
                                                 + abs(c[i]) * (c[i] < 0) * phi_predictor_n[i + 1, n + 1]  ) / ( 1 + abs(c[i]) );

        # Corrector
        r_downwind_i_minus = - phi[i, n] + (c[i] > 0) * phi[i - 1, n + 1] + (c[i] < 0) * phi[i + 1, n + 1];
        r_upwind_i_minus = - (c[i] > 0) * phi[i - 1, n] - (c[i] < 0) * phi[i + 1, n] + (c[i] > 0) * phi_left + (c[i] < 0) * phi_right;

        r_downwind_i = - phi_predictor_i[i, n + 1] + (c[i] > 0) * phi[i + 1, n] + (c[i] < 0) * phi[i - 1, n];
        r_upwind_i = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1] - (c[i] < 0) * phi[i + 1, n + 1];

        r_downwind_n_old = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1] - (c[i] < 0) * phi[i + 1, n + 1];
        r_upwind_n_old = - (c[i] > 0) *  phi[i - 1, n] - (c[i] < 0) * phi[i + 1, n] + phi_old[i];

        r_downwind_n_new = - phi_predictor_i[i, n + 1] + (c[i] > 0) * phi_predictor_n2[i - 1, n + 1] + (c[i] < 0) * phi_predictor_n2[i + 1, n + 1];
        r_upwind_n_new = (c[i] > 0) * phi[i - 1, n + 1] + (c[i] < 0) * phi[i + 1, n + 1] - phi[i, n];


        # ENO parameters
        if abs(r_downwind_i_minus + r_downwind_i) <= abs(r_upwind_i_minus + r_upwind_i)
            s[i,n+1] = 1
        else
            s[i,n+1] = 0
        end
        if abs(r_downwind_n_new + r_downwind_n_old) <= abs(r_upwind_n_new + r_upwind_n_old)
            p[i,n+1] = 1
        else
            p[i,n+1] = 0
        end
        # Second order solution
        phi[i, n + 1] = ( phi[i, n] + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] + abs(c[i]) * (c[i] < 0) * phi[i + 1, n + 1] 
                                    - 0.5 * abs(c[i]) * min( 1, max( 0, 3/2 - c[i] ) ) * ( s[i, n + 1] * (r_downwind_i_minus + r_downwind_i) + (1 - s[i, n + 1]) * (r_upwind_i + r_upwind_i_minus ) ) 
                                           - 0.5 * max( 0, min( 1, c[i] - 1/2 ) ) * ( p[i, n + 1] * (r_downwind_n_old + r_downwind_n_new) + (1 - p[i, n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + abs(c[i]) );


        # Predictor for next time step
        phi_predictor_n2[i, n + 1] =  ( phi[i, n + 1] + abs(c[i]) * (c[i] > 0) * phi_predictor_n2[i - 1, n + 1] 
                                                      + abs(c[i]) * (c[i] < 0) * phi_predictor_n2[i + 1, n + 1] ) / ( 1 + abs(c[i]) );

    end
end

# Print error
Error_t_h = tau * h * sum(abs(phi[i, n] - phi_exact.(x[i], (n-1)*tau)) for n in 1:Ntau+1 for i in 1:Nx+1)
println("Error t*h: ", Error_t_h)
println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)

# Print first order error
println("Error L2 first order: ", norm(phi_first_order[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf firts order: ", norm(phi_first_order[:, end] - phi_exact.(x, Ntau * tau), Inf)* h)

println("=============================")

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Compact TVD", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace3 = scatter(x = x, y = phi_first_order[:, end], mode = "lines", name = "First-order", line=attr(color="royalblue", width=2))

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([trace1, trace2, trace3], layout)

plot_phi
