# TVD scheme for linear advection equation

using LinearAlgebra
using PlotlyJS
using CSV 
using DataFrames

include("InitialFunctions.jl")
include("ExactSolutions.jl")

## Definition of basic parameters

# Level of refinement
level = 0;

# Courant number
# C = 4;

# Grid settings
xL = - 1 * π / 2
xR = 1 * 3 * π / 2
Nx = 100 * 2^level
h = (xR - xL) / Nx

# Velocity
u(x) = 1 + 3/4 * cos(x)

# Initial condition
phi_0(x) = asin( sin(x + π/2) ) * 2 / π;

# Exact solution
phi_exact(x, t) = cosVelocityNonSmooth(x, t);              

## Comptutation

# Grid initialization
x = range(xL, xR, length = Nx + 1)

# Time
T = 0.2 * π
# tau = C * h / u
Ntau = 100 * 2^level
tau = T / Ntau
# Ntau = 1

c = zeros(Nx+1,1) .+ u.(x) * tau / h

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
        r_downwind_i_minus = - phi_predictor_i[i, n] + (c[i] > 0) * phi_predictor_i[i - 1, n + 1] + (c[i] < 0) * phi_predictor_i[i + 1, n + 1];
        r_upwind_i_minus = - (c[i] > 0) * phi[i - 1, n] - (c[i] < 0) * phi[i + 1, n] + (c[i] > 0) * phi_left + (c[i] < 0) * phi_right;

        r_downwind_i = - phi_predictor_i[i, n + 1] + (c[i] > 0) * phi_predictor_i[i + 1, n] + (c[i] < 0) * phi_predictor_i[i - 1, n];
        r_upwind_i = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1] - (c[i] < 0) * phi[i + 1, n + 1];

        r_downwind_n_old = phi_predictor_n[i, n] - (c[i] > 0) * phi_predictor_n[i - 1, n + 1] - (c[i] < 0) * phi_predictor_n[i + 1, n + 1];
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
        phi[i, n + 1] = ( phi[i, n] +  max( 0, min( 1, abs(c[i]) - 1/2 ) ) * ( (c[i] > 0) * 0.5/(abs(c[i])+eps) * (c[i] - c[i-1]) * phi[i, n] + (c[i] < 0) * 0.5/(abs(c[i])+eps) * (c[i+1] - c[i]) * phi[i, n] ) 
                                    + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] + abs(c[i]) * (c[i] < 0) * phi[i + 1, n + 1] 
                                    - 0.5 * abs(c[i]) * min( 1, max( 0, 3/2 - abs(c[i]) ) ) * ( s[i, n + 1] * (r_downwind_i_minus + r_downwind_i) + (1 - s[i, n + 1]) * (r_upwind_i + r_upwind_i_minus ) ) 
                                           - 0.5 * max( 0, min( 1, abs(c[i]) - 1/2 ) ) * ( p[i, n + 1] * (r_downwind_n_old + r_downwind_n_new) + (1 - p[i, n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + abs(c[i]) + max( 0, min( 1, abs(c[i]) - 1/2 ) ) * 0.5 / (abs(c[i])+eps) * ( (c[i]>0) * (c[i] - c[i-1]) + (c[i]<0) * (c[i+1]-c[i]) ) );


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
        r_downwind_i_minus = - phi_predictor_i[i, n] + (c[i] > 0) * phi_predictor_i[i - 1, n + 1] + (c[i] < 0) * phi_predictor_i[i + 1, n + 1];
        r_upwind_i_minus = - (c[i] > 0) * phi[i - 1, n] - (c[i] < 0) * phi[i + 1, n] + (c[i] > 0) * phi_left + (c[i] < 0) * phi_right;

        r_downwind_i = - phi_predictor_i[i, n + 1] + (c[i] > 0) * phi_predictor_i[i + 1, n] + (c[i] < 0) * phi_predictor_i[i - 1, n];
        r_upwind_i = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1] - (c[i] < 0) * phi[i + 1, n + 1];

        r_downwind_n_old = phi_predictor_n[i, n] - (c[i] > 0) * phi_predictor_n[i - 1, n + 1] - (c[i] < 0) * phi_predictor_n[i + 1, n + 1];
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
        phi[i, n + 1] = ( phi[i, n] +  max( 0, min( 1, abs(c[i]) - 1/2 ) ) * ( (c[i] > 0) * 0.5/(abs(c[i])+eps) * (c[i] - c[i-1]) * phi[i, n] + (c[i] < 0) * 0.5/(abs(c[i])+eps) * (c[i+1] - c[i]) * phi[i, n] ) 
                                    + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] + abs(c[i]) * (c[i] < 0) * phi[i + 1, n + 1] 
                                    - 0.5 * abs(c[i]) * min( 1, max( 0, 3/2 - abs(c[i]) ) ) * ( s[i, n + 1] * (r_downwind_i_minus + r_downwind_i) + (1 - s[i, n + 1]) * (r_upwind_i + r_upwind_i_minus ) ) 
                                           - 0.5 * max( 0, min( 1, abs(c[i]) - 1/2 ) ) * ( p[i, n + 1] * (r_downwind_n_old + r_downwind_n_new) + (1 - p[i, n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + abs(c[i]) + max( 0, min( 1, abs(c[i]) - 1/2 ) ) * 0.5 / (abs(c[i])+eps) * ( (c[i]>0) * (c[i] - c[i-1]) + (c[i]<0) * (c[i+1]-c[i]) ) );


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

# Read the data from the file
# df_8 = CSV.File("c_8.csv") |> DataFrame
# phi_8 = Matrix(df_8)

# df_1 = CSV.File("c_1.csv") |> DataFrame
# phi_1 = Matrix(df_1)

# df_16 = CSV.File("c_1.6.csv") |> DataFrame
# phi_16 = Matrix(df_16)

# df_08 = CSV.File("c_0.8.csv") |> DataFrame
# phi_08 = Matrix(df_08)

# Plot of the result at the final time together with the exact solution
# trace1 = scatter(x = x, y = phi_8[:,end], mode = "lines", name = "Compact TVD C = 8", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact.(x, Ntau*tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace1 = scatter(x = x, y = phi_0.(x), mode = "lines", name = "Initial Condition", line=attr(color="black", width=1, dash = "dash") )
# trace4 = scatter(x = x, y = phi_1[:,end], mode = "lines", name = "Compact TVD C = 1", line=attr(color="green", width=2))
# trace5 = scatter(x = x, y = phi_16[:,end], mode = "lines", name = "Compact TVD C = 1.6", line=attr(color="orange", width=2))
trace3 = scatter(x = x, y = phi[:,end], mode = "lines", name = "Compact TVD", line=attr(color="firebrick", width=2))


layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
# plot_phi = plot([ trace2, trace1,trace5, trace4, trace3], layout)
plot_phi = plot([ trace1, trace2, trace3], layout)

plot_phi

# CSV.write("c_1.6.csv", DataFrame(phi,:auto))

# Plot of the numerical derivative of the solution and the exact solution at the final time
trace1_d = scatter(x = x, y = diff(phi[:, end]) / h, mode = "lines", name = "Compact sol. gradient")
trace2_d = scatter(x = x, y = diff(phi_exact.(x, Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
trace3_d = scatter(x = x, y = diff(phi_first_order[:, end]) / h, mode = "lines", name = "First order sol. gradient")

layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

plod_d_phi = plot([trace1_d, trace2_d, trace3_d], layout_d)

p = [plot_phi; plod_d_phi]
relayout!(p, width = 1000, height = 500)
p