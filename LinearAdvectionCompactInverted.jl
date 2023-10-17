# Inverted linear advection equation with compact finite differences

using LinearAlgebra
using PlotlyJS
using CSV
using DataFrames

include("InitialFunctions.jl")

## Definition of basic parameters

# Level of refinement
level = 5;

# Courant number
C = 5

# Grid settings
xL = -pi/2
xR = 3*pi/2
Nx = 100 * 2^level
h = (xR - xL) / Nx
x = range(xL, xR, length = Nx + 1)

# Velocity
# u = 1.0
u(x) = 2 + cos.(x);

# Time settings
T = 2 * pi / sqrt(3)
# tau = C * h / u
Ntau = 10 * 2^level
tau = T / Ntau

# c = u * tau / h
c = zeros(Nx+1,1) .+ u.(x) * tau / h

# Initial condition
# phi_0(x) = piecewiseLinear(x);
# phi_0(x) = makePeriodic(allInOne,-1,1)(x);
# phi_0(x) = piecewiseConstant(x);
# phi_0(x) = makePeriodic(continuesMix,-1,1)(x);
phi_0(x) = cos(x);
# phi_0(x) = makePeriodic(nonSmooth,-1,1)(x - 0.5);

# Exact solution
# phi_exact(x, t) = phi_0(x - u * t);
phi_exact(x, t) = cos(2*atan(sqrt(3)*tan((sqrt(3).*(t - (2*atan(tan(x/2.)./sqrt(3)))./sqrt(3)))/2.)))
## Comptutation

# Grid initialization
phi = zeros(Nx + 1, Ntau + 1);
phi_predictor = zeros(Nx + 1, Ntau + 1); # predictor in time n+1
phi_predictor_n2 = zeros(Nx + 1, Ntau + 1); # predictor in time n+2
phi_first_order = zeros(Nx + 1, Ntau + 1);

# Initial condition
phi[:, 1] = phi_0.(x);
phi_predictor[:, 1] = phi_0.(x);
phi_predictor_n2[:, 1] = phi_0.(x);
phi_first_order[:, 1] = phi_0.(x);

# Boundary conditions
phi[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[1, :] = phi_exact.(x[1], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n2[1, :] = phi_exact.(x[1], range(tau, (Ntau + 1) * tau, length = Ntau + 1));

phi[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_first_order[end, :] = phi_exact.(x[end], range(0, Ntau * tau, length = Ntau + 1));
phi_predictor_n2[end, :] = phi_exact.(x[end], range(tau, (Ntau + 1) * tau, length = Ntau + 1));

# Ghost point on the time -1
ghost_point_time = phi_exact.(x, -tau);

# ENO parameters
s = zeros(Nx + 1, Ntau + 1);

eps = 1e-8;

# Time loop
for n = 1:Ntau

    if n > 1
        phi_old = phi[:, n-1];
    else
        phi_old = ghost_point_time;
    end

    # Space loop
    for i = 2:1:Nx+1

        # First order solution
        phi_first_order[i, n + 1] = ( phi_first_order[i, n] + abs(c[i]) * (c[i] > 0) * phi_first_order[i - 1, n + 1] ) / ( 1 + abs(c[i]) );

        # Predictor
        phi_predictor[i, n + 1] = ( phi[i, n] + abs(c[i]) * (c[i] > 0) * phi_predictor[i - 1, n + 1] ) / ( 1 + abs(c[i]) );

        # Corrector
        r_downwind_n_old = phi[i, n] - (c[i] > 0) * phi[i - 1, n + 1];
        r_upwind_n_old = - (c[i] > 0) *  phi[i - 1, n] + phi_old[i];

        r_downwind_n_new = - phi_predictor[i, n + 1] + (c[i] > 0) * phi_predictor_n2[i - 1, n + 1];
        r_upwind_n_new = (c[i] > 0) * phi[i - 1, n + 1];

        # ENO parameter 
        if abs(r_downwind_n_new + r_downwind_n_old) <= abs(r_upwind_n_new + r_upwind_n_old)
            s[i,n+1] = 1
        else
            s[i,n+1] = 0
        end

        phi[i, n + 1] = ( phi[i, n] + 0.5/c[i] * (c[i] - c[i-1]) * phi[i, n] + abs(c[i]) * (c[i] > 0) * phi[i - 1, n + 1] - 0.5 * ( s[i,n + 1] * (r_downwind_n_old + r_downwind_n_new) 
                                                                    + (1-s[i,n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + abs(c[i]) + 0.5 / c[i] * (c[i] - c[i-1]) );
        
        
        # Predictor for next time step
        phi_predictor_n2[i, n + 1] = ( phi[i, n + 1] + abs(c[i]) * (c[i] > 0) * phi_predictor_n2[i - 1, n + 1] ) / ( 1 + abs(c[i]) );

    end

    # for i = Nx:-1:2

    #     # First order solution
    #     phi_first_order[i, n + 1] = ( phi_first_order[i, n] + abs(c) * (c > 0) * phi_first_order[i - 1, n + 1] 
    #                                                         + abs(c) * (c < 0) * phi_first_order[i + 1, n + 1] ) / ( 1 + abs(c) );

    #     # Predictor
    #     phi_predictor[i, n + 1] = ( phi[i, n] + abs(c) * (c > 0) * phi_predictor[i - 1, n + 1]
    #                                           + abs(c) * (c < 0) * phi_predictor[i + 1, n + 1]  ) / ( 1 + abs(c) );

    #     # Corrector
    #     r_downwind_n_old = phi[i, n] - (c > 0) * phi[i - 1, n + 1] - (c < 0) * phi[i + 1, n + 1];
    #     r_upwind_n_old = - (c > 0) *  phi[i - 1, n] - (c < 0) * phi[i + 1, n] + phi_old[i];

    #     r_downwind_n_new = - phi_predictor[i, n + 1] + (c > 0) * phi_predictor_n2[i - 1, n + 1] + (c < 0) * phi_predictor_n2[i + 1, n + 1];
    #     r_upwind_n_new = (c > 0) * phi[i - 1, n + 1] + (c < 0) * phi[i + 1, n + 1] - phi[i, n];

    #     # ENO parameter 
    #     if abs(r_downwind_n_new + r_downwind_n_old) <= abs(r_upwind_n_new + r_upwind_n_old)
    #         s[i,n+1] = 1
    #     else
    #         s[i,n+1] = 0
    #     end

    #     phi[i, n + 1] = ( phi[i, n] + abs(c) * (c > 0) * phi[i - 1, n + 1] + abs(c) * (c < 0) * phi[i + 1, n + 1] - 0.5 * ( s[i,n + 1] * (r_downwind_n_old + r_downwind_n_new) + (1-s[i,n + 1]) * (r_upwind_n_new + r_upwind_n_old ) ) ) / ( 1 + abs(c) );
        
        
    #     # Predictor for next time step
    #     phi_predictor_n2[i, n + 1] = ( phi[i, n + 1] + abs(c) * (c > 0) * phi_predictor_n2[i - 1, n + 1] 
    #                                                  + abs(c) * (c < 0) * phi_predictor_n2[i + 1, n + 1] ) / ( 1 + abs(c) );

    # end
end


# Print the error
println("Error L2: ", norm(phi[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf: ", norm(phi[:, end] - phi_exact.(x, Ntau * tau), Inf) * h)

# Print first order error
println("Error L2 first order: ", norm(phi_first_order[:,end] - phi_exact.(x, Ntau * tau), 2) * h)
println("Error L_inf firts order: ", norm(phi_first_order[:, end] - phi_exact.(x, Ntau * tau), Inf)* h)

# Read the data from the file
# df = CSV.File("data.csv") |> DataFrame
# phi_normal = Matrix(df)

# Plot of the result at the final time together with the exact solution
trace1 = scatter(x = x, y = phi[:, end], mode = "lines", name = "Inverted Compact TVD", line=attr(color="firebrick", width=2))
trace2 = scatter(x = x, y = phi_exact.(x, Ntau * tau), mode = "lines", name = "Exact", line=attr(color="black", width=2) )
trace3 = scatter(x = x, y = phi_first_order[:, end], mode = "lines", name = "First-order", line=attr(color="royalblue", width=2, dash="dash"))
# trace4 = scatter(x = x, y = phi_normal[:, end], mode = "lines", name = "Normal Compact TVD", line=attr(color="green", width=2))


layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), yaxis=attr(zerolinecolor="gray", gridcolor="lightgray",tickfont=attr(size=20)))
plot_phi = plot([trace1, trace2, trace3], layout)

plot_phi

# Plot of the numerical derivative of the solution and the exact solution at the final time
# trace1_d = scatter(x = x, y = diff(phi[:, end]) / h, mode = "lines", name = "Compact sol. gradient")
# trace2_d = scatter(x = x, y = diff(phi_exact.(x, Ntau * tau)) / h, mode = "lines", name = "Exact sol. gradient")
# trace3_d = scatter(x = x, y = diff(phi_first_order[:, end]) / h, mode = "lines", name = "First order sol. gradient")

# layout_d = Layout(title = "Linear advection equation - Gradient", xaxis_title = "x", yaxis_title = "Dphi/Dx")

# plod_d_phi = plot([trace1_d, trace2_d, trace3_d], layout_d)

# p = [plot_phi; plod_d_phi]
# relayout!(p, width = 1000, height = 500)
# p