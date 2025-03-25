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
level = 3;

# Courant number
C = 5;

# Grid settings
xL = - 1 * π / 2
xR = 3 * π / 2
yL = xL
yR = xR
N = 40 * 2^level
h = (xR - xL) / N

# Constant velocity example
# u(x, y) = -1
# v(x, y) = 0
# phi_0(x , y) = cos.(y);
# phi_exact(x, y, t) = phi_0.(x - u(x,y) .* t, y - v(x,y) .* t);    

# Variable velocity example
offset = 0.5
u(x, y) = 0.0
v(x, y) = sin.(y - offset)
phi_0(x, y) = sin.(y - offset);
phi_exact(x, y, t) = sin.( 2 * atan.( exp.(-t) .* tan.( (y-offset) ./ 2) ) )

## Comptutation

# Grid initialization
x = range(xL - h, xR + h, length = N + 3)
y = range(yL - h, yR + h, length = N + 3)

X = repeat(x, 1, length(y))
Y = repeat(y, 1, length(x))'

# Time
T = 1.2
tau = C * h / maximum(max(abs.(u.(x, y)), abs.(v.(x, y))))
Ntau = Int(round(T / tau))

# Ntau = 1

t = range(0, (Ntau + 2) * tau, length = Ntau + 3)

c = zeros(N+3, N+3) .+ u.(x, y)' * tau / h
d = zeros(N+3, N+3) .+ v.(x, y)' * tau / h

# Set c plus and c minus
cp = max.(c, 0)
cm = min.(c, 0)
dp = max.(d, 0)
dm = min.(d, 0)

phi = zeros(N + 3, N + 3, Ntau + 3);
phi1 = zeros(N + 3, N + 3, Ntau + 3);
phi2 = zeros(N + 3, N + 3, Ntau + 3);
phi_first_order = zeros(N + 3, N + 3, Ntau + 3);

# Initial condition
phi[:, :, 1] = phi_0.(X, Y);
phi[:, :, 2] = phi_exact.(X, Y, tau);

# Boundary conditions right side - inflow
for n = 3:Ntau + 3
    # phi[end-1, :, n] = phi_exact.(x[end-1], y, t[n]);
    # phi[end, :, n] = phi_exact.(x[end], y, t[n]);
    phi[:, end-1, n] = phi_exact.(x, y[end-1], t[n]);
    phi[:, end, n] = phi_exact.(x, y[end], t[n]);
    # phi[1, :, n] = phi_exact.(x[1], y, t[n]);
    # phi[2, :, n] = phi_exact.(x[2], y, t[n]);
end

phi1 = copy(phi)
phi2 = copy(phi)
phi_first_order = copy(phi)

# Fast sweeping
sweep_cases = Dict(
    1 => (3:1:N+1, 3:1:N+1),
    2 => (3:1:N+1, N+1:-1:3),
    3 => (N+1:-1:3, 3:1:N+1),
    4 => (N+1:-1:3, N+1:-1:3)
)

@time begin

# Time Loop
for n = 2:Ntau + 1

    for k = 1:1
    for sweep in 1:4
        swI, swJ = sweep_cases[sweep]

        for i = swI
        for j = swJ

            # if (d[i, j] > 0 && d[i, j - 1] < 0 && sweep == 1)
            #     # Solve the subsistem
            #     A = [1  + c[i, j - 1] - d[i, j - 1]  d[i, j - 1]; 
            #           -d[i, j]        1 + c[i, j] + d[i, j] ]
            #     b = [phi[i, j - 1, n] + c[i, j - 1] * phi[i - 1, j - 1, n + 1]; phi[i, j, n] + c[i, j] * phi[i - 1, j, n + 1]]
            #     x = A \ b  # Solves the system A * x = b
            #     phi[i, j - 1, n + 1] = x[1]
            #     phi1[i, j - 1, n + 1] = x[2]
            #     phi1[i, j, n + 1] = x[2]
            #     phi[i, j, n + 1] = x[2]
            #     continue
            # end

            phi_first_order[i, j, n + 1] = ( phi_first_order[i, j, n] + cp[i, j] * phi_first_order[i - 1, j, n + 1] 
                                                                      + dp[i, j] * phi_first_order[i, j - 1, n + 1] 
                                                                      - cm[i, j] * phi_first_order[i + 1, j, n + 1]
                                                                      - dm[i, j] * phi_first_order[i, j + 1, n + 1] ) / ( 1 + cp[i, j] + dp[i, j] - cm[i, j] - dm[i, j] );
            phi1[i, j, n + 1] = ( phi1[i, j, n] + cp[i, j] * phi[i - 1, j, n + 1] 
                                              + dp[i, j] * phi[i, j - 1, n + 1] 
                                              - cm[i, j] * phi[i + 1, j, n + 1]
                                              - dm[i, j] * phi[i, j + 1, n + 1] ) / ( 1 + cp[i, j] + dp[i, j] - cm[i, j] - dm[i, j] );

            phi[i, j, n + 1] = ( phi[i, j, n] - 0.5 * ( phi1[i, j, n + 1] - phi[i, j, n] - phi1[i, j, n] + phi[i, j, n - 1] ) 
                                + cp[i, j] * ( phi[i - 1, j, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i - 1, j, n + 1] - phi1[i - 1, j, n + 1] + phi[i - 2, j, n + 1] ) )
                                - cm[i, j] * ( phi[i + 1, j, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i + 1, j, n + 1] - phi1[i + 1, j, n + 1] + phi[i + 2, j, n + 1] ) )
                                + dp[i, j] * ( phi[i, j - 1, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i, j - 1, n + 1] - phi1[i, j - 1, n + 1] + phi[i, j - 2, n + 1] ) )
                                - dm[i, j] * ( phi[i, j + 1, n + 1] - 0.5 * ( phi1[i, j, n + 1] - phi[i, j + 1, n + 1] - phi1[i, j + 1, n + 1] + phi[i, j + 2, n + 1] ) ) ) / ( 1 + cp[i, j] + dp[i, j] - cm[i, j] - dm[i, j] );
                            

        end
        end

        # Outlfow boundary condition on the left side
        phi_first_order[:, 2, n + 1] = 3 * phi_first_order[:, 3, n + 1] - 3 * phi_first_order[:, 4, n + 1] + phi_first_order[:, 5, n + 1]
        phi_first_order[:, 1, n + 1] = 3 * phi_first_order[:, 2, n + 1] - 3 * phi_first_order[:, 3, n + 1] + phi_first_order[:, 4, n + 1]
        phi1[:, 2, n + 1] = 3 * phi1[:, 3, n + 1] - 3 * phi1[:, 4, n + 1] + phi1[:, 5, n + 1]
        phi1[:, 1, n + 1] = 3 * phi1[:, 2, n + 1] - 3 * phi1[:, 3, n + 1] + phi1[:, 4, n + 1]
        phi[:, 2, n + 1] = 3 * phi[:, 3, n + 1] - 3 * phi[:, 4, n + 1] + phi[:, 5, n + 1]
        phi[:, 1, n + 1] = 3 * phi[:, 2, n + 1] - 3 * phi[:, 3, n + 1] + phi[:, 4, n + 1]

        # Outlfow boundary condition on x boundary
        phi_first_order[2, :, n + 1] = 3 * phi_first_order[3, :, n + 1] - 3 * phi_first_order[4, :, n + 1] + phi_first_order[5, :, n + 1]
        phi_first_order[1, :, n + 1] = 3 * phi_first_order[2, :, n + 1] - 3 * phi_first_order[3, :, n + 1] + phi_first_order[4, :, n + 1]
        phi1[2, :, n + 1] = 3 * phi1[3, :, n + 1] - 3 * phi1[4, :, n + 1] + phi1[5, :, n + 1]
        phi1[1, :, n + 1] = 3 * phi1[2, :, n + 1] - 3 * phi1[3, :, n + 1] + phi1[4, :, n + 1]
        phi[2, :, n + 1] = 3 * phi[3, :, n + 1] - 3 * phi[4, :, n + 1] + phi[5, :, n + 1]
        phi[1, :, n + 1] = 3 * phi[2, :, n + 1] - 3 * phi[3, :, n + 1] + phi[4, :, n + 1]

        phi_first_order[end-1, :, n + 1] = 3 * phi_first_order[end-2, :, n + 1] - 3 * phi_first_order[end-3, :, n + 1] + phi_first_order[end-4, :, n + 1]
        phi_first_order[end, :, n + 1] = 3 * phi_first_order[end-1, :, n + 1] - 3 * phi_first_order[end-2, :, n + 1] + phi_first_order[end-3, :, n + 1]
        phi1[end-1, :, n + 1] = 3 * phi1[end-2, :, n + 1] - 3 * phi1[end-3, :, n + 1] + phi1[end-4, :, n + 1]
        phi1[end, :, n + 1] = 3 * phi1[end-1, :, n + 1] - 3 * phi1[end-2, :, n + 1] + phi1[end-3, :, n + 1]
        phi[end-1, :, n + 1] = 3 * phi[end-2, :, n + 1] - 3 * phi[end-3, :, n + 1] + phi[end-4, :, n + 1]
        phi[end, :, n + 1] = 3 * phi[end-1, :, n + 1] - 3 * phi[end-2, :, n + 1] + phi[end-3, :, n + 1]
    end
end
end
end


# Print error
Error_t_h = tau * h^2 * sum(sum(abs.(phi[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 2:Ntau+2)
println("Error t*h: ", Error_t_h)
Error_t_h_1 = tau * h^2 * sum(sum(abs.(phi_first_order[:, :, n] - phi_exact.(X, Y, t[n]))) for n in 2:Ntau+2)
println("Error t*h first order: ", Error_t_h_1)

# Load the last error
last_error = load_last_error()
if last_error !== nothing
    println("Order: ", log(2, last_error / Error_t_h))
end 
# Save the last error
save_last_error(Error_t_h)
println("=============================")

# Compute gradient of the solution in the last time step
d_phi_x = zeros(N + 3, N + 3);
d_phi_y = zeros(N + 3, N + 3);


for i = 2:1:N+2
for j = 2:1:N+2
    d_phi_x[i, j] = sign(cp[i,j]) * (phi[i + 1, j, end-1] - phi[i, j, end-1]) / h - sign(cm[i,j]) * (phi[i, j, end-1] - phi[i - 1, j, end-1]) / h;
    d_phi_y[i, j] = sign(dp[i,j]) * (phi[i, j + 1, end-1] - phi[i, j, end-1]) / h - sign(dm[i,j]) * (phi[i, j, end-1] - phi[i, j - 1, end-1]) / h;
end
end


println("minimum derivative x: ", minimum(d_phi_x))
println("maximum derivative x: ", maximum(d_phi_x))
println("minimum derivative y: ", minimum(d_phi_y))
println("maximum derivative y: ", maximum(d_phi_y))

# Plot of the result at the final time together with the exact solution
trace3 = surface(x = x, y = y, z = phi_exact.(X, Y, t[end - 1]), 
                 mode = "lines", name = "Exact Solution", 
                 showscale=false, contours_coloring="lines", 
                 colorscale="Greys", line_width=3, opacity=1.0)

trace0 = surface(x = x, y = y, z = phi_0.(X, Y), 
                 mode = "lines", name = "Initial Condition", 
                 showscale=false, colorscale = "Greys", 
                 contours_coloring="lines", line_width=2, opacity=0.36)

trace2 = surface(x = x, y = y, z = phi[:, :, end - 1], 
                 mode = "lines", name = "First Order (Method A)", 
                 showscale=false, colorscale = "Reds", 
                 contours_coloring="lines", line_width=2, opacity=1.0)

trace1 = surface(x = x, y = y, z = phi_first_order[:, :, end - 1], 
                 mode = "lines", name = "First Order (Method B)", 
                 showscale=false, colorscale = "Reds", 
                 contours_coloring="lines", line_width=2, opacity=0.5)

layout = Layout(plot_bgcolor="white", 
                xaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)), 
                yaxis=attr(zerolinecolor="gray", gridcolor="lightgray", tickfont=attr(size=20)))

plot_phi = plot([trace3, trace0, trace1, trace2], layout)
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