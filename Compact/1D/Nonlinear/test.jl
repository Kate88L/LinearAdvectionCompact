using QuadGK
using Plots
using Interpolations
using Trapz

include("../../../Utils/Solvers.jl")
include("../../../Utils/InitialFunctions.jl")
include("../../../Utils/ExactSolutions.jl")


# Define the range of integration and the number of points
a = -3
b = 3
N = 100

# Perform numerical integration
# result = integrate_function(burgersLozanoAslam, a, b)

# Generate x values for plotting
x_vals = range(a, stop=b, length=N)

u0 = smoothBurgers

burgers(x) = exactSmoothBurgersDerivative(x, 6/10)
uExact = integrate_function(burgers, a, b)

# Plot the results
plot(x_vals, uExact.(x_vals), xlabel="x", ylabel="Integral Value", label="Numerical Integration", legend=:topleft)
