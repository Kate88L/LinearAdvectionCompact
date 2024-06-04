using QuadGK
using Plots
using Interpolations
using Trapz

include("../Utils/Solvers.jl")

function burgersLozanoAslam(x)
    return x^2 - 1;
end

function integrate_function(f, a, b)
    N = 1000;
    integral_values = zeros(N)
    x_vals = a:(b-a)/(N-1):b
    for i in 1:N
        integral_values[i] = trapz(x_vals[1:i], f.(x_vals[1:i]))
    end

    return LinearInterpolation(x_vals, integral_values)
end

# Define the range of integration and the number of points
a = -2
b = 2
N = 100

# Perform numerical integration
# result = integrate_function(burgersLozanoAslam, a, b)

# Generate x values for plotting
x_vals = range(a, stop=b, length=N)

# Plot the results
# plot(x_vals, result.(x_vals), xlabel="x", ylabel="Integral Value", label="Numerical Integration", legend=:topleft)

# y = newtonMethod(burgersLozanoAslam, x -> 2*x, 0.5)
y = newtonMethod(burgersLozanoAslam, x -> (burgersLozanoAslam(x + 0.000001) - burgersLozanoAslam(x)) / 0.000001, 0.5)
println(y)