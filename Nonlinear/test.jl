using QuadGK
using Plots
using Interpolations
using Trapz

function burgersLozanoAslam(x)
    if abs(x + 0.2) >= 0.2
        return 0
    else
        return 1
    end
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
a = -1
b = 1
N = 100

# Perform numerical integration
result = integrate_function(burgersLozanoAslam, a, b)

# Generate x values for plotting
x_vals = range(a, stop=b, length=N)

# Plot the results
plot(x_vals, result.(x_vals), xlabel="x", ylabel="Integral Value", label="Numerical Integration", legend=:topleft)
