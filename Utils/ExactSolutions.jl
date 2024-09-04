## Exact solutionS

# Exact solution for the advection equation for Velocity = 2 + 3/2 * cos(x) and initial condition = cos(x)
function cosVelocitySmooth(x, t)
    return cos.(2*atan.(sqrt(7).*tan((sqrt(7).*(t - (4*atan.(tan.(x/2)./sqrt(7)))./sqrt(7)))/4)))
end

# Exact solution for the advection equation for Velocity = sin(x) and initial condition = sin(x)
function sinVelocitySmooth(x, t)
    return sin.(2 * atan.( tan.(x / 2) * exp.(-t) ))
end

# Exact solution for the advection equation for Velocity = 1 + 3/4 * cos(x) and initial condition = arcsin( sin(x + Pi/2) * 2 / Pi )
function cosVelocityNonSmooth(x, t)
    return (2 * asin(cos(2 * atan(sqrt(7) * tan((1/8) * sqrt(7) * (t - (8 * atan(tan(x/2) / sqrt(7))) / sqrt(7))))))) / π

end

# Function derivative
function phi_derivative_x(phi, x, t)
    return ForwardDiff.derivative(x -> phi(x, t), x)
end

function phi_derivative_t(phi, x, t)
    return ForwardDiff.derivative(t -> phi(x, t), t)
end

# Rotation of a square in 2D
function nonSmoothRotation(x, y, t) 

    x_hat = x .* cos(t) + y .* sin(t) - 1/4
    y_hat = y .* cos(t) - x .* sin(t) - 1/4

    if y_hat >= abs.(x_hat)
        result = y_hat
    elseif - y_hat >= abs.(x_hat)
        result = - y_hat
    elseif x_hat >= abs.(y_hat)
        result = x_hat
    else
        result = - x_hat
    end

    return min.(result, 0.15)
end

function exactLozanoAslam(x, t)
    if ( t < 0.5 )
        if 0.3 - 0.0 * t <= x <= 0.3 + t
            return  (x - 0.3) / t
        elseif 0.3 + t < x <= 0.6 + 0.5 * t
            return 1
        else
            return 0.0
        end
    else 
        if 0.3 - 0.0 * t <= x <= 0.3 - 0.0 * t + 0.6 * sqrt(2*t)
            return (x - 0.3) / t
        else
            return 0.0
        end
    end
end

# Exact solution for the derivative of the Burgers' equation for smooth initial condition
function exactSmoothBurgersDerivative(x, t)
    try
        if abs.(1 - sqrt(1 + 4 * t * (t - x))) <= 2 * t
            return 1 -  ( (1 - sqrt(1 + 4 * t * (t - x))).^2 ) / (4 * t.^2)
        else
            return 0.0
        end
    catch
        return 0.0
    end
end

function integrate_function(f, a, b)
    N = Int(1e3);
    integral_values = zeros(N)
    x_vals = a:(b-a)/(N-1):b

    for i in 1:N
        integral_values[i] = trapz(x_vals[1:i], f.(x_vals[1:i]))
    end

    return LinearInterpolation(x_vals, integral_values)
end