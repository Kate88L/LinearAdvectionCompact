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