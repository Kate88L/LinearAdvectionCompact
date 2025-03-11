using SymPy

# Define symbols
N = 6                    # Number of grid points
h = Sym("h")            # Grid spacing
c = Sym("c")             # Advection speed

# velocity
settings = Dict( 
    "constant positive" => ones(N),
    "constant negative" => -ones(N),
    "shock" => vcat(ones(div(N,2)), -ones(div(N,2))),
    "rarefaction" => vcat(-ones(div(N,2)), ones(div(N,2)))
)

# select vecolity
vel = settings["rarefaction"]

# Create a symbolic differentiation matrix for upwind scheme
Dp = sympy.zeros(N, N)    # Matrix for system with positive velocity
Dm = sympy.zeros(N, N)    # Matrix for system with negative velocity
D = sympy.zeros(N, N)    # Matrix for system with both positive and negative velocity

# Implicit second order upwind scheme, positive velocity
# Used scheme is :
# ∂ₜφ + c ∂ₓφ = 0 ----> ∂ₓφ = (φᵢ - φᵢ₋₁)/h
Dp[1, 1] = 1
Dp[2, 2] = 1
for i in 3:N  # Julia uses 1-based indexing, so range starts from 2
    Dp[i, i] = (1 + c)
    Dp[i, i-1] = -c - c/2
    Dp[i, i-2] = c/2
end

# Implicit second order upwind scheme, negative velocity
# Used scheme is :
# ∂ₜφ + c ∂ₓφ = 0 ----> ∂ₓφ = (-φᵢ + φᵢ₊₁)/h
Dm[end, end] = 1
Dm[end-1, end-1] = 1
for i in 1:N-2  # Julia uses 1-based indexing, so range starts from 2
    Dm[i, i] = (1 + c)
    Dm[i, i+1] = -c - c/2
    Dm[i, i+2] = c/2
end

# Combine matrices based on the sign of the velocity
D = Dp .* (vel .> 0) + Dm .* (vel .< 0)

# Display the symbolic differentiation matrix
println("Upwind Differentiation Matrix:")
display(D)
