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
u = settings["rarefaction"]

# Create a symbolic differentiation matrix for upwind scheme
Dp = sympy.zeros(N, N)    # Matrix for system with positive velocity
Dm = sympy.zeros(N, N)    # Matrix for system with negative velocity
D = sympy.zeros(N, N)    # Matrix for system with both positive and negative velocity

# Implicit first order scheme, positive velocity
# Used scheme is :
# ∂ₜφ + c ∂ₓφ = 0 ----> ∂ₓφ = (φᵢ - φᵢ₋₁)/h
Dp[1, 1] = 1
for i in 2:N  # Julia uses 1-based indexing, so range starts from 2
    Dp[i, i] = (1 + c)
    Dp[i, i-1] = -c
end

# Implicit first order scheme, negative velocity
# Used scheme is :
# ∂ₜφ + c ∂ₓφ = 0 ----> ∂ₓφ = (-φᵢ + φᵢ₊₁)/h
Dm[end, end] = 1
for i in 1:N-1  # Julia uses 1-based indexing, so range starts from 2
    Dm[i, i] = (1 + c)
    Dm[i, i+1] = -c
end

# Combine matrices based on the sign of the velocity
D = Dp .* (u .> 0) + Dm .* (u .< 0)

# Display the symbolic differentiation matrix
println("Upwind Differentiation Matrix:")
display(D)
