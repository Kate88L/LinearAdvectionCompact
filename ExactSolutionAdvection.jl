using SymPy
using Plots

include("Utils/ExactSolutions.jl")

f(x, t) = cosVelocityNonSmooth(x, t);

# generate grid
x = range(-π / 2, stop = 3 * π / 2 , length = 1000)
t = range(0, stop = 4 * π, length = 1000)

# plot the solution in time 0
# plot(t, f.(0, t), label = "t = 0", xlabel = "x", ylabel = "f(x, t)", title = "Advection equation", lw = 2)

# creat symbolic equation from f and x
@syms X T
f_sym = f(X, T)
# subsitute the value of x 
f_sym = subs(f_sym, X, 0)
d_f_sym = diff(f_sym, T)

solve(d_f_sym, T)

# plot d_f_sym as a function of t
plot(t, d_f_sym.(t), label = "df/dt", xlabel = "t", ylabel = "df/dt", title = "Advection equation", lw = 2)

