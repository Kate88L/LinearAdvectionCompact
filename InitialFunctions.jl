## Initial functions

# Piecewise constant function
function piecewiseConstant(x)
    if -0.5 <= x <= 0.5
        return  1.0
    else
        return 0.0
    end
end

# Piecewise linear function
function piecewiseLinear(x)
    if 0.25 <= x <= 0.5
        return  4.0 * (x - 0.25)
    elseif 0.5 <= x <= 0.75
        return -4 * (x - 0.75)
    else 
        return 0.0
    end
end

# Non smooth function
function nonSmooth(x)
    c = -(sqrt(3)/2 + 9/2 + 2*pi/3) * (x + 1)

    if -1 <= x < -1/3
        return c + 2 * cos(3 * pi * x^2 / 2) - sqrt(3)
    elseif -1/3 <= x < 0
        return c + 3/2 + 3 * cos(2 * pi * x)
    elseif 0 <= x < 1/3
        return c + 15/2 - 3 * cos(2 * pi * x)
    elseif 1/3 <= x < 1
        return c + 6 * pi * x * (x - 1) + ( 28 + 4 * pi + cos(3 * pi * x ) ) / 3
    else
        return 0.0
    end
end

function allInOne(x)
    if - 0.8 <= x < -0.4
        return 1
    elseif -0.2 <= x < 0.2
        return - (5*x)^2 + 1
    elseif 0.4 <= x < 0.6
        return 5 * (x - 0.4)
    elseif 0.6 <= x < 0.8
        return -5 * (x - 0.8)
    else
        return 0.0
    end
end

# Function to convert a function f to a periodic function on interval xL, xR
function makePeriodic(f, xL, xR)
    return x -> f(mod(x - xL, xR - xL) + xL)
end