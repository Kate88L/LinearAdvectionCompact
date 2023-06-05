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
