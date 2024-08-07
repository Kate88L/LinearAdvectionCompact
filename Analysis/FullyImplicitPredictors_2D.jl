using SymPy

# Define your symbolic variables
@syms h tau # space and time steps, velocity and Courant number
@syms u_i_j_n u_im_j_n u_ip_j_n u_i_j_nm u_imm_j_n u_i_j_nmm u_i_j_np u_i_jm_n u_i_jp_n u_i_jnn_n  # grid points
@syms DXXX DXX DX DTTT DTT DT DXXT DXT DXTT DYYY DYY DY DYT DYYT DYTT DXYT DXXY DXYY DXY # derivatives
@syms ω α μ # ENO parameters
@syms c d
@syms u v # velocity

# Taylor expansion
u_imm_j_n = u_i_j_n - 2 * h * DX + 4 * (h^2 / 2) * DXX - 8 * (h^3 / 6) * DXXX
u_im_j_n = u_i_j_n - h * DX + (h^2 / 2) * DXX - h^3/6 * DXXX
u_ip_j_n = u_i_j_n + h * DX + (h^2 / 2) * DXX + h^3/6 * DXXX

u_i_jm_n = u_i_j_n - h * DY + (h^2 / 2) * DYY - h^3/6 * DYYY
u_i_jp_n = u_i_j_n + h * DY + (h^2 / 2) * DYY + h^3/6 * DYYY
u_i_jmm_n = u_i_j_n - 2 * h * DY + 4 * (h^2 / 2) * DYY - 8 * (h^3 / 6) * DYYY

u_im_jm_n = u_i_j_n - h * DY - h * DX + (h^2 / 2) * DXX + (h^2 / 2) * DYY + h^2 * DXY -
                    h^3/6 * DXXX - h^3/6 * DYYY - h^3 / 6 * DXXY - h^3 / 6 * DXYY

u_i_j_np = u_i_j_n + tau * DT + (tau^2 / 2) * DTT + tau^3/6 * DTTT
u_i_j_nm = u_i_j_n - tau * DT + (tau^2 / 2) * DTT - (tau^3 / 6) * DTTT
u_i_j_nmm = u_i_j_n - 2 * tau * DT + 4 * (tau^2 / 2) * DTT - 8 * (tau^3 / 6) * DTTT 


# Fully implicit second order scheme for ̂̂v|∇u| = 1 where ̂v = (1, u, v) and ∇ = (∂_t, ∂_x, ∂_y)
S_x = ( u_i_j_n - u_im_j_n ) / h + ω * ( u_ip_j_n - 2 * u_i_j_n + u_im_j_n ) / (2 * h) + (1 - ω) * ( u_i_j_n - 2 * u_im_j_n + u_imm_j_n ) / ( 2 * h )
S_y = ( u_i_j_n - u_i_jm_n ) / h + μ * ( u_i_jp_n - 2 * u_i_j_n + u_i_jm_n ) / (2 * h) + (1 - μ) * ( u_i_j_n - 2 * u_i_jm_n + u_i_jmm_n ) / ( 2 * h )
S_t = ( u_i_j_n - u_i_j_nm ) / tau + α * ( u_i_j_np - 2 * u_i_j_n + u_i_j_nm ) / ( 2 * tau ) + (1 - α) * ( u_i_j_n - 2 * u_i_j_nm + u_i_j_nmm ) / ( 2 * tau )

S = ( u_i_j_n - u_i_j_nm + α/2 *  ( u_i_j_np - 2 * u_i_j_n + u_i_j_nm ) + (1 - α)/2 * ( u_i_j_n - 2 * u_i_j_nm + u_i_j_nmm ) 
                + c * ( u_i_j_n - u_im_j_n + ω/2 * ( u_ip_j_n - 2 * u_i_j_n + u_im_j_n ) + (1 - ω)/2 * ( u_i_j_n - 2 * u_im_j_n + u_imm_j_n ) ) 
                + d * ( u_i_j_n - u_i_jm_n + μ/2 * ( u_i_jp_n - 2 * u_i_j_n + u_i_jm_n ) + (1 - μ)/2 * ( u_i_j_n - 2 * u_i_jm_n + u_i_jmm_n ) ) )


S = S.subs(c, u * tau / h)
S = S.subs(d, v * tau / h)
S = S.subs(DT, - u * DX - v * DY)


print(simplify("\n"))
print(simplify(S))



