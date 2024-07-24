using SymPy

# Define your symbolic variables
@syms h tau # space and time steps, velocity and Courant number
@syms u_i_n u_im_n u_ip_n u_i_nm u_imm_n u_i_nmm u_i_np  # grid points
@syms DXXX DXX DX DTTT DTT DT DXXT DXT DXTT # derivatives
@syms ω α # ENO parameters

@syms c cm cmm cp dc

@syms v # velocity

# Taylor expansion
u_imm_n = u_i_n - 2 * h * DX + 4 * (h^2 / 2) * DXX - 8 * (h^3 / 6) * DXXX
u_im_n = u_i_n - h * DX + (h^2 / 2) * DXX - h^3/6 * DXXX
u_ip_n = u_i_n + h * DX + (h^2 / 2) * DXX + h^3/6 * DXXX

u_i_np = u_i_n + tau * DT + (tau^2 / 2) * DTT + tau^3/6 * DTTT
u_i_nm = u_i_n - tau * DT + (tau^2 / 2) * DTT - (tau^3 / 6) * DTTT
u_i_nmm = u_i_n - 2 * tau * DT + 4 * (tau^2 / 2) * DTT - 8 * (tau^3 / 6) * DTTT 

u_ip_nm = u_i_n + h * DX - tau * DT + 
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT + 
                (h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT + (h * tau^2 / 2) * DXTT 

u_im_nm = u_i_n - h * DX - tau * DT +
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT + tau * h * DXT - 
                (h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT

u_im_np = u_i_n - h * DX + tau * DT +
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT - 
                (h^3 / 6) * DXXX + (tau^3 / 6) * DTTT + (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT


u_imm_nm = u_i_n - 2 * h * DX - tau * DT +
                (4 * h^2 / 2) * DXX + (tau^2 / 2) * DTT + 2 * tau * h * DXT - 
                (8 * h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (4 * h^2 * tau / 2) * DXXT - (2 * h * tau^2 / 2) * DXTT


u_immm_n = u_i_n - 3 * h * DX + 6 * (h^2 / 2) * DXX - 9 * (h^3 / 6) * DXXX

# First order Predictors
dc = 0
cm = c - dc * h
cp = c + dc * h
cmm = cm - dc * h

uP_i_n = ( u_i_nm + c * u_im_n ) / (1 + c)
uP_im_n = ( u_im_nm + cm * u_imm_n ) / (1 + cm)
uP_ip_n = ( u_ip_nm + cp * u_i_n ) / (1 + cp)
# uP_imm_n = ( u_imm_nm + cmm * u_immm_n ) / (1 + cmm)

uP_i_np = ( u_i_n + c * u_im_np ) / (1 + c)
uP_i_nm = ( u_i_nmm + c * u_im_nm ) / (1 + c)

# Fully implicit second order scheme for ̂̂v|∇u| = 1 where ̂v = (1, v) and ∇ = (∂_t, ∂_x)
S_x = ( u_i_n - u_im_n ) / h + ω * ( u_ip_n - 2 * u_i_n + u_im_n ) / (2 * h) + (1 - ω) * ( u_i_n - 2 * u_im_n + u_imm_n ) / ( 2 * h )
S_t = ( u_i_n - u_i_nm ) / tau + α * ( u_i_np - 2 * u_i_n + u_i_nm ) / ( 2 * tau ) + (1 - α) * ( u_i_n - 2 * u_i_nm + u_i_nmm ) / ( 2 * tau )

S = ( u_i_n - u_i_nm + α/2 *  ( uP_i_np - 2 * uP_i_n + uP_i_nm ) + (1 - α)/2 * ( u_i_n - 2 * u_i_nm + u_i_nmm ) 
                + c * ( u_i_n - u_im_n + ω/2 * ( uP_ip_n - 2 * uP_i_n + uP_im_n ) + (1 - ω)/2 * ( u_i_n - 2 * u_im_n + u_imm_n ) ) )

S = ( u_i_n - u_i_nm + 1/2 * ( u_i_n - 2 * u_i_nm + u_i_nmm ) 
                    + c * ( u_i_n - u_im_n + ω/2 * ( uP_ip_n - 2 * uP_i_n + uP_im_n ) + (1 - ω)/2 * ( u_i_n - 2 * u_im_n + u_imm_n ) ) )

S = S.subs(c, v * tau / h)
S = S.subs(DT, - v * DX)


print(simplify("\n"))
print(simplify(S))



