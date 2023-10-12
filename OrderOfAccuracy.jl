using SymPy

# Define your symbolic variables
@vars h tau c # space and time steps 
@vars u_i_n u_im_n u_imm_n # points of stencil
@vars DXXX DXX DX DTTT DTT DT DXXT DXT DXTT # derivatives
@vars s w # ENO parameters

# Courant number
c = tau / h

# Taylor expansion
u_im_n = u_i_n - h * DX + (h^2 / 2) * DXX - h^3/6 * DXXX
u_ip_n = u_i_n + h * DX + (h^2 / 2) * DXX + h^3/6 * DXXX

u_i_nm = u_i_n - tau * DT + (tau^2 / 2) * DTT - (tau^3 / 6) * DTTT

u_imm_n = u_i_n - 2 * h * DX + 4 * (h^2 / 2) * DXX - 8 * (h^3 / 6) * DXXX
u_i_nmm = u_i_n - 2 * tau * DT + 4 * (tau^2 / 2) * DTT - 8 * (tau^3 / 6) * DTTT 

u_ip_nm = u_i_n + h * DX - tau * DT + 
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT + 
                (h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT + (h * tau^2 / 2) * DXTT 
u_im_nm = u_i_n - h * DX - tau * DT + 
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT + tau * h * DXT - 
                (h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT
u_im_np = u_i_n - h * DX + tau * DT + 
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT + 
                (h^3 / 6) * DXXX + (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT

# Predictors
uP_i_n = ( u_i_nm + tau/h * u_im_n ) / ( 1 + tau/h );
uP_im_n = ( u_im_nm + tau/h * u_imm_n) / ( 1 + tau/h );
UP_i_np = ( u_i_n + c * u_im_np ) / ( 1 + c );


# Final scheme
s = 1/2
w = 1/2

f_normal = (u_i_n - u_i_nm)/tau + (u_i_n - u_im_n) / h + 
    1 / (2 * h) * ( (1 - w) * ( u_ip_nm - u_i_nm - u_i_n + u_im_n ) + 
                   w * ( u_i_nm - u_im_nm - u_im_n + u_imm_n ) ) - DT - DX;

f_inverted = (u_i_n - u_i_nm)/tau + (u_i_n - u_im_n)/h + 
    1 / (2 * tau) * ( (1 - s) * ( u_im_np - u_im_n - u_i_n + u_i_nm ) + 
                   s * ( u_im_n - u_im_nm - u_i_nm + u_i_nmm ) ) - DT - DX;

simplified_f_normal = simplify(f_normal)
simplified_f_inverted = simplify(f_inverted)

# Print the result in a readable format
println(simplified_f_normal)
println(simplified_f_inverted)