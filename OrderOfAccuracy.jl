using SymPy

# Define your symbolic variables
@vars h tau c a # space and time steps 
@vars u_i_n u_im_n u_imm_n # points of stencil
@vars DXXX DXX DX DTTT DTT DT DXXT DXT DXTT # derivatives
@vars s w # ENO parameters

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
                (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT - 
                (h^3 / 6) * DXXX + (tau^3 / 6) * DTTT + (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT

# Predictors
uP_i_n = ( u_i_nm + c * u_im_n ) / (1 + c);
uP_im_n = ( u_im_nm + c * u_imm_n ) / (1 + c);
uP_im_np = ( u_i_n + c * u_im_np ) / (1 + c);


# Final scheme
# s = 1/2
# w = 1/3

f_normal = (u_i_n - u_i_nm) + c *(u_i_n - u_im_n) + 
    c / 2 * ( (1 - w) * ( u_ip_nm - u_i_nm - uP_i_n + uP_im_n ) + 
                   w * ( u_i_nm - u_im_nm - u_im_n + u_imm_n ) );

f_inverted = (u_i_n - u_i_nm) + c * (u_i_n - u_im_n) + 
    1 / 2 * ( (1 - s) * ( u_im_np - u_im_n - u_i_n + u_i_nm ) + 
                   s * ( u_im_n - u_im_nm - u_i_nm + u_i_nmm ) );

f_normal = f_normal.subs(DT, - a * DX)
f_inverted = f_inverted.subs(DX, - 1/a * DT)
f_normal = f_normal.subs(DTT, - a * DXT)
f_inverted = f_inverted.subs(DXX, - 1/a * DXT)

# Courant number
f_normal = f_normal.subs(c, a * tau / h)
f_inverted = f_inverted.subs(c, a * tau / h)

f_normal = f_normal.subs(DTTT, -a^3 * DXXX)
f_normal = f_normal.subs(DXTT, a^2 * DXXX)
f_normal = f_normal.subs(DXXT, -a * DXXX)
f_normal = f_normal.subs(DXT, -a * DXX)

f_inverted = f_inverted.subs(DXXX, -1/a^3 * DTTT)
f_inverted = f_inverted.subs(DXTT, 1/a^2 * DTTT)
f_inverted = f_inverted.subs(DXXT, -1/a * DTTT)
f_inverted = f_inverted.subs(DXT, -1/a * DTT)

# f_normal = f_normal.subs(w, 1/3)

simplified_f_normal = cancel(expand(f_normal))
simplified_f_inverted = cancel(expand(f_inverted))

# Print the result in a readable format
println(simplified_f_normal)
# println(simplified_f_inverted)
