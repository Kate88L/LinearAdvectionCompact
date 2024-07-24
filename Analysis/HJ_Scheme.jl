using SymPy

# Define your symbolic variables
@syms h tau # space and time steps, velocity and Courant number
@syms u_i_n u_im_n u_ip_n u_i_nm u_imm_n u_i_nmm u_ip_nm u_im_nm u_im_np u_ip_nmm u_imm_np # grid points
@syms DXXX DXX DX DTTT DTT DT DXXT DXT DXTT # derivatives
@syms ω # ENO parameters

@syms dx_u_i_np dx_u_i_n dx_u_i_nm dx_u_im_n dx_u_ip_n dx_u_im_np dx_u_ip_nm dx_u_im_nm dx_u_ip_np dx_u_i_nmm dx_u_ip_nmm dx_u_imm_np # derivatives

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

u_ip_nmm = u_i_n + h * DX - 2 * tau * DT + 
                (h^2 / 2) * DXX + (4 * tau^2 / 2) * DTT - 2 * tau * h * DXT + 
                (h^3 / 6) * DXXX - (8 * tau^3 / 6) * DTTT - (2 * h^2 * tau / 2) * DXXT + (4 * h * tau^2 / 2) * DXTT 

u_imm_np = u_i_n - 2 * h * DX + tau * DT +
                (4 * h^2 / 2) * DXX + (tau^2 / 2) * DTT - 2 * tau * h * DXT - 
                (8 * h^3 / 6) * DXXX + (tau^3 / 6) * DTTT + (4 * h^2 * tau / 2) * DXXT - (2 * h * tau^2 / 2) * DXTT

# Tayler expansion of ∂x u_i^n+1/2
S = dx_u_i_np - Sym(1/2) * ω *( dx_u_i_np - dx_u_i_n ) - (Sym(1) - ω) * (dx_u_ip_np - dx_u_ip_n);

