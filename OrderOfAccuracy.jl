using SymPy

# Define your symbolic variables
@vars h tau c cm cp dc da a # space and time steps, velocity and Courant number
@vars u_i_n u_im_n u_ip_n u_i_nm u_imm_n u_i_nmm u_ip_nm u_im_nm u_im_np u_ip_nmm u_imm_np # grid points
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

u_ip_nmm = u_i_n + h * DX - 2 * tau * DT + 
                (h^2 / 2) * DXX + (4 * tau^2 / 2) * DTT - 2 * tau * h * DXT + 
                (h^3 / 6) * DXXX - (8 * tau^3 / 6) * DTTT - (2 * h^2 * tau / 2) * DXXT + (4 * h * tau^2 / 2) * DXTT 

u_imm_np = u_i_n - 2 * h * DX + tau * DT +
                (4 * h^2 / 2) * DXX + (tau^2 / 2) * DTT - 2 * tau * h * DXT - 
                (8 * h^3 / 6) * DXXX + (tau^3 / 6) * DTTT + (4 * h^2 * tau / 2) * DXXT - (2 * h * tau^2 / 2) * DXTT

# Predictors 
# NOTE: the following is for constant velocity only!!!

dc = 0;
da = dc * h / tau;
cm = c - dc * h;
cp = c + dc * h;

uP_i_n = ( u_i_nm + c * u_im_n ) / (1 + c);
uP_im_n = ( u_im_nm + cm * u_imm_n ) / (1 + cm);
uP_im_np = ( u_im_n + cm * u_imm_np ) / (1 + cm);
uP_ip_nm = ( u_ip_nmm + cp * u_i_nm ) / (1 + cp);
uP_i_nm = ( u_i_nmm + c * u_im_nm ) / (1 + c);


# Final scheme
f_normal = (u_i_n - u_i_nm) + c *(u_i_n - u_im_n) + 
    c / 2 * ( (1 - w) * ( uP_ip_nm - uP_i_nm - uP_i_n + uP_im_n ) + 
                   w * ( u_i_nm - u_im_nm - u_im_n + u_imm_n ) );

f_inverted = (u_i_n - u_i_nm) + c * (u_i_n - u_im_n) + 
    1 / 2 * ( (1 - s) * ( uP_im_np - uP_im_n - uP_i_n + uP_i_nm ) + 
                   s * ( u_im_n - u_im_nm - u_i_nm + u_i_nmm ) );

f_normal = f_normal.subs(DT, - a * DX)
f_inverted = f_inverted.subs(DX, - 1/a * DT)
f_normal = f_normal.subs(DTT, - a * DXT)
f_inverted = f_inverted.subs(DXX, - 1/a * DXT)
f_inverted = f_inverted.subs(DTT, - a * DXT)

# Courant number
# f_normal = f_normal.subs(c, a * tau / h)
# f_inverted = f_inverted.subs(c, a * tau / h)

f_normal = f_normal.subs(DTTT, -a^3 * DXXX)
f_normal = f_normal.subs(DXTT, a^2 * DXXX)
f_normal = f_normal.subs(DXXT, -a * DXXX)
f_normal = f_normal.subs(DXT, -a * DXX)

f_inverted = f_inverted.subs(DTTT, -a^3 * DXXX)
f_inverted = f_inverted.subs(DXTT, a^2 * DXXX)
f_inverted = f_inverted.subs(DXXT, -a * DXXX)

f_normal = f_normal.subs(a, c * h / tau)
f_inverted = f_inverted.subs(a, c * h / tau)

simplified_f_normal = cancel(expand(f_normal))
simplified_f_inverted = cancel(expand(f_inverted))

# Print the result in a readable format
# println(simplify(simplified_f_normal))
println(simplify(simplified_f_inverted))
# println(simplify((simplified_f_inverted + simplified_f_normal)/2))

# equation = Eq(simplified_f_normal/DXXX, 0);
# solve(equation, w)

equation = Eq(simplified_f_inverted/DTTT, 0);
solve(equation, s)
