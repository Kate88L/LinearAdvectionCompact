using SymPy

# Define your symbolic variables
@vars h tau c d cm dm cp dp dc dd da a db b # space and time steps, velocity and Courant number
@vars u_i_j_n u_im_j_n u_ip_j_n u_i_j_nm u_imm_j_n u_i_j_nmm u_ip_j_nm u_im_j_nm u_im_j_np u_ip_j_nmm u_imm_j_np # grid points
@vars u_i_jm_n u_i_jp_n u_i_jmm_n u_i_jp_nm u_i_jm_nm u_i_jm_np u_i_jp_nmm u_i_jmm_np u_im_jm_n u_i_j_np u_im_jm_np u_im_jm_nm u_imm_jm_np u_im_jmm_np
@vars DXXX DXX DX DTTT DTT DT DXXT DXT DXTT # derivatives
@vars DYYY DYY DY DXXY DXYY DXY DYTT DYYT DXYT DYT
@vars DXp DYp DXXp DYYp DXXXp DYYYp DXYp DYTTp DYYTp DXYTp DYTp
@vars DXn DYn DXXn DYYn DXXXn DYYYn DXYn DYTTn DYYTn DXYTn DYTn
@vars α β # ENO parameters

# Taylor expansion
u_im_j_n = u_i_j_n - h * DX + (h^2 / 2) * DXX - h^3/6 * DXXX
u_ip_j_n = u_i_j_n + h * DX + (h^2 / 2) * DXX + h^3/6 * DXXX

u_i_jm_n = u_i_j_n - h * DY + (h^2 / 2) * DYY - h^3/6 * DYYY
u_i_jp_n = u_i_j_n + h * DY + (h^2 / 2) * DYY + h^3/6 * DYYY

u_i_j_nm = u_i_j_n - tau * DT + (tau^2 / 2) * DTT - (tau^3 / 6) * DTTT
u_i_j_np = u_i_j_n + tau * DT + (tau^2 / 2) * DTT + (tau^3 / 6) * DTTT
u_i_j_nmm = u_i_j_n - 2 * tau * DT + 4 * (tau^2 / 2) * DTT - 8 * (tau^3 / 6) * DTTT 

u_imm_j_n = u_i_j_n - 2 * h * DX + 4 * (h^2 / 2) * DXX - 8 * (h^3 / 6) * DXXX
u_i_jmm_n = u_i_j_n - 2 * h * DY + 4 * (h^2 / 2) * DYY - 8 * (h^3 / 6) * DYYY

u_ip_j_nm = u_i_j_n + h * DX - tau * DT + 
            (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT + 
            (h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT + (h * tau^2 / 2) * DXTT 
u_i_jp_nm = u_i_j_n + h * DY - tau * DT + 
            (h^2 / 2) * DYY + (tau^2 / 2) * DTT - tau * h * DYT + 
            (h^3 / 6) * DYYY - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DYYT + (h * tau^2 / 2) * DYTT 

u_im_j_nm = u_i_j_n - h * DX - tau * DT +
            (h^2 / 2) * DXX + (tau^2 / 2) * DTT + tau * h * DXT - 
            (h^3 / 6) * DXXX - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT
u_i_jm_nm = u_i_j_n - h * DY - tau * DT +
            (h^2 / 2) * DYY + (tau^2 / 2) * DTT + tau * h * DYT - 
            (h^3 / 6) * DYYY - (tau^3 / 6) * DTTT - (h^2 * tau / 2) * DYYT - (h * tau^2 / 2) * DYTT

u_im_jm_n = u_i_j_n - h * ( DX + DY ) +
            (h^2 / 2) * ( DXX + DYY ) + h^2 * DXY -
            (h^3 / 6) * ( DXXX + 3 * DXXY + 3 * DXYY + DYYY)

DXp = DX + tau * DXT + (tau^2 / 2) * DXTT
DYp = DY + tau * DYT + (tau^2 / 2) * DYTT
DXXp = DXX + tau * DXXT
DYYp = DYY + tau * DYYT
DXYp = DXY + tau * DXYT
DXXXp = DXXX
DXYYp = DXYY
DXXYp = DXXY
DYYYp = DYYY

DXn = DX - tau * DXT + (tau^2 / 2) * DXTT
DYn = DY - tau * DYT + (tau^2 / 2) * DYTT
DXXn = DXX - tau * DXXT
DYYn = DYY - tau * DYYT
DXYn = DXY - tau * DXYT
DXXXn = DXXX
DXYYn = DXYY
DXXYn = DXXY
DYYYn = DYYY

u_im_jm_np = u_i_j_np - h * ( DXp + DYp ) +
            (h^2 / 2) * ( DXXp + DYYp ) + h^2 * DXYp -
            (h^3 / 6) * ( DXXXp + 3 * DXXYp + 3 * DXYYp + DYYYp)

u_im_jm_nm = u_i_j_nm - h * ( DXn + DYn ) +
            (h^2 / 2) * ( DXXn + DYYn ) + h^2 * DXYn -
            (h^3 / 6) * ( DXXXn + 3 * DXXYn + 3 * DXYYn + DYYYn)

u_im_j_np = u_i_j_n - h * DX + tau * DT +
            (h^2 / 2) * DXX + (tau^2 / 2) * DTT - tau * h * DXT - 
            (h^3 / 6) * DXXX + (tau^3 / 6) * DTTT + (h^2 * tau / 2) * DXXT - (h * tau^2 / 2) * DXTT
u_i_jm_np = u_i_j_n - h * DY + tau * DT +
                (h^2 / 2) * DYY + (tau^2 / 2) * DTT - tau * h * DYT - 
                (h^3 / 6) * DYYY + (tau^3 / 6) * DTTT + (h^2 * tau / 2) * DYYT - (h * tau^2 / 2) * DYTT

u_ip_j_nmm = u_i_j_n + h * DX - 2 * tau * DT + 
            (h^2 / 2) * DXX + (4 * tau^2 / 2) * DTT - 2 * tau * h * DXT + 
            (h^3 / 6) * DXXX - (8 * tau^3 / 6) * DTTT - (2 * h^2 * tau / 2) * DXXT + (4 * h * tau^2 / 2) * DXTT
u_i_jp_nmm = u_i_j_n + h * DY - 2 * tau * DT + 
            (h^2 / 2) * DYY + (4 * tau^2 / 2) * DTT - 2 * tau * h * DYT + 
            (h^3 / 6) * DYYY - (8 * tau^3 / 6) * DTTT - (2 * h^2 * tau / 2) * DYYT + (4 * h * tau^2 / 2) * DYTT 

u_imm_j_np = u_i_j_n - 2 * h * DX + tau * DT +
            (4 * h^2 / 2) * DXX + (tau^2 / 2) * DTT - 2 * tau * h * DXT - 
            (8 * h^3 / 6) * DXXX + (tau^3 / 6) * DTTT + (4 * h^2 * tau / 2) * DXXT - (2 * h * tau^2 / 2) * DXTT
u_i_jmm_np = u_i_j_n - 2 * h * DY + tau * DT +
            (4 * h^2 / 2) * DYY + (tau^2 / 2) * DTT - 2 * tau * h * DYT - 
            (8 * h^3 / 6) * DYYY + (tau^3 / 6) * DTTT + (4 * h^2 * tau / 2) * DYYT - (2 * h * tau^2 / 2) * DYTT

u_imm_jm_np = u_i_j_np - h * ( 2 * DXp + DYp ) +
            (h^2 / 2) * ( 4 * DXXp + DYYp ) + 2 * h^2 * DXYp -
            (h^3 / 6) * ( 8 * DXXXp + 4 * 3 * DXXYp + 2 * 3 * DXYYp + DYYYp)
u_im_jmm_np = u_i_j_np - h * ( DXp + 2 * DYp ) +
            (h^2 / 2) * ( DXXp + 4 * DYYp ) + 2 * h^2 * DXYp -
            (h^3 / 6) * ( DXXXp + 2 * 3 * DXXYp + 4 * 3 * DXYYp + 8 * DYYYp)
# Predictors 
# NOTE: the following is for constant velocity only!!! 
dc = 0;
da = dc * sqrt(2) * h / tau;
db = dc * sqrt(2) * h / tau;
cm = c - dc * h;
cp = c + dc * h;

uP_i_j_n = ( u_i_j_nm + c * sqrt(2) * u_im_jm_n) / (1 + sqrt(2) * c);
uP_i_j_nm = ( u_i_j_nmm + c * sqrt(2) * u_im_jm_nm) / (1 + sqrt(2) * c);

uP_im_j_np = ( u_im_j_n + cm * sqrt(2) * u_imm_jm_np ) / (1 + sqrt(2) * cm);
uP_i_jm_np = ( u_i_jm_n + cm * u_im_jmm_np ) / (1 + sqrt(2) * cm);


# Diagonal direction of the velocity a == b == C * sqrt(2) * h / tau
a = c * sqrt(2) * h / tau;
b = c * sqrt(2) * h / tau;

# Final scheme
f_inverted = (u_i_j_n - u_i_j_nm) + c * sqrt(2) * (u_i_j_n - u_im_jm_n)
f_inverted = f_inverted + 1/2 * ( (1 - α) * ( u_i_j_np - 2 * u_i_j_n + u_i_j_nm ) + α * ( u_i_j_n - 2 * u_i_j_nm + u_i_j_nmm ) ) # remove DTT terms
f_inverted = f_inverted - 1/2 * ( (1 - β) * ( u_i_j_np - u_i_j_n - u_im_j_np + u_im_j_n ) + β * ( u_i_j_n - u_i_j_nm - u_im_j_n + u_im_j_nm ) ) # remove DXT terms
f_inverted = f_inverted - 1/2 * ( (1 - β) * ( u_i_j_np - u_i_j_n - u_i_jm_np + u_i_jm_n ) + β * ( u_i_j_n - u_i_j_nm - u_i_jm_n + u_i_jm_nm ) ) # remove DYT terms
# f_inverted = f_inverted - 1/2 * ( (1 - α) * ( u_i_j_np - u_i_j_n - u_im_j_np + u_im_j_n ) + α * ( u_i_j_n - u_i_j_nm - u_im_j_n + u_im_j_nm ) ) # remove DXT terms
# f_inverted = f_inverted - 1/2 * ( (1 - α) * ( u_i_j_np - u_i_j_n - u_i_jm_np + u_i_jm_n ) + α * ( u_i_j_n - u_i_j_nm - u_i_jm_n + u_i_jm_nm ) ) # remove DYT terms

f_inverted = f_inverted.subs(DT, - a * DX - b * DY)

# Courant number
f_inverted = f_inverted.subs(DXX, -1/a * ( DXT + b * DXY ) )
f_inverted = f_inverted.subs(DYY, -1/b * ( DYT + a * DXY ) )

# Remove /beta  
f_inverted = f_inverted.subs(β, 1/2 * (1 + α))


simplified_f_inverted = cancel(expand(f_inverted))

# Print the result in a readable format
# println(simplify(simplified_f_normal))
println(simplify(simplified_f_inverted))
# println(simplify((simplified_f_inverted + simplified_f_normal)/2))

# equation = Eq(simplified_f_inverted/DTTT, 0);
# solve(equation, s)
