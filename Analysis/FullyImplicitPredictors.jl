using SymPy

# Define your symbolic variables
@syms h tau # space and time steps, velocity and Courant number
@syms u_i_n u_im_n u_ip_n u_i_nm u_imm_n u_i_nmm u_i_np  # grid points
@syms DXXX DXX DX DTTT DTT DT DXXT DXT DXTT # derivatives
@syms ω α # ENO parameters

@syms c cm cmm cp dc

@syms v dv # velocity

# Taylor expansion
u_imm_n = u_i_n -  Sym(2)* h * DX +  Sym(4) * (h^Sym(2) / Sym(2)) * DXX -  Sym(8) * (h^Sym(3) /  Sym(6)) * DXXX
u_im_n = u_i_n - h * DX + (h^Sym(2) / Sym(2)) * DXX - h^Sym(3)/ Sym(6) * DXXX
u_ip_n = u_i_n + h * DX + (h^Sym(2) / Sym(2)) * DXX + h^Sym(3)/ Sym(6) * DXXX

u_i_np = u_i_n + tau * DT + (tau^Sym(2) / Sym(2)) * DTT + tau^Sym(3)/ Sym(6) * DTTT
u_i_nm = u_i_n - tau * DT + (tau^Sym(2) / Sym(2)) * DTT - (tau^Sym(3) /  Sym(6)) * DTTT
u_i_nmm = u_i_n -  Sym(2)* tau * DT +  Sym(4) * (tau^Sym(2) / Sym(2)) * DTT -  Sym(8) * (tau^Sym(3) /  Sym(6)) * DTTT 

u_ip_nm = u_i_n + h * DX - tau * DT + 
                (h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT - tau * h * DXT + 
                (h^Sym(3) /  Sym(6)) * DXXX - (tau^Sym(3) /  Sym(6)) * DTTT - (h^Sym(2) * tau / Sym(2)) * DXXT + (h * tau^Sym(2) / Sym(2)) * DXTT 
u_ip_nmm = u_i_n + h * DX -  Sym(2)* tau * DT + 
                (h^Sym(2) / Sym(2)) * DXX + ( Sym(4) * tau^Sym(2) / Sym(2)) * DTT -  Sym(2)* tau * h * DXT + 
                (h^Sym(3) /  Sym(6)) * DXXX - ( Sym(8) * tau^Sym(3) /  Sym(6)) * DTTT - ( Sym(2)* h^Sym(2) * tau / Sym(2)) * DXXT + ( Sym(4) * h * tau^Sym(2) / Sym(2)) * DXTT

u_im_nm = u_i_n - h * DX - tau * DT +
                (h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT + tau * h * DXT - 
                (h^Sym(3) /  Sym(6)) * DXXX - (tau^Sym(3) /  Sym(6)) * DTTT - (h^Sym(2) * tau / Sym(2)) * DXXT - (h * tau^Sym(2) / Sym(2)) * DXTT

u_im_np = u_i_n - h * DX + tau * DT +
                (h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT - tau * h * DXT - 
                (h^Sym(3) /  Sym(6)) * DXXX + (tau^Sym(3) /  Sym(6)) * DTTT + (h^Sym(2) * tau / Sym(2)) * DXXT - (h * tau^Sym(2) / Sym(2)) * DXTT


u_imm_nm = u_i_n -  Sym(2)* h * DX - tau * DT +
                ( Sym(4) * h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT +  Sym(2)* tau * h * DXT - 
                ( Sym(8) * h^Sym(3) /  Sym(6)) * DXXX - (tau^Sym(3) /  Sym(6)) * DTTT - ( Sym(4) * h^Sym(2) * tau / Sym(2)) * DXXT - ( Sym(2)* h * tau^Sym(2) / Sym(2)) * DXTT


u_immm_n = u_i_n - Sym(3) * h * DX +  Sym(9) * (h^Sym(2) / Sym(2)) * DXX - Sym(27) * (h^Sym(3) /  Sym(6)) * DXXX

u_imm_np = u_i_n -  Sym(2)* h * DX + tau * DT +
                ( Sym(4) * h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT -  Sym(2)* tau * h * DXT - 
                ( Sym(8) * h^Sym(3) /  Sym(6)) * DXXX + (tau^Sym(3) /  Sym(6)) * DTTT + ( Sym(4) * h^Sym(2) * tau / Sym(2)) * DXXT - ( Sym(2)* h * tau^Sym(2) / Sym(2)) * DXTT

u_immm_np = u_i_n -  Sym(3)* h * DX + tau * DT +
                ( Sym(9) * h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT -  Sym(3)* tau * h * DXT - 
                ( Sym(27) * h^Sym(3) /  Sym(6)) * DXXX + (tau^Sym(3) /  Sym(6)) * DTTT + ( Sym(9) * h^Sym(2) * tau / Sym(2)) * DXXT - ( Sym(3)* h * tau^Sym(2) / Sym(2)) * DXTT
u_immmm_np = u_i_n -  Sym(4)* h * DX + tau * DT +
                ( Sym(16) * h^Sym(2) / Sym(2)) * DXX + (tau^Sym(2) / Sym(2)) * DTT -  Sym(4)* tau * h * DXT - 
                ( Sym(64) * h^Sym(3) /  Sym(6)) * DXXX + (tau^Sym(3) /  Sym(6)) * DTTT + ( Sym(16) * h^Sym(2) * tau / Sym(2)) * DXXT - ( Sym(4)* h * tau^Sym(2) / Sym(2)) * DXTT

# First order Predictors
dc = 0
cm = c - dc * h
cp = c + dc * h
cmm = cm - dc * h
cmmm = cmm - dc * h

uP_i_n = ( u_i_nm + c * u_im_n ) / ( Sym(1) + c) # predictor for u_i_n+1)
uP_im_n = ( u_im_nm + cm * u_imm_n ) / ( Sym(1) + cm) # predictor for u_i-m_n+1
uP_imm_n = ( u_imm_nm + cmm * u_immm_n ) / ( Sym(1) + cmm) # predictor for u_i-mm_n+1
uP_ip_nm = ( u_ip_nmm + cp * u_i_nm ) / ( Sym(1) + cp) # predictor for u_i+1_n
uP_i_nm = ( u_i_nmm + c * u_im_nm ) / ( Sym(1) + c) # predictor for u_i_n-1

# Fully implicit second order scheme for ̂̂v|∇u| =  1 where ̂v = ( 1, v) and ∇ = (∂_t, ∂_x)
S_x = ( u_i_n - u_im_n ) / h + ω * ( u_ip_n -  Sym(2)* u_i_n + u_im_n ) / ( Sym(2)* h) + ( Sym(1) - ω) * ( u_i_n -  Sym(2)* u_im_n + u_imm_n ) / (  Sym(2)* h )
S_t = ( u_i_n - u_i_nm ) / tau + α * ( u_i_np -  Sym(2)* u_i_n + u_i_nm ) / (  Sym(2)* tau ) + ( Sym(1) - α) * ( u_i_n -  Sym(2)* u_i_nm + u_i_nmm ) / (  Sym(2)* tau )

# Second order predictor for u_i_n+1
uP2_i_n = ( u_i_nm -  Sym(1)/ Sym(2)* ( uP_i_n - uP_i_nm - u_i_nm + u_i_nmm ) 
                + c * ( u_im_n -  Sym(1)/ Sym(2)* ( uP_i_n - uP_im_n - u_im_n + u_imm_n ) ) ) / (  Sym(1) + c )   
                
uP_ip_n = ( u_ip_nm  + cp * uP2_i_n ) / (  Sym(1) + cp ) # predictor for u_i+1_n

# Second order predictor for u_i+1_n+1 <------ OK
uP2_ip_n = ( u_ip_nm -  Sym(1) / Sym(2) * ( uP_ip_n - uP_ip_nm - u_ip_nm + u_ip_nmm ) 
                + cp * ( uP2_i_n -  Sym(1)/ Sym(2)* ( uP_ip_n - uP_i_n - uP2_i_n + u_im_n ) ) ) / (  Sym(1) + cp )

uP2_i_nm = u_i_nm # enough for second order
uP2_im_n = u_im_n # enough for second order

uP_immm_np = ( u_immm_n + cmmm * u_immmm_np ) / (  Sym(1) + cmmm ) # predictor for u_i-3_n+2
uP_imm_np = ( u_imm_n + cmm * u_immm_np ) / (  Sym(1) + cmm ) # predictor for u_i-2_n+2

uP2_imm_np = ( u_imm_n -  Sym(1) / Sym(2) * ( uP_imm_np - uP_imm_n - u_imm_n + u_imm_nm ) 
                + cmm * ( u_immm_np -  Sym(1) / Sym(2) * ( uP_imm_np - uP_immm_np - u_immm_np + u_immmm_np ) ) ) / (  Sym(1) + cmm )

uP_im_np = ( u_im_n + cm * uP2_imm_np ) / (  Sym(1) + cm ) # predictor for u_i-1_n+2

uP2_im_np = ( uP2_im_n -  Sym(1) / Sym(2) * ( uP_im_np - uP_im_n - uP2_im_n + u_im_nm ) 
                + cm * ( uP2_imm_np -  Sym(1) / Sym(2) * ( uP_im_np - uP_imm_np - uP2_imm_np + u_immm_np ) ) ) / (  Sym(1) + cm )

uP_i_np = ( uP2_i_n + c * uP2_im_np ) / (  Sym(1) + c ) # predictor for u_i_n+2

uP2_i_np = ( uP2_i_n -  Sym(1) / Sym(2) * ( uP_i_np - uP_i_n - uP2_i_n + u_i_nm ) 
                + c * ( uP2_im_np -  Sym(1) / Sym(2) * ( uP_i_np - uP_im_np - uP2_im_np + uP2_imm_np ) ) ) / (  Sym(1) + c )

# Final scheme
S = ( u_i_n - u_i_nm + α / Sym(2) * ( uP2_i_n - uP2_i_nm - u_i_nm + u_i_nmm ) + ( Sym(1) - α ) /  Sym(2) * ( uP2_i_np - uP2_i_n - uP2_i_n + u_i_nm )
                 + c * ( u_i_n - u_im_n + ω / Sym(2) * ( uP2_i_n - uP2_im_n - u_im_n + u_imm_n ) + ( Sym(1) - ω ) /  Sym(2) * ( uP2_ip_n - uP2_i_n - uP2_i_n + u_im_n ) ) ) 

# Purely upwind scheme
S = ( u_i_n - u_i_nm + 1 / Sym(2) * ( uP_i_n - uP_i_nm - u_i_nm + u_i_nmm ) 
                + c * ( u_i_n - u_im_n + 1 / Sym(2) * ( uP_i_n - uP_im_n - u_im_n + u_imm_n )) )

S = S.subs(c, v * tau / h)
S = S.subs(DT, - v * DX)
S = S.subs(DTT, - v * DXT)
S = S.subs(DXT, - v * DXX -0 * dv * DX)
# S = S.subs(dv, (h * c / tau - h * cm / tau) / h)

# S = S - tau / (2*(h + tau*v)) * 1*DX*dv*h*tau*v

S = S.subs(DTTT, - v^3 * DXXX)
S = S.subs(DXTT, - v^2 * DXXX)
S = S.subs(DXXT, - v * DXXX)

S = S.subs(α, Sym(1))
S = S.subs(ω, Sym(1))


print("\n")
print(simplify(S))
