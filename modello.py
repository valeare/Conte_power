import torch
from scipy.io import loadmat

def power_net_dae(model, y_n, h, IRK_weights):
    T = 1.0
    
    device = y_n.device
    
    # parameters
    params = get_params()
    Lf = params['Lf']; Cf = params['Cf']; Rf = params['Rf']; Rcf = params['Rcf']
    Lg = params['Lg']; Rg = params['Rg']; Kpi = params['Kpi']; Kii = params['Kii']
    Kp_pll = params['Kp_pll']; Ki_pll = params['Ki_pll']; omega_nom = params['omega_nom']
    tau_inv = params['tau_inv']

    # pinn
    yn = y_n.clone()
    
    #_, data_mat, _ = load_data("data/t_p.mat","data/U.mat", "data/Y_p.mat")
    #data_row = data_mat.to(device)
    #Pref, Qref, Vg_d, Vg_q, omega_g = [data_row[:, i] for i in range(5)]

    Pref, Qref, Vg_d, Vg_q, omega_g = yn[:, 12], yn[:, 13], yn[:, 14], yn[:, 15], yn[:, 16]
    Pref = Pref.unsqueeze(1)
    Qref = Qref.unsqueeze(1)
    Vg_d = Vg_d.unsqueeze(1)
    Vg_q = Vg_q.unsqueeze(1)
    omega_g = omega_g.unsqueeze(1)

    # TO DO: fourier and exponential features    
    Is_d, Is_q, Ic_d, Ic_q, vCf_d, vCf_q, xcc_d, xcc_q, xpll, theta_r, vc_d, vc_q, Vs_d, Vs_q, Vc_d, Vc_q, vs_d, vs_q, vc_d_ref, vc_q_ref, ic_d, ic_q, omega_pll = model(yn)

    Is_d = Is_d.to(device)
    Is_q = Is_q.to(device)
    Ic_d = Ic_d.to(device)
    Ic_q = Ic_q.to(device)
    vCf_d = vCf_d.to(device)
    vCf_q = vCf_q.to(device)
    xcc_d = xcc_d.to(device)
    xcc_q = xcc_q.to(device)
    xpll = xpll.to(device)
    theta_r = theta_r.to(device)
    vc_d = vc_d.to(device)
    vc_q = vc_q.to(device)
    Vs_d = Vs_d.to(device)
    Vs_q = Vs_q.to(device)
    Vc_d = Vc_d.to(device)
    Vc_q = Vc_q.to(device)
    vs_d = vs_d.to(device)
    vs_q = vs_q.to(device)
    vc_d_ref = vc_d_ref.to(device)
    vc_q_ref = vc_q_ref.to(device)
    ic_d = ic_d.to(device)
    ic_q = ic_q.to(device)
    omega_pll = omega_pll.to(device)
    
    xi_Is_d = Is_d[...,:-1].to(device)
    xi_Is_q = Is_q[...,:-1].to(device)
    xi_Ic_d = Ic_d[...,:-1].to(device)
    xi_Ic_q = Ic_q[...,:-1].to(device)
    xi_vCf_d = vCf_d[...,:-1].to(device)
    xi_vCf_q = vCf_q[...,:-1].to(device)
    xi_xcc_d = xcc_d[...,:-1].to(device)
    xi_xcc_q = xcc_q[...,:-1].to(device)
    xi_xpll = xpll[...,:-1].to(device)
    xi_theta_r = theta_r[...,:-1].to(device)
    xi_vc_d = vc_d[...,:-1].to(device)
    xi_vc_q = vc_q[...,:-1].to(device)
    
    zeta_Vs_d = Vs_d[...,:-1].to(device)
    zeta_Vs_q = Vs_q[...,:-1].to(device)
    zeta_Vc_d = Vc_d[...,:-1].to(device)
    zeta_Vc_q = Vc_q[...,:-1].to(device)
    zeta_vs_d = vs_d[...,:-1].to(device)
    zeta_vs_q = vs_q[...,:-1].to(device)
    zeta_vc_d_ref = vc_d_ref[...,:-1].to(device)
    zeta_vc_q_ref = vc_q_ref[...,:-1].to(device)
    zeta_ic_d = ic_d[...,:-1].to(device)
    zeta_ic_q = ic_q[...,:-1].to(device)
    zeta_omega_pll = omega_pll[...,:-1].to(device)
    
    # compute dynamic residuals
    F0 = T * (-Rg/Lg * xi_Is_d + omega_nom * xi_Is_q + (zeta_Vs_d - Vg_d)/Lg)
    F1 = T * (-omega_nom *  xi_Is_d - Rg/Lg *  xi_Is_q + (zeta_Vs_q - Vg_q)/Lg)
    F2 = T * ((1/Lf)*(zeta_Vc_d - (xi_Ic_d - xi_Is_d)*Rcf - xi_vCf_d - xi_Ic_d*Rf + omega_nom*xi_Ic_q*Lf))
    F3 = T * ((1/Lf)*(zeta_Vc_q - (xi_Ic_q - xi_Is_q)*Rcf - xi_vCf_q - xi_Ic_q*Rf - omega_nom*xi_Ic_d*Lf))
    F4 = T * ((1/Cf)*(xi_Ic_d - xi_Is_d) + omega_nom*xi_vCf_q)
    F5 = T * ((1/Cf)*(xi_Ic_q - xi_Is_q) - omega_nom*xi_vCf_d)

    iref_d = 2/3 * Pref / torch.clamp(zeta_vs_d, 1e-6)
    iref_q = -2/3 * Qref / torch.clamp(zeta_vs_d, 1e-6)

    F6 = T * (Kii*(iref_d - zeta_ic_d))
    F7 = T * (Kii*(iref_q - zeta_ic_q))
    F8 = T * (Ki_pll * zeta_vs_q)
    F9 = T * (zeta_omega_pll - omega_g)
    F10 = T * ( - 1/tau_inv * xi_vc_d + 1/tau_inv * zeta_vc_d_ref)
    F11 = T * ( - 1/tau_inv * xi_vc_q + 1/tau_inv * zeta_vc_q_ref)


    f0 = yn[...,0:1] -  (Is_d - h*F0.mm(IRK_weights.T.to(device)))
    f1 = yn[...,1:2] -  (Is_q - h*F1.mm(IRK_weights.T.to(device)))
    f2 = yn[...,2:3] -  (Ic_d - h*F2.mm(IRK_weights.T.to(device)))
    f3 = yn[...,3:4] -  (Ic_q - h*F3.mm(IRK_weights.T.to(device)))
    f4 = yn[...,4:5] -  (vCf_d - h*F4.mm(IRK_weights.T.to(device)))
    f5 = yn[...,5:6] -  (vCf_q - h*F5.mm(IRK_weights.T.to(device)))
    f6 = yn[...,6:7] -  (xcc_d - h*F6.mm(IRK_weights.T.to(device)))
    f7 = yn[...,7:8] -  (xcc_q - h*F7.mm(IRK_weights.T.to(device)))
    f8 = yn[...,8:9] -  (xpll - h*F8.mm(IRK_weights.T.to(device)))
    f9 = yn[...,9:10] -  (theta_r - h*F9.mm(IRK_weights.T.to(device)))
    f10 = yn[...,10:11] -  (vc_d - h*F10.mm(IRK_weights.T.to(device)))
    f11 = yn[...,11:12] -  (vc_q - h*F11.mm(IRK_weights.T.to(device)))
    
    
    # compute algebrtaic residuals
    G0 = -Vs_d + (Ic_d - Is_d)*Rcf + vCf_d
    G1 = -Vs_q + (Ic_q - Is_q)*Rcf + vCf_q
    iref_d = 2 / 3 * Pref / torch.clamp(vs_d, 1e-6)
    iref_q = -2/3 * Qref / torch.clamp(vs_d, 1e-6)

    G2 = -vc_d_ref + vs_d + xcc_d + Kpi*(iref_d - ic_d) - Lf*omega_pll*ic_q
    G3 = -vc_q_ref + vs_q + xcc_q + Kpi*(iref_q - ic_q) + Lf*omega_pll*ic_d
    G4 = -omega_pll + omega_nom + Kp_pll*vs_q + xpll

    c = torch.cos(theta_r)
    s = torch.sin(theta_r)

    G5 = c*Ic_d + s*Ic_q - ic_d
    G6 = -s*Ic_d + c*Ic_q - ic_q
    G7 = c*Vc_d + s*Vc_q - vc_d
    G8 = -s*Vc_d + c*Vc_q - vc_q
    G9 = c*Vs_d + s*Vs_q - vs_d
    G10 = -s*Vs_d + c*Vs_q - vs_q

    g0 = T * G0
    g1 = T * G1
    g2 = T * G2
    g3 = T * G3
    g4 = T * G4
    g5 = T * G5
    g6 = T * G6
    g7 = T * G7
    g8 = T * G8
    g9 = T * G9
    g10 = T * G10
    
    return [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11], [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, g10]


def get_params():
    return {
        'Lf': 4.1597e-05, 'Cf': 0.0018, 'Rf': 0.0017, 'Rcf': 0.0254,
        'Lg': 1.3866e-05, 'Rg': 0, 'Kpi': 0.0832, 'Kii': 5.4848,
        'Kp_pll': 0.1700, 'Ki_pll': 100, 'omega_nom': 2 * torch.pi * 50,
        'Vb': 660, 'Snom': 5_000_000, 'tau_inv': 0.0001
    }

def load_data(t_path, data_path, Y_path):
    t = loadmat(t_path)['t_p']
    data = loadmat(data_path)['U']
    Y = loadmat(Y_path)['Y_p']

    t = torch.tensor(t, dtype=torch.float32)
    data_t = torch.tensor(data, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    return t, data_t, Y_t