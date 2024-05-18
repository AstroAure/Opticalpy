import autograd.numpy as np
import astropy.units as u

def n_sellmeier(B1:float, C1:float|u.Quantity,
                B2:float, C2:float|u.Quantity,
                B3:float, C3:float|u.Quantity):
    C1:u.Quantity = C1 if type(C1)==u.Quantity else C1*u.um**2
    C2:u.Quantity = C2 if type(C2)==u.Quantity else C2*u.um**2
    C3:u.Quantity = C3 if type(C3)==u.Quantity else C3*u.um**2
    return lambda l: np.sqrt(1 + (B1*l**2)/(l**2-C1) + (B2*l**2)/(l**2-C2) + (B3*l**2)/(l**2-C3)).value

n_BK7 = n_sellmeier(1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653)
n_Crown = n_sellmeier(1.1273555, 0.00720341707, 0.124412303, 0.0269835916, 0.827100531, 100.384588)
n_Flint = n_sellmeier(1.34533359, 0.00997743871, 0.209073176, 0.0470450767, 0.937357162, 111.886764)

def rho_water(T:u.Quantity):
    """From https://en.wikipedia.org/wiki/Optical_properties_of_water_and_ice"""
    T = T.to(u.Celsius, equivalencies=u.temperature())
    a1 = -3.983035*u.Celsius
    a2 = 301.797*u.Celsius
    a3 = 522528.9*u.Celsius**2
    a4 = 69.34881*u.Celsius
    a5 = 999.974950*u.kg/u.m**3
    return a5*(1-((T+a1)**2*(T+a2))/(a3*(T+a4)))

def n_water(T:float|u.Quantity=273.15*u.K, rho:float|u.Quantity=None):
    """From https://en.wikipedia.org/wiki/Optical_properties_of_water_and_ice"""
    T:u.Quantity = T.to(u.K, equivalencies=u.temperature()) if type(T)==u.Quantity else (T*u.Celsius).to(u.K, equivalencies=u.temperature())
    T0 = 273.15*u.K
    T_ = T/T0
    rho:u.Quantity = rho_water(T) if rho is None else (rho if type(rho)==u.Quantity else rho*u.kg/u.m**3)
    rho0 = 1000*u.kg/u.m**3
    rho_ = rho/rho0
    l0 = 589*u.nm
    l_IR_ = 5.432937
    l_UV_ = 0.229202
    a0 = 0.244257733
    a1 = 0.00974634476
    a2 = -0.00373234996
    a3 = 0.000268678472
    a4 = 0.0015892057
    a5 = 0.00245934259
    a6 = 0.90070492
    a7 = -0.0166626219
    def n(l:u.Quantity):
        l_ = l/l0
        A = rho_*(a0 + a1*rho_ + a2*T_ + a3*l_**2*T_ + a4/l_**2 + a5/(l_**2-l_UV_**2) + a6/(l_**2-l_IR_**2) + a7*rho_**2)
        n = np.sqrt((2*A+1)/(1-A))
        return n.value
    return n