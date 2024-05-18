import autograd.numpy as np
import astropy.units as u

def wavelength2RGB(wavelength:u.Quantity) -> list[float]:
    # Human like RGB color filters
    w = (wavelength.to(u.nm)).value
    def piecewise_gaussian(x, mu, tau1, tau2):
        y = np.empty_like(x)
        y[x<mu] = np.exp(-tau1**2*(x-mu)**2/2)[x<mu]
        y[x>=mu] = np.exp(-tau2**2*(x-mu)**2/2)[x>=mu]
        return y
    x = 1.056*piecewise_gaussian(w, 599.8,0.0264,0.0323) + \
        0.362*piecewise_gaussian(w, 442.0, 0.0624, 0.0374) - \
        0.065*piecewise_gaussian(w, 501.1, 0.0490, 0.0382)
    y = 0.821*piecewise_gaussian(w, 568.8,0.0213,0.0247) + \
        0.286*piecewise_gaussian(w, 530.9, 0.0613, 0.0322)
    z = 1.217*piecewise_gaussian(w, 437.0,0.0845,0.0278) + \
        0.681*piecewise_gaussian(w, 459.0, 0.0385, 0.0725)
    XYZ2RGB = np.array([[2.36461385, -0.89654057, -0.46807328],
                        [-0.51516621, 1.4264081,   0.0887581],
                        [0.0052037,  -0.01440816, 1.00920446]])
    rgb = XYZ2RGB@[x,y,z]
    rgb = np.clip(rgb, 0, 1)
    return rgb

def dist_angle(ang1:u.Quantity, ang2:u.Quantity) -> u.Quantity:
    # Signed angle from ang1 to ang2, result in (-180deg,180deg]
    return np.arctan2(np.sin(ang2-ang1), np.cos(ang2-ang1))