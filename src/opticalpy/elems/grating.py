import autograd.numpy as np
import astropy.units as u

from .. import ray
from . import mirror, lens

class Grating(mirror.FlatMirror):
    def __init__(self,
                 aperture:u.Quantity|float,
                 period:u.Quantity|float,
                 order:int=1, 
                 scene=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Grating"):
        self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
        self.order : int = order
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        return np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - np.sin(angle).value)*u.rad

class TransmissionGrating(mirror.FlatMirror):
    def __init__(self,
                 aperture:u.Quantity|float,
                 period:u.Quantity|float,
                 order:int=1, 
                 scene=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Transmission grating"):
        self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
        self.order : int = order
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        return (np.pi+np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - np.sin(angle).value))*u.rad
    
class RefractionGrating(lens.FlatRefraction):
    def __init__(self,
                 aperture:u.Quantity|float,
                 period:u.Quantity|float,
                 order:int=1,
                 n_in=1.0, n_out=1.0, 
                 scene=None,
                 group=None, 
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Refraction grating"):
        self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
        self.order : int = order
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(aperture=aperture, n_in=n_in, n_out=n_out, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        # Dispersive lens
        if type(self.n_in)==float:
            opt_n_in = self.n_in
        else:
            opt_n_in = self.n_in(ray.wavelength)
        if type(self.n_out)==float:
            opt_n_out = self.n_out
        else:
            opt_n_out = self.n_out(ray.wavelength)
        # Entry or exit
        if ray.n_in==opt_n_in:
            n1, n2 = opt_n_in, opt_n_out
            ray.n_in = opt_n_out
        elif ray.n_in==opt_n_out:
            n1, n2 = opt_n_out, opt_n_in
            ray.n_in = opt_n_in
        else:
            print(f"WARNING : Ray ({ray.label}) entering a non-planned refraction interface ({self.label}) !")
            n1, n2 = opt_n_in, opt_n_out
        # Grism equation
        return (np.pi+np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - n1*np.sin(angle).value)/n2)*u.rad