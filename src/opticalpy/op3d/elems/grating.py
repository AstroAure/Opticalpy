import autograd.numpy as np
import astropy.units as u

from .. import ray
from . import mirror

class Grating(mirror.FlatMirror):
    def __init__(self,
                 width:u.Quantity|float,
                 height:u.Quantity|float,
                 period:u.Quantity|float,
                 label:str = "Grating",
                 **kwargs):
        # Normal along 'Z' relative axis
        # Grooves along 'X' relative axis
        # Dispersion along 'Y' relative axis
        self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
        self.width : float = width.to_value(u.mm) if type(width)==u.Quantity else width
        self.height : float = height.to_value(u.mm) if type(height)==u.Quantity else height
        super().__init__(diameter=np.maximum(width,height), label=label, **kwargs)
        self.eq_bounds = (lambda x,y,z: np.maximum(abs(x)-0.5*self.width,abs(y)-0.5*self.height))
        self.update_transforms()
    
    def interaction(self, dir:np.ndarray, normal:np.ndarray, r:ray.Ray, pos_coll:np.ndarray) -> np.ndarray:
        rel_x = [1,0,0]@self.rot_matrix
        rel_y = [0,1,0]@self.rot_matrix
        rel_z = [0,0,1]@self.rot_matrix
        proj_x = np.dot(dir, rel_x)
        proj_y = np.dot(dir, rel_y)
        proj_z = np.dot(dir, rel_z)
        angle_in = np.arctan2(abs(proj_x),abs(proj_z))
        angle_out = np.arcsin(np.sin(angle_in) - (r.order*r.wavelength*self.period).to_value(u.dimensionless_unscaled))
        proj_x_out = np.sin(angle_out)*(np.sign(proj_x) if proj_x!=0 else 1)
        proj_z_out = np.cos(angle_out)*(np.sign(proj_z) if proj_z!=0 else 1)
        refl_dir = proj_x_out*rel_x + proj_y*rel_y - proj_z_out*rel_z
        refl_dir /= np.linalg.norm(refl_dir)
        return refl_dir

# class TransmissionGrating(mirror.FlatMirror):
#     def __init__(self,
#                  aperture:u.Quantity|float,
#                  period:u.Quantity|float,
#                  order:int=1, 
#                  scene=None,
#                  group=None, 
#                  origin:np.ndarray|list[float]=[0,0], 
#                  rotation:float|u.Quantity=0*u.deg,
#                  label:str = "Transmission grating"):
#         self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
#         self.order : int = order
#         self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
#         super().__init__(aperture, scene, group, origin, rotation, label)

#     def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
#         return (np.pi+np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - np.sin(angle).value))*u.rad
    
# class RefractionGrating(lens.FlatRefraction):
#     def __init__(self,
#                  aperture:u.Quantity|float,
#                  period:u.Quantity|float,
#                  order:int=1,
#                  n_in=1.0, n_out=1.0, 
#                  scene=None,
#                  group=None, 
#                  origin:np.ndarray|list[float]=[0,0], 
#                  rotation:float|u.Quantity=0*u.deg,
#                  label:str = "Refraction grating"):
#         self.period : u.Quantity = period if type(period)==u.Quantity else period/u.mm
#         self.order : int = order
#         self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
#         super().__init__(aperture=aperture, n_in=n_in, n_out=n_out, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

#     def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
#         # Dispersive lens
#         if type(self.n_in)==float:
#             opt_n_in = self.n_in
#         else:
#             opt_n_in = self.n_in(ray.wavelength)
#         if type(self.n_out)==float:
#             opt_n_out = self.n_out
#         else:
#             opt_n_out = self.n_out(ray.wavelength)
#         # Entry or exit
#         if ray.n_in==opt_n_in:
#             n1, n2 = opt_n_in, opt_n_out
#             ray.n_in = opt_n_out
#         elif ray.n_in==opt_n_out:
#             n1, n2 = opt_n_out, opt_n_in
#             ray.n_in = opt_n_in
#         else:
#             print(f"WARNING : Ray ({ray.label}) entering a non-planned refraction interface ({self.label}) !")
#             n1, n2 = opt_n_in, opt_n_out
#         # Grism equation
#         return (np.pi+np.arcsin((self.order*ray.wavelength*self.period).to(u.dimensionless_unscaled).value - n1*np.sin(angle).value)/n2)*u.rad