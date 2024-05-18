import autograd.numpy as np
import astropy.units as u

from .. import ray, utils
from . import mirror

class Dichroic(mirror.FlatMirror):
    def __init__(self,
                 aperture:float|u.Quantity,
                 cutoff:float|u.Quantity,
                 reflectAbove:bool=True,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Dichroic"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        self.cutoff : u.Quantity = cutoff if type(cutoff)==u.Quantity else cutoff*u.nm
        self.reflectAbove : bool = reflectAbove
        super().__init__(aperture=aperture, scene=scene, group=group,origin=origin, rotation=rotation, label=label)
        self.color = utils.wavelength2RGB(self.cutoff)
    
    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        if ((ray.wavelength>self.cutoff)&self.reflectAbove)|((ray.wavelength<self.cutoff)&(not self.reflectAbove)):
            return -angle
        return angle+180*u.deg
    
class Filter(mirror.FlatMirror):
    def __init__(self,
                 aperture:float|u.Quantity,
                 pivot:float|u.Quantity,
                 bandwidth:float|u.Quantity,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Filter"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        self.pivot : u.Quantity = pivot if type(pivot)==u.Quantity else pivot*u.nm
        self.bandwidth : u.Quantity = bandwidth if type(bandwidth)==u.Quantity else bandwidth*u.nm
        super().__init__(aperture=aperture, scene=scene, group=group,origin=origin, rotation=rotation, label=label)
        self.color = utils.wavelength2RGB(self.pivot)
    
    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        if (ray.wavelength>self.pivot-0.5*self.bandwidth)&(ray.wavelength<self.pivot+0.5*self.bandwidth):
            return angle+180*u.deg
        return None