import autograd.numpy as np
import astropy.units as u

from .. import optic, ray, optgroup

class Mirror(optic.Optic):
    def __init__(self,
                 equations:list, 
                 extent:list[float],
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Mirror"):
        super().__init__(equations, extent, scene, group, origin, rotation, label)

    def interaction(self, angle:u.Quantity, ray:ray.Ray, t_opt:float) -> u.Quantity:
        return -angle
    

class FlatMirror(Mirror):
    def __init__(self,
                 aperture:float|u.Quantity,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "Flat mirror"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: t, lambda t: 0*t), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         scene=scene, group=group,origin=origin, rotation=rotation, label=label)
        
def Slit(full_size:float|u.Quantity,
         slit_width:float|u.Quantity,
         scene=None,
         group=None,
         origin:np.ndarray|list[float]=[0,0], 
         rotation:float|u.Quantity=0*u.deg,
         label:str = "Slit"):
    full_size = full_size.to(u.mm).value if type(full_size)==u.Quantity else full_size
    slit_width = slit_width.to(u.mm).value if type(slit_width)==u.Quantity else slit_width
    slit = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = FlatMirror(aperture=full_size, group=slit, origin=[0,0], rotation=0, label=1)
    botMirror = FlatMirror(aperture=full_size, group=slit, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*slit_width, 0.5*full_size]
    botMirror.extent = [-0.5*full_size, -0.5*slit_width]
    return slit


class GeneralMirror(Mirror):
    def __init__(self,
                 R:float|u.Quantity,
                 b:float,
                 aperture:float|u.Quantity,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str = "General mirror"):
        self.R : float= R.to(u.mm).value if type(R)==u.Quantity else R
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: t, lambda t: (t**2/R)/(1+np.sqrt(1-(1+b)*(t/R)**2))), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ParabolicMirror(focal:float|u.Quantity,
                    aperture:float|u.Quantity,
                    scene=None,
                    group=None,
                    origin:np.ndarray|list[float]=[0,0], 
                    rotation:float|u.Quantity=0*u.deg,
                    label:str = "Parabolic mirror"):
        return GeneralMirror(R=2*focal, b=-1, aperture=aperture, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ParabolicMirrorHole(focal:float|u.Quantity,
                        aperture:float|u.Quantity,
                        hole:float|u.Quantity,
                        scene=None,
                        group=None,
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Parabolic mirror"):
    hole = hole.to(u.mm).value if type(hole)==u.Quantity else hole
    mirror = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = ParabolicMirror(focal=focal, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=1)
    botMirror = ParabolicMirror(focal=focal, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*hole, 0.5*aperture]
    botMirror.extent = [-0.5*aperture, -0.5*hole]
    return mirror

def HyperbolicMirror(radius:float|u.Quantity,
                     aperture:float|u.Quantity,
                     b:float=-2,
                     scene=None,
                     group=None,
                     origin:np.ndarray|list[float]=[0,0], 
                     rotation:float|u.Quantity=0*u.deg,
                     label:str = "Hyperbolic mirror"):
        return GeneralMirror(R=radius, b=b, aperture=aperture, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def HyperbolicMirrorHole(radius:float|u.Quantity,
                         aperture:float|u.Quantity,
                         hole:float|u.Quantity,
                         b:float=-2, 
                         scene=None,
                         group=None,
                         origin:np.ndarray|list[float]=[0,0], 
                         rotation:float|u.Quantity=0*u.deg,
                         label:str = "Hyperbolic mirror"):
    hole = hole.to(u.mm).value if type(hole)==u.Quantity else hole
    mirror = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = HyperbolicMirror(radius=radius, aperture=aperture, b=b, group=mirror, origin=[0,0], rotation=0, label=1)
    botMirror = HyperbolicMirror(radius=radius, aperture=aperture, b=b, group=mirror, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*hole, 0.5*aperture]
    botMirror.extent = [-0.5*aperture, -0.5*hole]
    return mirror

def SphericalMirror(radius:float|u.Quantity,
                    aperture:float|u.Quantity,
                    scene=None,
                    group=None,
                    origin:np.ndarray|list[float]=[0,0], 
                    rotation:float|u.Quantity=0*u.deg,
                    label:str = "Spherical mirror"):
        return GeneralMirror(R=radius, b=0, aperture=aperture, scene=scene, group=group, origin=origin, rotation=rotation, label=label)
        
def SphericalMirrorHole(radius:float|u.Quantity,
                        aperture:float|u.Quantity,
                        hole:float|u.Quantity, 
                        scene=None,
                        group=None,
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Hyperbolic mirror"):
    hole = hole.to(u.mm).value if type(hole)==u.Quantity else hole
    mirror = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    topMirror = SphericalMirror(radius=radius, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=1)
    botMirror = SphericalMirror(radius=radius, aperture=aperture, group=mirror, origin=[0,0], rotation=0, label=2)
    topMirror.extent = [0.5*hole, 0.5*aperture]
    botMirror.extent = [-0.5*aperture, -0.5*hole]
    return mirror