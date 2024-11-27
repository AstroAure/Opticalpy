import autograd.numpy as np
import astropy.units as u

from .. import optic, ray

class Mirror(optic.Optic):
    def __init__(self,
                 label:str|None = "Mirror",
                 **kwargs):
        super().__init__(label=label, **kwargs)

    def interaction(self, dir:np.ndarray, normal:np.ndarray, r:ray.Ray, pos_coll:np.ndarray) -> np.ndarray:
        refl_dir = dir - 2*np.dot(dir, normal)*normal
        refl_dir /= np.linalg.norm(refl_dir)
        return refl_dir
    

class FlatMirror(Mirror):
    def __init__(self,
                 diameter:float|u.Quantity,
                 label:str = "Flat mirror",
                 **kwargs):
        self.diameter : float = diameter.to_value(u.mm) if type(diameter)==u.Quantity else diameter
        super().__init__(eq=(lambda x,y,z: z), 
                         eq_bounds=(lambda x,y,z: x*x + y*y - 0.25*self.diameter*self.diameter),
                         size=self.diameter, 
                         label=label, **kwargs)
        
def EllipticalFlatMirror(a:float|u.Quantity,
                         b:float|u.Quantity,
                         label:str = "Elliptical flat mirror",
                         **kwargs):
    mirror = FlatMirror(diameter=2*np.maximum(a,b), label=label, **kwargs)
    mirror.eq_bounds = (lambda x,y,z: (x/a)**2 + (y/b)**2 - 1)
    mirror.update_transforms()
    return mirror

def RectangularFlatMirror(width:float|u.Quantity,
                          height:float|u.Quantity,
                          label:str = "Rectangular flat mirror",
                          **kwargs):
    mirror = FlatMirror(diameter=np.maximum(width,height), label=label, **kwargs)
    mirror.eq_bounds = (lambda x,y,z: np.maximum(abs(x)-0.5*width,abs(y)-0.5*height))
    mirror.update_transforms()
    return mirror
        

class Slit(Mirror):
    def __init__(self,
                 width:float|u.Quantity,
                 slit_width:float|u.Quantity,
                 height:float|u.Quantity = None,
                 slit_height:float|u.Quantity = None,
                 label:str = "Slit",
                 **kwargs):
        # Slit frame bounds
        self.width : float = width.to_value(u.mm) if type(width)==u.Quantity else width
        height = height if height is not None else width
        self.height : float = height.to_value(u.mm) if type(height)==u.Quantity else height
        slit_height = slit_height if slit_height is not None else height
        frame_bounds = lambda x,y,z: np.maximum(abs(x)-0.5*self.width,abs(y)-0.5*self.height)
        # Slit bounds
        self.slit_width : float = slit_width.to_value(u.mm) if type(slit_width)==u.Quantity else slit_width
        self.slit_height : float = slit_height.to_value(u.mm) if type(slit_height)==u.Quantity else slit_height
        slit_bounds = lambda x,y,z: np.maximum(abs(x)-0.5*self.slit_width,abs(y)-0.5*self.slit_height)
        # Full slit bounds by XOR (*) frame and slit bounds
        super().__init__(eq=(lambda x,y,z: z), 
                         eq_bounds=(lambda x,y,z: frame_bounds(x,y,z)*slit_bounds(x,y,z)),
                         size=max(self.width,self.height),
                         label=label, **kwargs)


class GeneralMirror(Mirror):
    def __init__(self,
                 R:float|u.Quantity,
                 b:float,
                 diameter:float|u.Quantity,
                 label:str = "General mirror",
                 **kwargs):
        self.R : float= R.to_value(u.mm) if type(R)==u.Quantity else R
        self.diameter : float = diameter.to_value(u.mm) if type(diameter)==u.Quantity else diameter
        super().__init__(eq=(lambda x,y,z: ((x*x+y*y)/R)/(1+np.sqrt(1-(1+b)*(x*x+y*y)/R**2)) - z), 
                         eq_bounds=(lambda x,y,z: x*x + y*y - 0.25*self.diameter*self.diameter),
                         size=self.diameter,
                         label=label, **kwargs)

def ParabolicMirror(focal:float|u.Quantity,
                    diameter:float|u.Quantity,
                    label:str = "Parabolic mirror",
                    **kwargs):
    return GeneralMirror(R=2*focal, b=-1, diameter=diameter, label=label, **kwargs)

def HyperbolicMirror(radius:float|u.Quantity,
                     diameter:float|u.Quantity,
                     label:str = "Hyperbolic mirror",
                     **kwargs):
    return GeneralMirror(R=radius, b=-2, diameter=diameter, label=label, **kwargs)

def SphericalMirror(radius:float|u.Quantity,
                    diameter:float|u.Quantity,
                    label:str = "Spherical mirror",
                    **kwargs):
    return GeneralMirror(R=radius, b=0, diameter=diameter, label=label, **kwargs)

def add_hole(mirror:GeneralMirror, hole_diameter:float|u.Quantity):
    mirror.hole_diameter = hole_diameter.to_value(u.mm) if type(hole_diameter)==u.Quantity else hole_diameter
    assert mirror.hole_diameter < mirror.diameter
    # Hole added by XOR (*) two circle equations
    mirror.eq_bounds = (lambda x,y,z: (x*x + y*y - 0.25*mirror.hole_diameter*mirror.hole_diameter) * (x*x + y*y - 0.25*mirror.diameter*mirror.diameter))
    mirror.update_transforms()
    return mirror

def ParabolicMirrorHole(focal:float|u.Quantity,
                        diameter:float|u.Quantity,
                        hole_diameter:float|u.Quantity,
                        label:str = "Parabolic bored mirror",
                        **kwargs):
    mirror = ParabolicMirror(focal=focal, diameter=diameter, label=label, **kwargs)
    return add_hole(mirror, hole_diameter)

def HyperbolicMirrorHole(radius:float|u.Quantity,
                         diameter:float|u.Quantity,
                         hole_diameter:float|u.Quantity,
                         label:str = "Parabolic bored mirror",
                         **kwargs):
    mirror = HyperbolicMirror(radius=radius, diameter=diameter, label=label, **kwargs)
    return add_hole(mirror, hole_diameter)
        
def SphericalMirrorHole(radius:float|u.Quantity,
                        diameter:float|u.Quantity,
                        hole_diameter:float|u.Quantity,
                        label:str = "Parabolic bored mirror",
                        **kwargs):
    mirror = SphericalMirror(radius=radius, diameter=diameter, label=label, **kwargs)
    return add_hole(mirror, hole_diameter)