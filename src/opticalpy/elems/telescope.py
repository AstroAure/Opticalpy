import autograd.numpy as np
import astropy.units as u

from .. import optgroup
from . import mirror

def CassegrainTelescope(focal:float,
                        aperture:float,
                        backfocus:float,
                        length:float,
                        scene=None,
                        group=None, 
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Cassegrain",
                        return_geometry:bool=False):
    q = length+backfocus
    M = (focal-q)/(q-backfocus)
    f1 = focal/M
    p = (f1+backfocus)/(M+1)
    aperture2 = aperture*p/f1
    r2 = 2*q/(M-1)
    b = -(4*M)/(M-1)**2 -1
    telescope = optgroup.OpticalGroup(scene=scene, origin=origin, rotation=rotation, label=label)
    primary = mirror.ParabolicMirrorHole(focal=f1, aperture=aperture, hole=aperture2, group=telescope, origin=[backfocus,0], rotation=-90)
    secondary = mirror.HyperbolicMirror(radius=r2, aperture=aperture2, b=b, group=telescope, origin=[q,0], rotation=-90)
    if return_geometry:
        geometry = {'Primary':{'Diameter':aperture,'Focal':f1}, 'Secondary':{'Diameter':aperture2,'Curvature':r2,'Conic':b}, 'Backfocus':backfocus}
        return telescope, geometry
    return telescope

def RitcheyChretienTelescope(focal:float,
                             aperture:float,
                             backfocus:float,
                             length:float,
                             scene=None,
                             group=None, 
                             origin:np.ndarray|list[float]=[0,0], 
                             rotation:float|u.Quantity=0*u.deg,
                             label:str = "Ritech-Chrétien",
                             return_geometry:bool=False):
    q = length+backfocus
    d = q-backfocus
    s = q/d
    M = (focal-q)/(q-backfocus)
    r1 = 2*focal/M
    r2 = 2*q/(M-1)
    p = (r1+backfocus)/(M+1)
    aperture2 = aperture*p/r1
    b1 = -2*s/M**3 -1
    b2 = (-4*M*(M-1) - 2*(M+s))/(M-1)**3 - 1
    telescope = optgroup.OpticalGroup(scene=scene, origin=origin, rotation=rotation, label=label)
    primary = mirror.HyperbolicMirrorHole(radius=r1, aperture=aperture, b=b1, hole=aperture2, group=telescope, origin=[backfocus,0], rotation=-90)
    secondary = mirror.HyperbolicMirror(radius=r2, aperture=aperture2, b=b2, group=telescope, origin=[q,0], rotation=-90)
    if return_geometry:
        geometry = {'Primary':{'Diameter':aperture,'Curvature':r1,'Conic':b1}, 'Secondary':{'Diameter':aperture2,'Curvature':r2,'Conic':b2}, 'Backfocus':backfocus}
        return telescope, geometry
    return telescope