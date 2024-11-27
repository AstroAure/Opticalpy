import autograd.numpy as np
import astropy.units as u

from .. import ray

def WhiteRay(wavelengths:list[u.Quantity|float]|np.ndarray, 
             label:str|None="White",
             **kwargs):
    white = []
    for i, wave in enumerate(wavelengths):
        r = ray.Ray(wavelength=wave, label=f"{label} {i}", **kwargs)
        white.append(r)
    return white

# def CollimatedBeam(aperture:float,
#                    wavelength:float|u.Quantity|list[u.Quantity|float]|np.ndarray=450*u.nm,
#                    scene=None,
#                    group=None,
#                    origin:list[float]|np.ndarray=[0,0], 
#                    rotation:float|u.Quantity=0.0*u.deg, 
#                    N_sources:int=20, 
#                    hole:float=0.0,
#                    label:str|None="Collimated"):
#     rotation = rotation.to(u.deg).value if type(rotation)==u.Quantity else rotation
#     beam = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
#     for i, ang in enumerate(np.linspace(-0.5*aperture,0.5*aperture,N_sources)):
#         if (hole==0) | (abs(ang)>0.5*hole):
#             if (type(wavelength)==int) or (type(wavelength)==float) or (type(wavelength)==u.Quantity):
#                 r = ray.Ray(wavelength=wavelength, group=beam, origin=[0,ang], rotation=0, label=i)
#             else:
#                 r = WhiteRay(wavelengths=wavelength, group=beam, origin=[0,ang], rotation=0, label=i)
#     return beam

# def DivergingBeam(angle:float|u.Quantity,
#                   wavelength:float|u.Quantity|list[u.Quantity|float]|np.ndarray=450*u.nm, 
#                   scene=None,
#                   group=None,
#                   origin:list[float]|np.ndarray=[0,0], 
#                   rotation:float|u.Quantity=0.0*u.deg, 
#                   N_sources:int=20, 
#                   hole:float|u.Quantity=0.0*u.deg,
#                   label:str|None="Diverging"):
#     rotation = rotation.to(u.deg).value if type(rotation)==u.Quantity else rotation
#     angle = angle.to(u.deg).value if type(angle)==u.Quantity else angle
#     hole = hole.to(u.deg).value if type(hole)==u.Quantity else hole
#     beam = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
#     for i, ang in enumerate(np.linspace(-0.5*angle,0.5*angle,N_sources)):
#         if (hole==0) | (abs(ang)>0.5*hole):
#             if (type(wavelength)==int) or (type(wavelength)==float) or (type(wavelength)==u.Quantity):
#                 r = ray.Ray(wavelength=wavelength, group=beam, origin=[0,0], rotation=ang, label=i)
#             else:
#                 r = WhiteRay(wavelengths=wavelength, group=beam, origin=[0,0], rotation=ang, label=i)
#     return beam

# def ConvergingBeam(focal:float|u.Quantity,
#                    F:float,
#                    wavelength:float|u.Quantity|list[u.Quantity|float]|np.ndarray=450*u.nm,
#                    scene=None,
#                    group=None,
#                    origin:list[float]|np.ndarray=[0,0], 
#                    rotation:float|u.Quantity=0.0*u.deg, 
#                    N_sources:int=20, 
#                    hole:float|u.Quantity=0.0*u.deg,
#                    label:str|None="Converging"):
#     rotation = rotation.to(u.deg).value if type(rotation)==u.Quantity else rotation
#     hole = hole.to(u.deg).value if type(hole)==u.Quantity else hole
#     angle = 2*np.rad2deg(np.arctan(0.5/F))
#     beam = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
#     for i, ang in enumerate(np.linspace(-0.5*angle,0.5*angle,N_sources)):
#         if (hole==0) | (abs(ang)>0.5*hole):
#             offset = focal*np.array([1-np.cos(np.deg2rad(ang)),np.sin(np.deg2rad(ang))])
#             if (type(wavelength)==int) or (type(wavelength)==float) or (type(wavelength)==u.Quantity):
#                 r = ray.Ray(wavelength=wavelength, group=beam, origin=offset, rotation=-ang, label=i)
#             else:
#                 r = WhiteRay(wavelengths=wavelength, group=beam, origin=offset, rotation=-ang, label=i)
#     return beam