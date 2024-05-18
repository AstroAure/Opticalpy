import autograd.numpy as np
import astropy.units as u

from .. import optic, ray, optgroup

class Refraction(optic.Optic):
    def __init__(self,
                 equations:list, 
                 extent:list[float],
                 n_in=1.0, n_out=1.0,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Refraction"):
        self.n_in = n_in
        self.n_out = n_out
        super().__init__(equations, extent, scene, group, origin, rotation, label)

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
        # Total internal reflection
        if abs((n1/n2)*np.sin(angle).value) > 1:
            ray.n_in = n1
            return -angle
        # Snell's law
        return (np.pi+np.arcsin((n1/n2)*np.sin(angle).value))*u.rad

class FlatRefraction(Refraction):
    def __init__(self,
                 aperture:float|u.Quantity,
                 n_in=1.0, n_out=1.0,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Flat refraction"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: t, lambda t: 0*t), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         n_in=n_in, n_out=n_out, 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)

class CircularRefraction(Refraction):
    def __init__(self,
                 radius:float|u.Quantity,
                 aperture:float|u.Quantity,
                 n_in=1.0, n_out=1.0,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Circular refraction"):
        self.radius : float= radius.to(u.mm).value if type(radius)==u.Quantity else radius
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        super().__init__(equations=(lambda t: self.radius*np.sin(t), lambda t: -self.radius*(np.cos(np.arcsin(0.5*self.aperture/self.radius))-np.cos(t))), 
                         extent=[-np.arcsin(0.5*self.aperture/self.radius), np.arcsin(0.5*self.aperture/self.radius)], 
                         n_in=n_in, n_out=n_out, 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)
        
def Lens(radius1:float|u.Quantity,
         radius2:float|u.Quantity,
         aperture:float|u.Quantity,
         n,
         mid_width:float|u.Quantity=0,
         scene=None,
         group=None,
         origin:np.ndarray|list[float]=[0,0], 
         rotation:float|u.Quantity=0*u.deg,
         label:str|None = "Lens"):
    radius1 : float= radius1.to(u.mm).value if type(radius1)==u.Quantity else radius1
    radius2 : float= radius2.to(u.mm).value if type(radius2)==u.Quantity else radius2
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    mid_width : float= mid_width.to(u.mm).value if type(mid_width)==u.Quantity else mid_width
    lens = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    if radius1==np.inf:
        front = FlatRefraction(aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,0.5*mid_width], rotation=0, label=f"{label} front")
    else:
        front = CircularRefraction(radius=radius1, aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,0.5*mid_width], rotation=0, label=f"{label} front")
    if radius2==np.inf:
        back = FlatRefraction(aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,-0.5*mid_width], rotation=180, label=f"{label} back")
    else:
        back = CircularRefraction(radius=radius2, aperture=aperture, n_in=1.0, n_out=n, group=lens, origin=[0,-0.5*mid_width], rotation=180, label=f"{label} back")
    if mid_width!=0:
        top    = FlatRefraction(aperture=mid_width, n_in=1.0, n_out=n, group=lens, origin=[0.5*aperture,0], rotation=90, label=f"{label} top")
        bottom = FlatRefraction(aperture=mid_width, n_in=1.0, n_out=n, group=lens, origin=[-0.5*aperture,0], rotation=90, label=f"{label} bottom")
    return lens

def ThinSymmetricalLens(focal:float|u.Quantity,
                        aperture:float|u.Quantity,
                        n,
                        width:float=0,
                        f_wavelength:float|u.Quantity=520,
                        scene=None,
                        group=None,
                        origin:np.ndarray|list[float]=[0,0], 
                        rotation:float|u.Quantity=0*u.deg,
                        label:str = "Lens"):
    focal : float= focal.to(u.mm).value if type(focal)==u.Quantity else focal
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    f_wavelength : u.Quantity = f_wavelength if type(f_wavelength)==u.Quantity else f_wavelength*u.nm
    n_f : float = n if type(n)==float else n(f_wavelength)
    radius = 2*focal*(n_f-1)
    d = abs(radius*(1-np.cos(np.arcsin(0.5*aperture/radius))))
    mid_width = max(0,width-2*d)
    if radius<0:
        mid_width = max(2*d+width,2*1.5*d)
    return Lens(radius1=radius, radius2=radius, aperture=aperture, n=n, mid_width=mid_width, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ThinBackFlatLens(focal:float|u.Quantity,
                    aperture:float|u.Quantity,
                    n,
                    width:float=0,
                    f_wavelength:float|u.Quantity=520,
                    scene=None,
                    group=None,
                    origin:np.ndarray|list[float]=[0,0], 
                    rotation:float|u.Quantity=0*u.deg,
                    label:str = "Lens"):
    focal : float= focal.to(u.mm).value if type(focal)==u.Quantity else focal
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    f_wavelength : u.Quantity = f_wavelength if type(f_wavelength)==u.Quantity else f_wavelength*u.nm
    n_f : float = n if type(n)==float else n(f_wavelength)
    radius = focal*(n_f-1)
    d = abs(radius*(1-np.cos(np.arcsin(0.5*aperture/radius))))
    mid_width = max(0,width-d)
    if radius<0:
        mid_width = max(d+width,1.5*d)
    return Lens(radius1=radius, radius2=np.inf, aperture=aperture, n=n, mid_width=mid_width, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def ThinFrontFlatLens(focal:float|u.Quantity,
                      aperture:float|u.Quantity,
                      n,
                      width:float=0,
                      f_wavelength:float|u.Quantity=520,
                      scene=None,
                      group=None,
                      origin:np.ndarray|list[float]=[0,0], 
                      rotation:float|u.Quantity=0*u.deg,
                      label:str = "Lens"):
    focal : float= focal.to(u.mm).value if type(focal)==u.Quantity else focal
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    f_wavelength : u.Quantity = f_wavelength if type(f_wavelength)==u.Quantity else f_wavelength*u.nm
    n_f : float = n if type(n)==float else n(f_wavelength)
    radius = focal*(n_f-1)
    d = abs(radius*(1-np.cos(np.arcsin(0.5*aperture/radius))))
    mid_width = max(0,width-d)
    if radius<0:
        mid_width = max(d+width,1.5*d)
    return Lens(radius1=np.inf, radius2=radius, aperture=aperture, n=n, mid_width=mid_width, scene=scene, group=group, origin=origin, rotation=rotation, label=label)

def Prism(size:float|u.Quantity,
          n,
          angles:list[float|u.Quantity]=[60*u.deg,60*u.deg],
          scene=None,
          group=None,
          origin:np.ndarray|list[float]=[0,0], 
          rotation:float|u.Quantity=0*u.deg,
          label:str = "Prism"):
    size : float = size.to(u.mm).value if type(size)==u.Quantity else size
    for i,angle in enumerate(angles):
        angles[i] = angle if type(angle)==u.Quantity else angle*u.deg
    prism = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    sin_ratio = size/np.sin(180*u.deg-angles[0]-angles[1]).value
    size_l = sin_ratio*np.sin(angles[1]).value
    size_r = sin_ratio*np.sin(angles[0]).value
    A = np.array([0,0])
    B = np.array([size,0])
    C = A + np.array([size_l*np.cos(angles[0]), size_l*np.sin(angles[0])])
    center = (A+B+C)/3
    center_b = (A+B)/2
    center_l = (A+C)/2
    center_r = (B+C)/2
    base  = FlatRefraction(aperture=size,   n_in=1.0, n_out=n, group=prism, origin=center_b-center, rotation=0, label=f"Base")
    left  = FlatRefraction(aperture=size_l, n_in=1.0, n_out=n, group=prism, origin=center_l-center, rotation=angles[0], label=f"Left")
    right = FlatRefraction(aperture=size_r, n_in=1.0, n_out=n, group=prism, origin=center_r-center,  rotation=180*u.deg-angles[1], label=f"Right")
    return prism

class SchmidtRefraction(Refraction):
    def __init__(self,
                 aperture:float|u.Quantity,
                 radius:float|u.Quantity,
                 n,
                 k:float=1.5,
                 opt_wavelength:float|u.Quantity=520*u.nm,
                 scene=None,
                 group=None,
                 origin:np.ndarray|list[float]=[0,0], 
                 rotation:float|u.Quantity=0*u.deg,
                 label:str|None = "Schmidt refraction"):
        self.aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
        self.radius : float = radius.to(u.mm).value if type(radius)==u.Quantity else radius
        self.n = n
        self.k = k
        if type(self.n)==float:
            opt_n = self.n
        else:
            opt_n = self.n(opt_wavelength)
        super().__init__(equations=(lambda t: t, lambda t: (0.25*k*(self.aperture*t)**2 - t**4)/(4*(opt_n-1)*self.radius**3)), 
                         extent=[-0.5*self.aperture,0.5*self.aperture], 
                         n_in=1.0, n_out=self.n, 
                         scene=scene, group=group, origin=origin, rotation=rotation, label=label)
        
def SchmidtCorrector(aperture:float|u.Quantity,
                     radius:float|u.Quantity,
                     n:float,
                     k:float=1.5,
                     width:float|u.Quantity=1,
                     opt_wavelength:float|u.Quantity=520*u.nm,
                     scene=None,
                     group=None,
                     origin:np.ndarray|list[float]=[0,0], 
                     rotation:float|u.Quantity=0*u.deg,
                     label:str|None = "Schmidt corrector"):
    aperture : float = aperture.to(u.mm).value if type(aperture)==u.Quantity else aperture
    radius : float = radius.to(u.mm).value if type(radius)==u.Quantity else radius
    width : float = width.to(u.mm).value if type(width)==u.Quantity else width
    corrector = optgroup.OpticalGroup(scene=scene, group=group, origin=origin, rotation=rotation, label=label)
    schmidt = SchmidtRefraction(aperture=aperture, radius=radius, n=n, k=k, opt_wavelength=opt_wavelength, group=corrector, origin=[0,-width], rotation=0, label="Schmidt")
    flat = FlatRefraction(aperture=aperture, n_in=1.0, n_out=n, group=corrector, origin=[0,0], rotation=0, label="Flat")
    real_width = width-schmidt.raw_equation_y(0.5*aperture)
    left = FlatRefraction(aperture=real_width, n_in=1.0, n_out=n, group=corrector, origin=[-0.5*aperture, -0.5*real_width], rotation=90, label="Left")
    right = FlatRefraction(aperture=real_width, n_in=1.0, n_out=n, group=corrector, origin=[0.5*aperture, -0.5*real_width], rotation=90, label="Right")
    return corrector