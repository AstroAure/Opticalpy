import autograd.numpy as np
import astropy.units as u
from autograd import grad
from skimage.measure import marching_cubes
from scipy import optimize

from . import ray

class Optic():
    def __init__(self, 
                 eq, 
                 eq_bounds=None,
                 size:float=1e3,
                 hitbox:dict|None=None,
                 scene:None=None,
                 origin:np.ndarray|list[float]=[0,0,0], 
                 direction:np.ndarray|list[float]=[0,0,1],
                 label:str|None = None,
                 color:str|list='k',
                 plot_res:int=20,
                 plot_alpha:float=0.2) -> None:
        self.eq = eq # Optic surface where = 0
        self.eq_bounds = eq_bounds # Optic surface where <= 0
        self.size : float = size # Maximum size
        if (hitbox is None) and (eq_bounds is not None): hitbox = calc_hitbox(eq, eq_bounds, self.size)
        self.hitbox : dict = hitbox
        self.origin : np.ndarray = np.array(origin).astype(float)
        self.direction : np.ndarray = np.array(direction).astype(float)
        self.direction /= np.linalg.norm(self.direction)
        self.rot_matrix = rotation_matrix(np.array([0,0,1]), self.direction)
        self.label : str|None = label
        self.color : str|list = color
        self.plot_res : int = plot_res
        self.plot_alpha : float = plot_alpha
        self.scene = scene
        if (self.scene is not None): self.scene.append(self)
        self.update_transforms()
    
    def plot_hitbox(self, ax, color='r') -> None:
        x0,x1 = self.trans_hitbox['x'][0], self.trans_hitbox['x'][1]
        y0,y1 = self.trans_hitbox['y'][0], self.trans_hitbox['y'][1]
        z0,z1 = self.trans_hitbox['z'][0], self.trans_hitbox['z'][1]
        ax.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], [z0,z0,z0,z0,z0], color=color)
        ax.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], [z1,z1,z1,z1,z1], color=color)
        ax.plot([x0,x0], [y0,y0], [z0,z1], color=color)
        ax.plot([x1,x1], [y0,y0], [z0,z1], color=color)
        ax.plot([x0,x0], [y1,y1], [z0,z1], color=color)
        ax.plot([x1,x1], [y1,y1], [z0,z1], color=color)
    
    def __plot__(self, ax, show_hitbox=False) -> None:
        # Calculate surface
        x0,x1 = self.hitbox['x'][0], self.hitbox['x'][1]
        y0,y1 = self.hitbox['y'][0], self.hitbox['y'][1]
        z0,z1 = self.hitbox['z'][0], self.hitbox['z'][1]
        xl, yl ,zl = np.linspace(x0,x1,self.plot_res), np.linspace(y0,y1,self.plot_res), np.linspace(z0,z1,self.plot_res)
        X,Y,Z = np.meshgrid(xl, yl, zl)
        F = self.eq(X,Y,Z)
        # Marching cubes
        verts, faces, normals, values = marching_cubes(F, 0, spacing=[np.diff(xl)[0],np.diff(yl)[0],np.diff(zl)[0]])
        verts += [x0,y0,z0]
        if self.eq_bounds is not None:
            far_idx = np.where(self.eq_bounds(verts[:,0], verts[:,1], verts[:,2])>=0)[0]
            verts[far_idx] = np.nan
            faces_out = np.where(np.isin(faces, far_idx))
            faces = np.delete(faces, faces_out, axis=0)
        # Rotate and translate
        verts = np.dot(verts, self.rot_matrix) + self.origin
        # TODO: Correct rotation to accept assymetric optics
        # Plot
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], 
                        color=self.color, alpha=self.plot_alpha, lw=0.2, ec=self.color, shade=False, 
                        label=self.label)
        if show_hitbox: self.plot_hitbox(ax)

    def calc_normal_func(self):
        return grad(lambda X: -self.trans_eq(*X))

    def interaction(self, dir:np.ndarray, normal:np.ndarray, r:ray.Ray, pos_coll:np.ndarray) -> np.ndarray:
        pass

    def update_transforms(self) -> None:
        self.trans_eq = transform_eq(self.eq, self.origin, self.rot_matrix)
        self.trans_eq_bounds = transform_eq(self.eq_bounds, self.origin, self.rot_matrix)
        if self.scene is not None:
            self.trans_hitbox = calc_hitbox(self.trans_eq, self.trans_eq_bounds, size=self.size, expand=self.scene.step)
        else:
            self.trans_hitbox = calc_hitbox(self.trans_eq, self.trans_eq_bounds, size=self.size)
        self.trans_normal = self.calc_normal_func()

    def set_origin(self, new_origin:np.ndarray|list[float]) -> None:
        self.origin = np.array(new_origin)
        self.update_transforms()

    def set_direction(self, new_direction:np.ndarray|list[float], rot_angle:float=np.pi) -> None:
        self.direction = np.array(new_direction)
        self.rot_matrix = rotation_matrix(np.array([0,0,1]), self.direction, rot_angle)
        self.update_transforms()

    def rotate(self, angle:float, axis:np.ndarray|list[float]=None, absolute=True) -> None:
        if axis is None:
            if absolute:
                axis = self.direction
            else:
                axis = [0,0,1]
        axis = axis/np.linalg.norm(axis)
        kx, ky, kz = axis
        c = np.cos(angle)
        s = np.sin(angle)
        rot = np.array([[c+kx**2*(1-c), kx*ky*(1-c)-kz*s, kx*kz*(1-c)+ky*s],
                        [ky*kx*(1-c)+kz*s, c+ky**2*(1-c), ky*kz*(1-c)-kx*s],
                        [kz*kx*(1-c)-ky*s, kz*ky*(1-c)+kx*s, c+kz**2*(1-c)]])
        rot = rot.T
        self.rot_matrix = rot@self.rot_matrix if not absolute else self.rot_matrix@rot
        self.direction = [0,0,1]@self.rot_matrix
        self.update_transforms()


### UTILS ###

def rotation_matrix(start:np.ndarray, end:np.ndarray, rot_angle:float=np.pi) -> np.ndarray:
    """
    Compute the rotation matrix that aligns the vector `start` to the vector `end`.

    Parameters
    ----------
    start : np.ndarray
        The starting vector.
    end : np.ndarray
        The ending vector.

    Returns
    -------
    np.ndarray
        The rotation matrix that aligns `start` with `end`.

    Notes
    -----
    The function normalizes the input vectors and computes the rotation matrix using the formula,
    derivated from Rodrigues' rotation formula:
    `R = 2 * (sum @ sum.T) / (sum.T @ sum) - I`
    where `sum` is the sum of the normalized `start` and `end` vectors, and `I` is the identity matrix.
    """
    start = np.array([start]).T/np.linalg.norm(start)
    end = np.array([end]).T/np.linalg.norm(end)
    sum = start + end
    if np.linalg.norm(sum) == 0:
        mat = -np.eye(len(start))
    else:
        mat = 2*sum@sum.T/(sum.T@sum) - np.eye(len(start))
    rot = np.array([[np.cos(rot_angle) , np.sin(rot_angle), 0],
                    [-np.sin(rot_angle), np.cos(rot_angle), 0],
                    [0                 , 0                , 1]])
    return rot@mat

def transform_eq(eq, pos:np.ndarray=np.zeros(3), rot:np.ndarray=np.eye(3)):
    """
    Transforms a given equation by applying a translation and a rotation.

    Parameters
    ----------
    eq : function
        The equation to be transformed. It should be a function of three variables (x, y, z).
    pos : array-like, optional
        The translation vector. Default is a zero vector of length 3.
    rot : array-like, optional
        The rotation matrix. Default is the identity matrix of size 3x3.

    Returns
    -------
    function
        The transformed equation as a function of three variables (x, y, z).
    """
    invert_rot = np.linalg.inv(rot)
    eq_trans = lambda x,y,z: eq(*np.tensordot(np.array([x,y,z])-pos, invert_rot, axes=1))
    return eq_trans

def calc_hitbox(eq, eq_bounds, size:float=1e3, expand:float=0.1) -> dict:
        """
        Calculate the hitbox for a given equation and its bounds.

        Parameters
        ----------
        eq : function
            The equation function that defines the surface.
        eq_bounds : function
            The equation function that defines the bounds of the surface.
        expand : float, optional
            The amount to expand the hitbox by (default is 0.1).

        Returns
        -------
        dict
            A dictionary with keys 'x', 'y', and 'z', each containing a tuple
            representing the minimum and maximum bounds of the hitbox in that dimension.
        """
        cons = ({'type': 'eq', 'fun': lambda x: eq(*x)},
                {'type': 'ineq', 'fun': lambda x: -eq_bounds(*x)})
        x_min = optimize.minimize(lambda x: x[0], x0=np.full(3, -size), constraints=cons, tol=1e-6).x[0]
        x_max = optimize.minimize(lambda x: -x[0], x0=np.full(3, size), constraints=cons, tol=1e-6).x[0]
        y_min = optimize.minimize(lambda x: x[1], x0=np.full(3, -size), constraints=cons, tol=1e-6).x[1]
        y_max = optimize.minimize(lambda x: -x[1], x0=np.full(3, size), constraints=cons, tol=1e-6).x[1]
        z_min = optimize.minimize(lambda x: x[2], x0=np.full(3, -size), constraints=cons, tol=1e-6).x[2]
        z_max = optimize.minimize(lambda x: -x[2], x0=np.full(3, size), constraints=cons, tol=1e-6).x[2]
        hitbox = {'x':(x_min-expand, x_max+expand), 'y':(y_min-expand, y_max+expand), 'z':(z_min-expand, z_max+expand)}
        return hitbox