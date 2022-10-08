import numpy as np

class Fields:

    def __init__(
        self,
        res: int,
        w_scale: float,
        l_scale: float
    ) -> None:

        self.N1: int = int(2*res*w_scale)
        self.N2: int = int(2*res*l_scale)
        self.N3: int = res+2

        self.length: float= 4*np.pi*l_scale
        self.width: float = 2*np.pi*w_scale
        self.height: float = 2

        self.dx: float = self.length/self.N2
        self.dy: float = self.width/self.N1
        self.dz: float = self.height/(self.N3-2)

        self.x: np.ndarray = self.dx*np.arange(1, self.N2+1, self.dx)\
            - self.dx/2
        self.y: np.ndarray = self.dy*np.arange(1, self.N1+1, self.dy)\
            - self.dy/2
        self.z: np.ndarray = np.zeros(self.N3)

        self.A: np.ndarray = self.init_enum_matrix(self.N1, self.N2, self.N3)

        self.FX: np.ndarray = np.zeros(
            (self.N1, self.N2, self.N3)).fill(self.dx)
        self.FY: np.ndarray = np.zeros(
            (self.N1, self.N2, self.N3)).fill(self.dy)
        self.FZ: np.ndarray = np.zeros((self.N1, self.N2, self.N3))
        
        self.init_wall_normal_points()
    
    def init_enum_matrix(self, N1: int, N2: int, N3: int) -> np.ndarray:
        A = np.arange(N1*N2*N3)
        A = np.reshape(A, (N1,N2,N3))
        return A


    def init_wall_normal_points(self) -> None:
        fz = np.arange(-(self.N3/2 - 1), self.N3/2)
        fz = np.tanh(5e-2*fz)
        fz = fz - fz[0]

        self.z = np.zeros(len(fz) + 1)
        self.z[0] = -(fz[1] - fz[0])*0.5
        self.z[1:-1] = fz[0:-2] + 0.5*(fz[1:-1] - fz[0:-2])
        self.z[-1] = fz[-1] + 0.5*(fz[-1] - fz[-2])
        
        self.z /= fz[-1]*self.height
        fz /= fz[-1]*self.height

        self.FZ = np.zeros((self.N1, self.N2, self.N3))
        
        # TODO: figure out how to get rid of this loop
        for i in range(1, self.N3):
            self.FZ[:,:,i] = fz[i] - fz[i-1]

            
        self.FZ[:,:,0] = self.FZ[:,:,1]
        self.FZ[:,:,-1] = self.FZ[:,:,-2]


    def init_grid(self):
        pass


    def init_channel_flow(self):
        pass
