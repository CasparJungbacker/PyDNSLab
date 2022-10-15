import numpy as np

class Fields:

    def __init__(
        self,
        case: dict
    ) -> None:

        self.N1: int = int(2*case['res']*case['w_scale'])
        self.N2: int = int(2*case['res']*case['l_scale'])
        self.N3: int = case['res']+2

        self.length: float= 4*np.pi*case['l_scale']
        self.width: float = 2*np.pi*case['w_scale']
        self.height: float = 2

        self.dx: float = self.length/self.N2
        self.dy: float = self.width/self.N1
        self.dz: float = self.height/(self.N3-2)

        self.x: np.ndarray = self.dx*np.arange(1, self.N2+1) - self.dx/2
        self.y: np.ndarray = self.dy*np.arange(1, self.N1+1) - self.dy/2
        self.z: np.ndarray = np.zeros(self.N3)

        self.A: np.ndarray = self.init_enum_matrix(self.N1, self.N2, self.N3)

        self.FX: np.ndarray = np.empty((self.N1, self.N2, self.N3))
        self.FX.fill(self.dx)
        self.FY: np.ndarray = np.empty((self.N1, self.N2, self.N3))
        self.FY.fill(self.dy)
        self.FZ: np.ndarray = np.zeros((self.N1, self.N2, self.N3))
        
        self._init_wall_normal_points()

        X, Y, Z = np.meshgrid(self.x, self.y, self.z)
        self.X: np.ndarray = X[:,:,1:self.N3-1]
        self.Y: np.ndarray = Y[:,:,1:self.N3-1]
        self.Z: np.ndarray = Z[:,:,1:self.N3-1]

        self.inx: np.ndarray = np.arange(self.N2)
        self.iny: np.ndarray = np.arange(self.N1)
        self.inz: np.ndarray = np.arange(1, self.N3 - 1)
    
        self.north: np.ndarray = self.iny+1
        self.south: np.ndarray = self.iny-1
        self.east: np.ndarray = self.inx+1
        self.west: np.ndarray = self.inx-1

        self.north[self.N1-1] = 0
        self.south[0] = self.N1 - 1
        self.east[self.N2-1] = 0
        self.west[0] = self.N2-1

        self.air: np.ndarray = self.inz + 1
        self.ground: np.ndarray = self.inz - 1

        self.U = np.zeros((self.N1, self.N2, self.N3-2))
        self.V = np.copy(self.U)
        self.W = np.copy(self.U)
        
        self._init_channel_flow(case['runmode'],
                                case['u_nom'],
                                case['u_f'])
        
        self.A0 = self.A[self.iny, self.inx, self.inz] - self.N1*self.N2
        self.AN = self.A[self.north, self.inx, self.inz] - self.N1*self.N2
        self.AS = self.A[self.south, self.inx, self.inz] - self.N1*self.N2
        self.AE = self.A[self.iny, self.east, self.inz] - self.N1*self.N2
        self.AW = self.A[self.iny, self.west, self.inz] - self.N1*self.N2
        self.AA = self.A[self.iny, self.inx, self.air] - self.N1*self.N2
        self.AG = self.A[self.iny, self.inx, self.ground] - self.N1*self.N2

        self.u = np.reshape(self.U, (self.N1*self.N2*(self.N3-2), 1))
        self.v = np.reshape(self.V, (self.N1*self.N2*(self.N3-2), 1))
        self.w = np.reshape(self.W, (self.N1*self.N2*(self.N3-2), 1))

        self.pold = np.zeros((self.N1*self.N2*(self.N3-2), 1))
        
        
    def init_enum_matrix(self, N1: int, N2: int, N3: int) -> np.ndarray:
        A = np.arange(N1*N2*N3)
        A = np.reshape(A, (N1,N2,N3))
        return A


    def _init_wall_normal_points(self) -> None:
        fz = np.arange(-(self.N3/2 - 1), self.N3/2)
        fz = np.tanh(5e-2*fz)
        fz = fz - fz[0]

        self.z = np.zeros(len(fz) + 1)
        self.z[0] = -(fz[1] - fz[0])*0.5
        self.z[1:-1] = fz[0:-1] + 0.5*(fz[1:] - fz[0:-1])
        self.z[-1] = fz[-1] + 0.5*(fz[-1] - fz[-2])
        
        self.z = (self.z/fz[-1])*self.height
        fz = (fz/fz[-1])*self.height
        
        # TODO: figure out a better way to do this
        for i in range(1, self.N3 - 1):
            self.FZ[:,:,i] = fz[i] - fz[i-1]

        self.FZ[:,:,0] = self.FZ[:,:,1]
        self.FZ[:,:,-1] = self.FZ[:,:,-2]


    def _init_channel_flow(self,
                           runmode: int,
                           u_nom: float,
                           u_f: float) -> None:
        if runmode == 0:
            self.U[:, :, :] = u_nom
        elif runmode == 1:
            UF1 = u_f*(np.random.rand(self.N1, self.N2, self.N3 - 2) - 0.5)
            UF2 = u_f*(np.random.rand(self.N1, self.N2, self.N3 - 2) - 0.5)
            UF3 = u_f*(np.random.rand(self.N1, self.N2, self.N3 - 2) - 0.5)
            self.U += UF1*np.amax(UF1)
            self.V += UF2*np.amax(UF2)
            self.W += UF3*np.amax(UF3)
        elif runmode == 2:
            raise NotImplementedError
        else:
            raise ValueError('Invalid runmode.')
