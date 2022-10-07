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

        self.A: np.ndarray = self.init_enum_matrix(self.N1, self.N2, self.N3)

        
    
    
    def init_enum_matrix(self, N1: int, N2: int, N3: int) -> np.ndarray:
        A = np.arange(N1*N2*N3)
        A = np.reshape(A, (N1,N2,N3))
        return A


    def init_wall_normal_points(self):
        pass


    def init_grid(self):
        pass


    def init_channel_flow(self):
        pass

    
    @staticmethod
    def new_grid_velocity():
        pass