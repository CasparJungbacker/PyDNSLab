import scipy as sp
import scipy.sparse as sps
import numpy as np

from pydnslab.createfields import Fields


class Operators:
    """Input: probably a Fields object
    """
    def __init__(self):
        pass

        
    @staticmethod
    def differentiate_1(
        N1: int,
        N2: int,
        N3: int,
        FX: np.ndarray,
        FY: np.ndarray,
        FZ: np.ndarray,
        inz: np.ndarray,
        inx: np.ndarray,
        iny: np.ndarray,
        A0: np.ndarray,
        AN: np.ndarray,
        AS: np.ndarray,
        AE: np.ndarray,
        AW: np.ndarray,
        AA: np.ndarray,
        AG: np.ndarray,
        east: np.ndarray,
        west: np.ndarray,
        north: np.ndarray,
        south: np.ndarray,
        air: np.ndarray,
        ground: np.ndarray,
        index: int
        ) -> sps.dia_matrix:
        """First order derivatives
        """
        m = np.zeros(N1*N2*(N3-2))
        M = sps.spdiags(
            [m, m, m, m, m, m, m],
            [-N1*(N3 - 2), -N1, -1, 0, 1, N1, N1*(N3 - 2)],
            N1*N2*(N3-2),
            N1*N2*(N3-2))

        for i in iny:
            for j in inx:
                for k in inz - 1:
                    FY0 = FY[i,j,k+1]
                    FYN = FY[north[i],j,k+1]
                    FYS = FY[south[i],j,k+1]
            
                    FX0 = FX[i,j,k+1]
                    FXE = FX[i,east[j],k+1]
                    FXW = FX[i,west[j],k+1]
            
                    FZ0 = FZ[i,j,k+1]
                    FZA = FZ[i,j,air[k]]
                    FZG = FZ[i,j,ground[k]]

                if index == 1:
                    M[A0[i,j,k], A0[i,j,k]] = (1/FY0)*(FYN/(FY0+FYN)-\
                        FYS/(FYS+FY0))
                if index == 2:
                    M[A0[i,j,k], A0[i,j,k]] = (1/FX0)*(FXE/(FX0+FXE)-\
                        FXW/(FXW+FX0))
                if index == 3:
                    M[A0[i,j,k], A0[i,j,k]] = (1/FZ0)*(FZA/(FZ0+FZA)-\
                        FZG/(FZG+FZ0))
                    
        return M

    
    @staticmethod
    def differentiate_2():
        """Second order derivatives
        """
        pass
    
    
    @staticmethod
    def differentiate_1p():
        """First order pressure derivatives
        """ 
        pass

    
    @staticmethod
    def poisson_matrix():
        """Poisson operator
        """
        pass
