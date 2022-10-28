import scipy.sparse as sps

def preconditioners(solver: str, M: sps.dok_matrix) -> sps.dok_matrix:
    
    if solver == 'sbigc':
        A_prc = sps.linalg.spilu(M)
    elif solver == 'spcg':
        