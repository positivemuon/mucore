import numpy as np
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.symmetry import analyzer


def find_equivalent_positions(frac_coords, host_lattice, atol=1e-3, energies = None, e_tol=0.05):
    """
    Returns eqivalent atoms list of
    Energies and energy tolerance (e_tol) are in eV
    
    """

    lattice = host_lattice.lattice
    # Bring to unit cell
    frac_coords %= 1
    
    # prepare list of equivalent atoms. -1 mean "not yet analyzed".
    eq_list = np.zeros(len(frac_coords), dtype=np.int32) - 1
    
    spg = analyzer.SpacegroupAnalyzer(host_lattice, symprec=atol)
    
    ops = spg.get_symmetry_operations()
    
    # This hosts all the equivalent positions obtained for each of the
    # lattice points using all symmetry operations.
    eq_pos = np.zeros([len(ops), len(frac_coords), 3])
    
    for i, op in enumerate(ops):
        eq_pos[i] = op.operate_multi(frac_coords) % 1
        
    # Compute equivalence list
    for i in range(len(frac_coords)):
        if eq_list[i] >= 0:
            continue
        
        for j in range(i, len(frac_coords)):
            diff = pbc_shortest_vectors(
                lattice, eq_pos[:, j, :], frac_coords[i]
            ).squeeze()
            if (energies is not None) and (len(energies) == len(frac_coords)):
                if (np.linalg.norm(diff, axis=1) < atol).any() and (abs(energies[i]- energies[j]) < e_tol):
                    eq_list[j] = i
            else:
                if (np.linalg.norm(diff, axis=1) < atol).any():
                    eq_list[j] = i
    return eq_list
            
    
def prune_too_close_pos(frac_positions, host_lattice, min_distance, energies = None, e_tol=0.05):
    """Returns index of too close atoms"""
    #energies and tolerance should be in eV
    lattice = host_lattice.lattice
    
    s_idx  = np.arange(len(frac_positions))
    
    for i,pi in enumerate(frac_positions):
        for j,pj in enumerate(frac_positions):
            if j > i:
                diff = pbc_shortest_vectors(lattice, pi, pj).squeeze()
                #print(i,j,diff,np.linalg.norm(diff, axis=0))
                if (energies is not None) and (len(energies) == len(frac_positions)):
                    if (np.linalg.norm(diff, axis=0) < min_distance) \
                        and (abs(energies[i]- energies[j]) < e_tol):
                        s_idx[j]=-1
                else:
                    if np.linalg.norm(diff, axis=0) < min_distance:
                        s_idx[j]=-1
                    
    #frac_positions = np.delete(frac_positions,s_idx,0) #use if append
    #frac_positions = frac_positions[s_idx == np.arange(len(frac_positions))]
    return s_idx


def get_poslist1_not_in_list2(pos_lst1, pos_lst2, host_lattice, d_tol=0.5):
    """
    Function that compares two position list
    and returns position of list1 not in list2 
    """
    lattice = host_lattice.lattice
    s_idx  = np.zeros(len(pos_lst1), dtype=np.int32) - 1
    for i,pi in enumerate(pos_lst1):
        for j, pj in enumerate(pos_lst2):
            diff = pbc_shortest_vectors(lattice, pi, pj).squeeze()
            if np.linalg.norm(diff, axis=0) < d_tol:
                s_idx[i]= i
    
    pos_not_in_list = pos_lst1[s_idx != np.arange(len(pos_lst1))]
    return pos_not_in_list
