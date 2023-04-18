import numpy as np
from pymatgen.symmetry import analyzer
from utils import (
	find_equivalent_positions, 
	prune_too_close_pos, 
	get_poslist1_not_in_list2
	)


def cluster_unique_sites(pk_list,mu_list,enrg_list,p_st, p_smag = None):
    """
    Function that clusters + get symmetry unique muon positions
    from list of muon sites from relax calculations.
    
    The clustering is in three steps.
    Step1: Prune equivalent (same position) positions in the list
           to a distance threshold of 0.5 Angstrom and energy difference within 0.05 eV.
           
    Step2: Find and remove magnetically+symmetrically (using p_smag ) eqvivalent sites 
           within symmetry tolerance of 0.05 Angstrom and energy difference within 0.05 eV.
    
    Step3: Check to see if all the magnetically inquivalent sites of given muon list are
            all captured, else find them and give new listof magnetically inequivalent sites 
            to be calculated.         
    
    Params:
        pk_list: list of the pk_lists corresponding to the calc. that gives the muon sites
        mu_list: list of the muon sites in fractional coordinates
        enrg_list: list of their corresponding relative DFT energies in units of eV
        p_st: A pymatgen "unitcell" structure instance 
        p_smag: A pymatgen "magnetic unitcell" structure instance 
        
    Returns:
          (i) list of symmterically unique muon positions from the initial 
                list (mu_list) provided. The corresponding pk_lists and 
                energies in eV are returned as well
          (ii) list of magnetically inequivalent positions to be sent
               back to the daemon for relaxations.
          
    """
    assert len(pk_list) == len(mu_list) == len(enrg_list)
    
    #if no magnetic symmetry
    if p_smag is None:
        p_smag = p_st
    
    #assert pymatgen structure instance
    
    #points to consider:
    #1what of when we have only the mcif? how to we get and decide the corresponding cif?
    #2We can set two sets of threshold, normal and loose?
    #3For step 3 of checking  non present magnetic inequival sites,
    #we can decide to check this only for sites with energy less than 0.6 eV? 

    #Normal thresholds
    d_tol = 0.5      #inter-site distance tolerance for clustering in Ang 
    s_tol = 0.05     #symmetry tolerance for clustering in Ang 
    e_tol = 0.05     #energy difference tolerance for clustering in eV
    a_tol = 1e-3     # symmetry tolerance for printing equi sites in Ang
    
    #Step1
    idx = prune_too_close_pos(mu_list, p_smag, d_tol,enrg_list)
    mu_list2 = mu_list[idx == np.arange(len(mu_list))]
    enrg_list2 = enrg_list[idx == np.arange(len(enrg_list))]
    pk_list2 = pk_list[idx == np.arange(len(pk_list))]

    #Step 2
    ieq = find_equivalent_positions(mu_list2, p_smag, s_tol, energies = enrg_list, e_tol = e_tol)
    mu_list3 = mu_list2[ieq == np.arange(len(mu_list2))]
    enrg_list3 = enrg_list2[ieq == np.arange(len(enrg_list2))]
    pk_list3 = pk_list2[ieq == np.arange(len(pk_list2))]
    
    #The cluster/unque positions from the given muon list
    clus_pos = list(zip(pk_list3,mu_list3,enrg_list3))
    clus_pos_sorted =sorted(clus_pos, key=lambda x:x[2])
    
    
    #Step 3: Now check if there are magnetic inequivalent sites not in the given list.
    # we can decide to check this only for sites with energy less than 0.6 eV?
    spg = analyzer.SpacegroupAnalyzer(p_st)
    ops=spg.get_symmetry_operations(cartesian=False)
    
    new_pos_to_calc = []
    for i,pp in enumerate(mu_list3):
        #get all the equivalent positions with unitcell symmetry
        pos = [x.operate(pp)%1 for x in ops]
        pos = np.unique(pos, axis=0)
        
        #find magnetically inequivalent in pos
        ieq_l = find_equivalent_positions(pos, p_smag, atol = a_tol)
        pos2 = pos[ieq_l == np.arange(len(pos))]
        
        #if magnetically inequivalent pos. exists
        if len(pos2) > 1:
            #check to see if already in the given muon list
            new_pos = get_poslist1_not_in_list2(pos2, mu_list,  host_lattice = p_st, d_tol = d_tol)
            
            if new_pos.any():
            	for i in new_pos:
            		new_pos_to_calc.append(i.tolist())
                
    
    return  clus_pos_sorted, new_pos_to_calc