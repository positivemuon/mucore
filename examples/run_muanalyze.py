import numpy as np
from pymatgen.core import Structure

import sys
sys.path.append('../')

from muanalyze import cluster_unique_sites

def mu_file_5col_read(filename):
    """Just to read a 5 colum mu file"""
    f = open(filename,'r')
    lbl = []
    mu_frac_coord = []
    enrg = []
    for data in f:
        flot = [float(x) for x in data.split()[1:5]]
        lb1 = [str(y)for y in data.split()[0:1]]
        lbl.append(lb1[0])
        mu_frac_coord.append([flot[0],flot[1],flot[2]])
        enrg.append(flot[3])
    f.close()
    lbl = np.array(lbl)
    mu_frac_coord = np.array(mu_frac_coord)
    enrg = np.array(enrg)
    return lbl, mu_frac_coord, enrg



if __name__ == '__main__':
    """run Examples for MnO and BaFe2As2"""
    #run cluster MnO
    s = Structure.from_file('data/mno2.cif')
    smag = Structure.from_file('data/MnO.mcif')
    lbl,mu_frac_coord,enrg = mu_file_5col_read('data/mno_+_musites.txt')
    cluster_u, to_calc = cluster_unique_sites(lbl,mu_frac_coord,enrg,s,smag)
    print("cluster",cluster_u)
    print("sites_to_calc",to_calc)
    print(f"{len(to_calc)} new sites have to be calculated")

    #when magnetic symmetry not considered for MnO
    cluster_u, to_calc = cluster_unique_sites(lbl,mu_frac_coord,enrg,s)
    print("---------------")
    print("cluster",cluster_u)
    print("sites_to_calc",to_calc)
    print(f"{len(to_calc)} new sites have to be calculated")
    
      

    #run cluster BaFe2As2
    s = Structure.from_file('data/BaFeAs1_4123005.cif') #wrong unitcell cif
    #s = Structure.from_file('data/1.16_BaFe2As2.mcif') #use mcif
    smag = Structure.from_file('data/1.16_BaFe2As2.mcif')
    lbl, mu_frac_coord,enrg = mu_file_5col_read('data/BaFe2As2_ucel.txt')
    cluster_u, to_calc = cluster_unique_sites(lbl,mu_frac_coord,enrg,s,smag)
    print("---------------")
    print("cluster",cluster_u)
    print("sites_to_calc",to_calc)
    print(f"{len(to_calc)} new sites have to be calculated")