import joblib, os
import numpy as np
from tqdm import tqdm
from ase.io import write
from ase import spacegroup
from mp_api.client import MPRester  # ✅ Updated API v3 Import

def get_info(ftcp_designs, 
             max_elms=3, 
             max_sites=20, 
             elm_str=joblib.load('data/element.pkl'),
             to_CIF=True,
             check_uniqueness=True,
             mp_api_key=None,
             ):
    
    '''
    Gets chemical information for designed FTCP representations,
    i.e., formulas, lattice parameters, site fractional coordinates.

    Parameters
    ----------
    ftcp_designs : numpy ndarray
        Designed FTCP representations for decoded sampled latent points.
    max_elms : int, optional
        Maximum number of elements per designed crystal (default = 3).
    max_sites : int, optional
        Maximum number of sites per designed crystal (default = 20).
    elm_str : list of element symbols, optional
        List of allowed elements (default = "elements.pkl").
    to_CIF : bool, optional
        Whether to save CIFs in "designed_CIFs" (default = True).
    check_uniqueness : bool, optional
        Whether to check if designed formulas exist in Materials Project.
    mp_api_key : str, optional
        API key for Materials Project.

    Returns
    -------
    pred_formula : list
        List of predicted chemical formulas.
    pred_abc : numpy.ndarray
        Predicted lattice constants (abc).
    pred_ang : numpy.ndarray
        Predicted lattice angles (alpha, beta, gamma).
    pred_latt : numpy.ndarray
        Combined lattice parameters (abc + angles).
    pred_site_coor : list
        List of predicted site coordinates.
    ind_unique : list
        Indices of unique formulas not in Materials Project (if `check_uniqueness=True`).
    '''
    
    Ntotal_elms = len(elm_str)
    pred_elm = np.argmax(ftcp_designs[:, :Ntotal_elms, :max_elms], axis=1)
    
    def get_formula(ftcp_designs):
        pred_formula = []
        pred_site_occu = ftcp_designs[:, Ntotal_elms+2+max_sites:Ntotal_elms+2+2*max_sites, :max_elms]
        temp = np.repeat(np.expand_dims(np.max(pred_site_occu, axis=2), axis=2), max_elms, axis=2)
        pred_site_occu[pred_site_occu < temp] = 0
        pred_site_occu[pred_site_occu < 0.05] = 0
        pred_site_occu = np.ceil(pred_site_occu)
        
        for i in range(len(ftcp_designs)):
            site_elements = pred_site_occu[i].dot(pred_elm[i])
            if np.all(site_elements == 0):
                pred_formula.append([elm_str[0]])  # Default element if empty
            else:
                temp = site_elements[:np.where(site_elements > 0)[0][-1] + 1].tolist()
                pred_formula.append([elm_str[int(j)] for j in temp])
        return pred_formula
    
    pred_formula = get_formula(ftcp_designs)
    pred_abc = ftcp_designs[:, Ntotal_elms, :3]
    pred_ang = ftcp_designs[:, Ntotal_elms+1, :3]
    pred_latt = np.concatenate((pred_abc, pred_ang), axis=1)
    pred_site_coor = [ftcp_designs[i, Ntotal_elms+2:Ntotal_elms+2+max_sites, :3] for i in range(len(ftcp_designs))]

    if check_uniqueness:
        assert mp_api_key is not None, "You need an MP API key to check uniqueness!"

        mpr = MPRester(mp_api_key)
        ind = []
        op = tqdm(range(len(pred_formula)))

        for i in op:
            op.set_description("Checking uniqueness of designed compositions...")
            formula_str = "".join(pred_formula[i])  # Convert list to string formula
            query = mpr.materials.summary.search(formula=formula_str, fields=["material_id"])
            if not query:  # If query is empty, the formula is unique
                ind.append(i)
    else:
        ind = list(np.arange(len(pred_formula)))

    if to_CIF:
        os.makedirs("designed_CIFs", exist_ok=True)
        op = tqdm(ind)

        for i, j in enumerate(op):
            op.set_description("Writing designed crystals as CIFs...")
            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=pred_site_coor[j],
                                             cellpar=pred_latt[j])
                write(f"designed_CIFs/{i}.cif", crystal)
            except Exception:
                pass  # Ignore errors during CIF creation
    
    return (pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, ind) if check_uniqueness else \
           (pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor)
