import joblib
import json
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from pymatgen.core import Structure
from mp_api.client import MPRester  # New API Client

tqdm = partial(tqdm, position=0, leave=True)

def data_query(mp_api_key, max_elms=3, min_elms=3, max_sites=20, include_te=False):
    """
    Queries data from the Materials Project using the new API v3.

    Parameters
    ----------
    mp_api_key : str
        The API key for Materials Project.
    max_elms : int, optional
        Maximum number of components/elements for crystals to be queried. Default is 3.
    min_elms : int, optional
        Minimum number of components/elements for crystals to be queried. Default is 3.
    max_sites : int, optional
        Maximum number of sites for crystals to be queried. Default is 20.
    include_te : bool, optional
        Whether to include thermoelectric properties. Default is False.

    Returns
    -------
    dataframe : pandas.DataFrame
        DataFrame with the queried materials and their properties.
    """
    with MPRester(mp_api_key) as mpr:
        results = mpr.materials.summary.search(
            num_elements=(min_elms, max_elms),  # Exactly 3 elements
            energy_above_hull=(0, 0.08),  # Stability filter
            fields=[
                "material_id", "formation_energy_per_atom", "band_gap",
                "formula_pretty", "energy_above_hull", "composition_reduced",
                "symmetry", "structure", "nsites"
            ]
        )
    
    # Convert results to DataFrame
    data = []
    for result in results:
        entry = {
            "material_id": result.material_id,
            "formation_energy_per_atom": result.formation_energy_per_atom,
            "band_gap": result.band_gap,
            "pretty_formula": result.formula_pretty,
            "e_above_hull": result.energy_above_hull,
            "elements": result.composition_reduced,
            "spacegroup.number": result.symmetry.number if result.symmetry else None,
            "cif": result.structure.to(fmt="cif") if result.structure else None,
            "nsites": result.nsites
        }
        data.append(entry)
    
    dataframe = pd.DataFrame(data)
    dataframe = dataframe[dataframe["nsites"] <= max_sites].reset_index(drop=True)  # Apply max_sites filter manually
    
    if include_te:
        te = pd.read_csv('data/thermoelectric_prop.csv', index_col=0).dropna()
        ind = dataframe.index.intersection(te.index)
        dataframe = pd.concat([dataframe, te.loc[ind, :]], axis=1)
        dataframe['Seebeck'] = dataframe['Seebeck'].apply(np.abs)
    
    return dataframe


def FTCP_represent(dataframe, max_elms=3, max_sites=20, return_Nsites=False):
    '''
    This function represents crystals in the dataframe to their FTCP representations.

    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe containing cyrstals to be converted; 
        CIFs need to be included under column 'cif'.
    max_elms : int, optional
        Maximum number of components/elements for crystals in the dataframe. 
        The default is 3.
    max_sites : int, optional
        Maximum number of sites for crystals in the dataframe.
        The default is 20.
    return_Nsites : bool, optional
        Whether to return number of sites to be used in the error calculation
        of reconstructed site coordinate matrix
    
    Returns
    -------
    FTCP : numpy ndarray
        FTCP representation as numpy array for crystals in the dataframe.

    '''
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Read string of elements considered in the study
    elm_str = joblib.load('data/element.pkl')
    # Build one-hot vectors for the elements
    elm_onehot = np.arange(1, len(elm_str)+1)[:,np.newaxis]
    elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()
    
    # Read elemental properties from atom_init.json from CGCNN (https://github.com/txie-93/cgcnn)
    with open('data/atom_init.json') as f:
        elm_prop = json.load(f)
    elm_prop = {int(key): value for key, value in elm_prop.items()}
    
    # Initialize FTCP array
    FTCP = []
    if return_Nsites:
        Nsites = []
    # Represent dataframe
    op = tqdm(dataframe.index)
    for idx in op:
        op.set_description('representing data as FTCP ...')
        
        crystal = Structure.from_str(dataframe['cif'][idx],fmt="cif")
        
        # Obtain element matrix
        elm, elm_idx = np.unique(crystal.atomic_numbers, return_index=True)
        # Sort elm to the order of sites in the CIF
        site_elm = np.array(crystal.atomic_numbers)
        elm = site_elm[np.sort(elm_idx)]
        # Zero pad element matrix to have at least 3 columns
        ELM = np.zeros((len(elm_onehot), max(max_elms, 3),))
        ELM[:, :len(elm)] = elm_onehot[elm-1,:].T
        
        # Obtain lattice matrix
        latt = crystal.lattice
        LATT = np.array((latt.abc, latt.angles))
        LATT = np.pad(LATT, ((0, 0), (0, max(max_elms, 3)-LATT.shape[1])), constant_values=0)
        
        # Obtain site coordinate matrix
        SITE_COOR = np.array([site.frac_coords for site in crystal])
        # Pad site coordinate matrix up to max_sites rows and max_elms columns
        SITE_COOR = np.pad(SITE_COOR, ((0, max_sites-SITE_COOR.shape[0]), 
                                       (0, max(max_elms, 3)-SITE_COOR.shape[1])), constant_values=0)
        
        # Obtain site occupancy matrix
        elm_inverse = np.zeros(len(crystal), dtype=int) # Get the indices of elm that can be used to reconstruct site_elm
        for count, e in enumerate(elm):
            elm_inverse[np.argwhere(site_elm == e)] = count
        SITE_OCCU = OneHotEncoder().fit_transform(elm_inverse[:,np.newaxis]).toarray()
        # Zero pad site occupancy matrix to have at least 3 columns, and max_elms rows
        SITE_OCCU = np.pad(SITE_OCCU, ((0, max_sites-SITE_OCCU.shape[0]),
                                       (0, max(max_elms, 3)-SITE_OCCU.shape[1])), constant_values=0)
        
        # Obtain elemental property matrix
        ELM_PROP = np.zeros((len(elm_prop[1]), max(max_elms, 3),))
        ELM_PROP[:, :len(elm)] = np.array([elm_prop[e] for e in elm]).T
        
        # Obtain real-space features; note the zero padding is to cater for the distance of k point in the reciprocal space
        REAL = np.concatenate((ELM, LATT, SITE_COOR, SITE_OCCU, np.zeros((1, max(max_elms, 3))), ELM_PROP), axis=0)
        
        # Obtain FTCP matrix
        recip_latt = latt.reciprocal_lattice_crystallographic
        # First use a smaller radius, if not enough k points, then proceed with a larger radius
        hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.297, zip_results=False)
        if len(hkl) < 60:
            hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.4, zip_results=False)
        # Drop (000)
        not_zero = g_hkl!=0
        hkl = hkl[not_zero,:]
        g_hkl = g_hkl[not_zero]
        # Convert miller indices to be type int
        hkl = hkl.astype('int16')
        # Sort hkl
        hkl_sum = np.sum(np.abs(hkl),axis=1)
        h = -hkl[:,0]
        k = -hkl[:,1]
        l = -hkl[:,2]
        hkl_idx = np.lexsort((l,k,h,hkl_sum))
        # Take the closest 59 k points (to origin)
        hkl_idx = hkl_idx[:59]
        hkl = hkl[hkl_idx,:]
        g_hkl = g_hkl[hkl_idx]
        # Vectorized computation of (k dot r) for all hkls and fractional coordinates
        k_dot_r = np.einsum('ij,kj->ik', hkl, SITE_COOR[:, :3]) # num_hkl x num_sites
        # Obtain FTCP matrix
        F_hkl = np.matmul(np.pad(ELM_PROP[:,elm_inverse], ((0, 0),
                                                           (0, max_sites-len(elm_inverse))), constant_values=0),
                          np.pi*k_dot_r.T)
        
        # Obtain reciprocal-space features
        RECIP = np.zeros((REAL.shape[0], 59,))
        # Prepend distances of k points to the FTCP matrix in the reciprocal-space features
        RECIP[-ELM_PROP.shape[0]-1, :] = g_hkl
        RECIP[-ELM_PROP.shape[0]:, :] = F_hkl
        
        # Obtain FTCP representation, and add to FTCP array
        FTCP.append(np.concatenate([REAL, RECIP], axis=1))
        
        if return_Nsites:
            Nsites.append(len(crystal))
    FTCP = np.stack(FTCP)
    
    if not return_Nsites:
        return FTCP
    else:
        return FTCP, np.array(Nsites)
