
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

def data_query(mp_api_key):
    """
    Queries batteries data from the Materials Project using the new API v3.

    Parameters
    ----------
    mp_api_key : str
        The API key for Materials Project.
        
    Returns
    -------
    dataframe : pandas.DataFrame
        DataFrame with the queried Batteries materials and their properties.
    """
    with MPRester(mp_api_key) as mpr:
        results = mpr.insertion_electrodes.search(
            fields=[
                "battery_id", "battery_formula", "working_ion", "num_steps", "max_voltage_step",
                "nelements", "chemsys", "formula_anonymous", "formula_charge", "formula_discharge",
                "max_delta_volume", "average_voltage", "capacity_grav", "capacity_vol", "energy_grav",
                "energy_vol", "fracA_charge", "fracA_discharge", "stability_charge", "stability_discharge",
                "id_charge", "id_discharge", "host_structure"
            ]
        )

    # Convert results to DataFrame
    data = []
    for result in results:
        try:
            # ✅ Correctly retrieve the discharged structure
            cif_structure = mpr.materials.get_structure_by_material_id(result.id_discharge).to(fmt="cif") if result.id_discharge else None
        except:
            cif_structure = None  # Handle missing structures gracefully

        entry = {
            "battery_id": result.battery_id,
            "battery_formula": result.battery_formula,
            "working_ion": result.working_ion,
            "num_steps": result.num_steps,
            "max_voltage_step": result.max_voltage_step,
            "nelements": result.nelements,
            "chemsys": result.chemsys,
            "formula_anonymous": result.formula_anonymous,
            "formula_charge": result.formula_charge,
            "formula_discharge": result.formula_discharge,
            "max_delta_volume": result.max_delta_volume,
            "average_voltage": result.average_voltage,
            "capacity_grav": result.capacity_grav,
            "capacity_vol": result.capacity_vol,
            "energy_grav": result.energy_grav,
            "energy_vol": result.energy_vol,
            "fracA_charge": result.fracA_charge,
            "fracA_discharge": result.fracA_discharge,
            "stability_charge": result.stability_charge,
            "stability_discharge": result.stability_discharge,
            "id_charge": result.id_charge,
            "id_discharge": result.id_discharge,
            #  Now uses the **discharged** structure (Li-inserted)
            "cif": cif_structure
        }
        data.append(entry)

    dataframe = pd.DataFrame(data).reset_index(drop=True)
    return dataframe


def FTCP_represent(dataframe, return_Nsites=False):
    """
    Converts battery materials into their FTCP representations.
    Parameters
    ----------
    dataframe : pandas dataframe
        Dataframe containing batteries to be converted; 
        CIFs need to be included under column 'cif'.
    return_Nsites : bool, optional
        Whether to return number of sites to be used in the error calculation
        of reconstructed site coordinate matrix
    
    Returns
    -------
    FTCP : numpy ndarray
        FTCP representation as numpy array for batteries in the dataframe.

    """
    import warnings
    warnings.filterwarnings("ignore")

    max_elms = dataframe["cif"].apply(lambda x: len(set(Structure.from_str(x, fmt="cif").atomic_numbers))).max()

    max_sites = dataframe["cif"].apply(lambda x: len(Structure.from_str(x, fmt="cif").sites)).max()

    print(f"🔍 Adjusted max_elms to {max_elms}, max_sites to {max_sites} based on dataset.")

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
        return FTCP, max_elms, max_sites
    else:
        return FTCP, np.array(Nsites), max_elms, max_sites 
