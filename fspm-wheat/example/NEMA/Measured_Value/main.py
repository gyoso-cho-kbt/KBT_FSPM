# -*- coding: latin-1 -*-

from __future__ import print_function
import datetime
import logging
import os
import random
import time
import sys

import numpy as np
import pandas as pd

from alinea.adel.adel_dynamic import AdelDyn
from cnwheat import tools as cnwheat_tools
from fspmwheat import cnwheat_facade, farquharwheat_facade, senescwheat_facade, growthwheat_facade, caribu_facade, elongwheat_facade

from openalea.plantgl.all import Viewer, Scene
from openalea.plantgl.math import Vector3
from math import radians

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename=r'C:\Users\gyoso.cho.TIRD\source\WheatFspm\WheatFspm\fspm-wheat\example\NEMA\Measured_Value\mylog.log', level=logging.INFO, filemode='a')


# authored by zhao
# poprogate some property from the reference plant to the plants in the canopy
def propogate_plant_properties(ref_g, canopy_g, some_property='is_green'):
    from openalea.mtg import traversal
    for plant_no in canopy_g.component_roots(0):
        for ref_no, tar_no in zip(traversal.iter_mtg2(ref_g, 1), traversal.iter_mtg2(canopy_g, plant_no)):
            if ref_no in ref_g.property(some_property).keys() and tar_no in canopy_g.property(some_property).keys():
                canopy_g.property(some_property)[tar_no] = ref_g.property(some_property)[ref_no]
    return canopy_g

# authored by zhao
# an utility function for adding base/top element into MTG so that the update_geometry function in adel_wheat becomes available.
def insert_base_top_element(g):
    from openalea.mtg import MTG
    newg = MTG()
    root_id = 0
    for plant_id_iter in g.components_iter(root_id): # plant level =1
        plant_label = g.node(plant_id_iter).label
        plant_properties = g.node(plant_id_iter).properties()
        new_plant_id = newg.add_component(root_id, **plant_properties)

        for axis_id_iter in g.components_iter(plant_id_iter): # axis level =2
            axis_label = g.node(axis_id_iter).label
            axis_properties = g.node(axis_id_iter).properties()
            new_axis_id = newg.add_component(new_plant_id, **axis_properties) # add phytomer to the axis
            
            # only operating on the 'MS' axis
            if axis_label.startswith('MS'):
                metamer0_properties = {}
                metamer_id = newg.add_component(new_axis_id, edge_type='/', label='metamer0', **metamer0_properties)
                collar_id = newg.add_component(metamer_id, edge_type='/', label='collar')
                baseEle_id = newg.add_component(collar_id, edge_type='/', label='baseElement')
                topEle_id = newg.add_child(baseEle_id, edge_type='<', label='topElement')      # baseElement < topElement
            else:
                continue

            for phytomer_id_iter in g.components_iter(axis_id_iter): # phytomer level =3
                new_parent_metamer_id = max(newg.components(new_axis_id)) # newg's maximum metamer id is the parent metamer id

                metamer_label = g.node(phytomer_id_iter).label
                metamer_properties = g.node(phytomer_id_iter).properties()
                new_metamer_id = newg.add_component(new_axis_id, **metamer_properties)

                # link with its previous phytomer
                newg.add_child(new_parent_metamer_id, new_metamer_id, edge_type='<')

                for organ_id_iter in g.components_iter(phytomer_id_iter): # organ level =4
                    new_parent_organ_id = None
                    if newg.components(new_metamer_id):  # components list of the newly added metamer
                        new_parent_organ_id = max( newg.components(new_metamer_id) )
                        
                    organ_label = g.node(organ_id_iter).label
                    organ_properties = g.node(organ_id_iter).properties()
                    # if organ_label.startswith('sheath'):
                    #     organ_properties['edge_type'] = '+'  # '+' type for sheath node
                    # else:
                    #     organ_properties['edge_type'] = '<'

                    new_organ_id = newg.add_component(new_metamer_id, **organ_properties)
                    # insert baseElement and topElement for each organ
                    newbaseEle_id = newg.add_component(new_organ_id, edge_type='/', label='baseElement') 
                    newtopEle_id = newg.add_child(newbaseEle_id, edge_type='<', label='topElement')
                    
                    if new_parent_organ_id: # in case of parent organ exists, link with the parent organ, also link at the Element level
                        new_parent_top_ele_id = [ele_id for ele_id in newg.components(new_parent_organ_id) if newg.node(ele_id).label.startswith('topElement')][0] # get the topElement in parent organ.
                        if organ_properties['label'].startswith('sheath'): # sheath case where edge_type is '+'
                            newg.add_child(new_parent_organ_id, new_organ_id, edge_type='+')
                            newg.add_child(new_parent_top_ele_id, newbaseEle_id, edge_type='+')
                        else:
                            newg.add_child(new_parent_organ_id, new_organ_id, edge_type='<')
                            newg.add_child(new_parent_top_ele_id, newbaseEle_id, edge_type='<')
                    else:
                        # in case of no parent organ exists in the current phytomer(i.e. the internode organ), 
                        # conduct the linkage at its parent phytomer, specifically linking with the first 
                        # (rather than the last, because it is not sure how far the phytomer is extended) organ in its parent phytomer, as well as the elements in the parent organ. 
                        new_first_parent_phytomer_organ_id = min(newg.components(new_parent_metamer_id))
                        newg.add_child(new_first_parent_phytomer_organ_id, new_organ_id, edge_type='<')
                        new_parent_top_ele_id = [ele_id for ele_id in newg.components(new_first_parent_phytomer_organ_id) if newg.node(ele_id).label.startswith('topElement')][0]
                        newg.add_child(new_parent_top_ele_id, newbaseEle_id, edge_type='<')
                        
                    all_ele_nodes_id = [ e_id for e_id in g.components_iter(organ_id_iter)] 
                    new_child_ele_id = newtopEle_id
                    for ele_id_iter in reversed(all_ele_nodes_id): # element level =5
                        ele_label = g.node(ele_id_iter).label
                        ele_properties = g.node(ele_id_iter).properties()
                        ele_properties['edge_type'] = '<'
                        new_child_ele_id = newg.insert_parent(new_child_ele_id, **ele_properties)
    
    return newg


# authored by zhao
# an utility function for altering the label name 'pedoncule' (French) in MTG to the name 'peduncle' (English)
def label_alter_function(gg):
    """
    change the label name from 'pedoncule' (french) to 'peduncle' (english)
    """
    alter_list = [ vid for vid, label_name in gg.properties()['label'].items() if label_name=='pedoncule']
    for vid in alter_list:
        gg.node(vid).label = 'peduncle'
        
    return gg

"""
    main
    ~~~~

    Script readatpted from example NEMA_H3 used in the paper Barillot et al. (2016).
    This example uses the format MTG to exchange data between the models.

    You must first install :mod:`alinea.adel`, :mod:`cnwheat`, :mod:`farquharwheat` and :mod:`senescwheat` (and add them to your PYTHONPATH)
    before running this script with the command `python`.

    :copyright: Copyright 2014-2016 INRA-ECOSYS, see AUTHORS.
    :license: TODO, see LICENSE for details.

    .. seealso:: Barillot et al. 2016.
"""

random.seed(1234)
np.random.seed(1234)

HOUR_TO_SECOND_CONVERSION_FACTOR = 3600

INPUTS_DIRPATH = 'inputs'
GRAPHS_DIRPATH = 'graphs'#'graphs'
# GRAPHS_DIRPATH = 'graphs_fructan_0_origin'

# adelwheat inputs at t0
ADELWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'adelwheat')  # the directory adelwheat must contain files 'adel0000.pckl' and 'scene0000.bgeom'

# cnwheat inputs at t0
CNWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'cnwheat')
CNWHEAT_PLANTS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'plants_inputs.csv')
CNWHEAT_AXES_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'axes_inputs.csv')
CNWHEAT_METAMERS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'metamers_inputs.csv')
CNWHEAT_ORGANS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'organs_inputs_calibrated.csv')
CNWHEAT_HIDDENZONE_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
CNWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')
CNWHEAT_SOILS_INPUTS_FILEPATH = os.path.join(CNWHEAT_INPUTS_DIRPATH, 'soils_inputs_calibrated.csv')

# farquharwheat inputs at t0
FARQUHARWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'farquharwheat')
FARQUHARWHEAT_INPUTS_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'inputs.csv')
FARQUHARWHEAT_AXES_INPUTS_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'SAM_inputs.csv')
METEO_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'weather_data_kumada_20240410-0510.csv')
# CARIBU_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'inputs_eabs.csv')
# alter the caribu file path for interpolation of the real data. The file must couple with the 'calculate_PARa_by_interploation_df' function
CARIBU_FILEPATH = os.path.join(FARQUHARWHEAT_INPUTS_DIRPATH, 'inputs_Eabs(origin version).csv')

# elongwheat inputs at t0
ELONGWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'elongwheat')
ELONGWHEAT_HZ_INPUTS_FILEPATH = os.path.join(ELONGWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
ELONGWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(ELONGWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')
ELONGWHEAT_AXES_INPUTS_FILEPATH = os.path.join(ELONGWHEAT_INPUTS_DIRPATH, 'SAM_inputs.csv')

# senescwheat inputs at t0
SENESCWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'senescwheat')
SENESCWHEAT_ROOTS_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'roots_inputs.csv')
SENESCWHEAT_AXES_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'SAM_inputs.csv')
SENESCWHEAT_ELEMENTS_INPUTS_FILEPATH = os.path.join(SENESCWHEAT_INPUTS_DIRPATH, 'elements_inputs.csv')

# growthwheat inputs at t0
GROWTHWHEAT_INPUTS_DIRPATH = os.path.join(INPUTS_DIRPATH, 'growthwheat')
GROWTHWHEAT_HIDDENZONE_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'hiddenzones_inputs.csv')
GROWTHWHEAT_ORGANS_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'organs_inputs.csv')
GROWTHWHEAT_ROOTS_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'roots_inputs.csv')
GROWTHWHEAT_AXES_INPUTS_FILEPATH = os.path.join(GROWTHWHEAT_INPUTS_DIRPATH, 'SAM_inputs.csv')

# the path of the CSV files where to save the states of the modeled system at each step
OUTPUTS_DIRPATH = 'outputs'
AXES_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'axes_states.csv')
ORGANS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'organs_states.csv')
ELEMENTS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'elements_states.csv')
SOILS_STATES_FILEPATH = os.path.join(OUTPUTS_DIRPATH, 'soils_states.csv')

# post-processing directory path
POSTPROCESSING_DIRPATH = 'postprocessing'
AXES_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'axes_postprocessing.csv')
ORGANS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'organs_postprocessing.csv')
HIDDENZONES_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'hiddenzones_postprocessing.csv')
ELEMENTS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'elements_postprocessing.csv')
SOILS_POSTPROCESSING_FILEPATH = os.path.join(POSTPROCESSING_DIRPATH, 'soils_postprocessing.csv')

AXES_INDEX_COLUMNS = ['t', 'plant', 'axis']
ELEMENTS_INDEX_COLUMNS = ['t', 'plant', 'axis', 'metamer', 'organ', 'element']
ORGANS_INDEX_COLUMNS = ['t', 'plant', 'axis', 'organ']
SOILS_INDEX_COLUMNS = ['t', 'plant', 'axis']

# Define culm density (culm m-2)
# DENSITY = 410.
DENSITY = 110 # zhao: kumada data
NPLANTS = 1
CULM_DENSITY = {i: DENSITY / NPLANTS for i in range(1, NPLANTS + 1)}

INPUTS_OUTPUTS_PRECISION = 5  # 10

LOGGING_CONFIG_FILEPATH = os.path.join('logging.json')

LOGGING_LEVEL = logging.INFO  # can be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

cnwheat_tools.setup_logging(LOGGING_CONFIG_FILEPATH, LOGGING_LEVEL, log_model=False, log_compartments=False, log_derivatives=False)


def calculate_PARa_by_interploation_df(g, Eabs_df, t, PARi):
    """
    Compute PARa by interpolate within Eabs_df
    """
    interplote_func = lambda init_val, ripen_val, t: np.interp(t, [calculate_PARa_by_interploation_df.start_t, calculate_PARa_by_interploation_df.ripen_t], [init_val, ripen_val]) if (t>=calculate_PARa_by_interploation_df.start_t and t<calculate_PARa_by_interploation_df.ripen_t) else ripen_val
    Eabs_df_grouped = Eabs_df.groupby(['plant', 'metamer', 'organ'])
    CARIBU_ELEMENTS_NAMES = {'StemElement', 'LeafElement1'}
    PARa_element_data_dict = {}
    for mtg_plant_vid in g.components_iter(g.root):
        mtg_plant_index = int(g.index(mtg_plant_vid))
        for mtg_axis_vid in g.components_iter(mtg_plant_vid):
            for mtg_metamer_vid in g.components_iter(mtg_axis_vid):
                mtg_metamer_index = int(g.index(mtg_metamer_vid))
                for mtg_organ_vid in g.components_iter(mtg_metamer_vid):
                    mtg_organ_label = g.label(mtg_organ_vid)
                    for mtg_element_vid in g.components_iter(mtg_organ_vid):
                        mtg_element_label = g.label(mtg_element_vid)
                        if mtg_element_label not in CARIBU_ELEMENTS_NAMES:
                            continue
                        element_id = (mtg_plant_index, mtg_metamer_index, mtg_organ_label)
                        if element_id in Eabs_df_grouped.groups.keys():
                            if PARi == 0:
                                PARa_element_data_dict[mtg_element_vid] = 0
                            else:
                                # # for debug
                                # if mtg_metamer_index==12 and mtg_organ_label == 'blade':
                                    # print('debug flag leaf eabs:', interplote_func(Eabs_df_grouped.get_group(element_id)['Eabs_init'].iloc[0], Eabs_df_grouped.get_group(element_id)['Eabs_ripen'].iloc[0], t))
                                PARa_element_data_dict[mtg_element_vid] = interplote_func(Eabs_df_grouped.get_group(element_id)['Eabs_init'].iloc[0], Eabs_df_grouped.get_group(element_id)['Eabs_ripen'].iloc[0], t) * PARi

    return PARa_element_data_dict



def calculate_PARa_from_df(g, Eabs_df, PARi, multiple_sources=False, ratio_diffus_PAR=None):
    """
    Compute PARa from an input dataframe having Eabs values.
    """

    Eabs_df_grouped = Eabs_df.groupby(['plant', 'metamer', 'organ'])

    #: the name of the elements modeled by FarquharWheat
    CARIBU_ELEMENTS_NAMES = {'StemElement', 'LeafElement1'}

    PARa_element_data_dict = {}
    # traverse the MTG recursively from top ...
    for mtg_plant_vid in g.components_iter(g.root):
        mtg_plant_index = int(g.index(mtg_plant_vid))
        for mtg_axis_vid in g.components_iter(mtg_plant_vid):
            for mtg_metamer_vid in g.components_iter(mtg_axis_vid):
                mtg_metamer_index = int(g.index(mtg_metamer_vid))
                for mtg_organ_vid in g.components_iter(mtg_metamer_vid):
                    mtg_organ_label = g.label(mtg_organ_vid)
                    for mtg_element_vid in g.components_iter(mtg_organ_vid):
                        mtg_element_label = g.label(mtg_element_vid)
                        if mtg_element_label not in CARIBU_ELEMENTS_NAMES:
                            continue
                        element_id = (mtg_plant_index, mtg_metamer_index, mtg_organ_label)
                        if element_id in Eabs_df_grouped.groups.keys():
                            if PARi == 0:
                                PARa_element_data_dict[mtg_element_vid] = 0
                            elif multiple_sources:
                                PARa_diffuse = Eabs_df_grouped.get_group(element_id)['Eabs_diffuse'].iloc[0] * PARi * ratio_diffus_PAR
                                PARa_direct = Eabs_df_grouped.get_group(element_id)['Eabs_direct'].iloc[0] * PARi * (1 - ratio_diffus_PAR)
                                PARa_element_data_dict[mtg_element_vid] = PARa_diffuse + PARa_direct
                            else:
                                PARa_element_data_dict[mtg_element_vid] = Eabs_df_grouped.get_group(element_id)['Eabs'].iloc[0] * PARi

    return PARa_element_data_dict

# authored by zhao
# iterate over the leaf elements and stem elements in internode
# if the green length is shorter than the senesced_length_element
# then set the 'is_green' to false for display purpose
def display_seneced_element(g):
    for vid in g.vertices_iter(5):
        node = g.node(vid)
        if node.length and node.length > 0:
            if node.label.startswith('Leaf') or (node.label.startswith('Stem') and g.node(g.complex(vid)).label.startswith('inter')):
                # senesced_length = node.properties().get('senesced_length_element', 0)
                # green_length = node.length - senesced_length
                # node.is_green = min(max(1 - float(senesced_length/node.length), 0.0),1.0)
                node.is_green = min(max(node.green_area/node.area,0.0),1.0)
                if node.label.startswith('Leaf'):
                    logger.info('{} area:{}, new_green_area: {}, ratio: {}'.format(node._vid, node.area, node.green_area, node.green_area/node.area))
    return g
            


def main(stop_time, run_simu=True, make_graphs=True):
    if run_simu:
        meteo = pd.read_csv(METEO_FILEPATH, index_col='t')
        Eabs_df = pd.read_csv(CARIBU_FILEPATH)

        # define the time step in hours for each simulator
        senescwheat_ts = 2
        growthwheat_ts = 2
        farquharwheat_ts = 2
        elongwheat_ts = 2
        cnwheat_ts = 1

        hour_to_second_conversion_factor = 3600

        # read adelwheat inputs at t0
        from alinea.adel.Stand import AgronomicStand
        stand = AgronomicStand(sowing_density=500, plant_density=500, inter_row=0.08)
        adel_wheat = AdelDyn(seed=1234, stand=stand, aspect='square', scene_unit='m', nplants = 500, duplicate=10)
        g = adel_wheat.load(dir=ADELWHEAT_INPUTS_DIRPATH)
        
        # zhao for test of the utilization function
        g = label_alter_function(insert_base_top_element(g))
        g = adel_wheat.update_geometry(g) # NE FONCTIONNE PAS car MTG non compatible (pas de top et base element)
        # zhao: record the length for the peduncle and ear to reset them after loading data from files.
        record_peduncle_length = g.property('length')[203]
        record_ear_length = g.property('length')[209]
        
        # create empty dataframes to shared data between the models
        shared_axes_inputs_outputs_df = pd.DataFrame()
        shared_organs_inputs_outputs_df = pd.DataFrame()
        shared_hiddenzones_inputs_outputs_df = pd.DataFrame()
        shared_elements_inputs_outputs_df = pd.DataFrame()
        shared_soils_inputs_outputs_df = pd.DataFrame()

        # read the inputs at t0 and create the facades

        # caribu
        caribu_facade_ = caribu_facade.CaribuFacade(g,
                                                    shared_elements_inputs_outputs_df,
                                                    adel_wheat)

        # senescwheat
        senescwheat_roots_inputs_t0 = pd.read_csv(SENESCWHEAT_ROOTS_INPUTS_FILEPATH)
        senescwheat_axes_inputs_t0 = pd.read_csv(SENESCWHEAT_AXES_INPUTS_FILEPATH)
        senescwheat_elements_inputs_t0 = pd.read_csv(SENESCWHEAT_ELEMENTS_INPUTS_FILEPATH)
        update_senescwheat_parameters = None
        # update_senescwheat_parameters = {'FRACTION_N_MAX' : {'blade': 0.002, 'stem': 0.00175}} # zhao: lower the senesce threshold to keep the blade alive.
        #################### 2024/05/16 calibrated ###########################
        update_senescwheat_parameters = {'SENESCENCE_MAX_RATE': 9.5E-10}
        #####################################################################
        senescwheat_facade_ = senescwheat_facade.SenescWheatFacade(g,
                                                                   senescwheat_ts * hour_to_second_conversion_factor,
                                                                   senescwheat_roots_inputs_t0,
                                                                   senescwheat_axes_inputs_t0,
                                                                   senescwheat_elements_inputs_t0,
                                                                   shared_organs_inputs_outputs_df,
                                                                   shared_axes_inputs_outputs_df, 
                                                                   shared_elements_inputs_outputs_df, 
                                                                   update_parameters=update_senescwheat_parameters, 
                                                                   option_static=False)
        # growthwheat
        growthwheat_hiddenzones_inputs_t0 = pd.read_csv(GROWTHWHEAT_HIDDENZONE_INPUTS_FILEPATH)
        growthwheat_organ_inputs_t0 = pd.read_csv(GROWTHWHEAT_ORGANS_INPUTS_FILEPATH)
        growthwheat_root_inputs_t0 = pd.read_csv(GROWTHWHEAT_ROOTS_INPUTS_FILEPATH)
        growthwheat_axes_inputs_t0 = pd.read_csv(GROWTHWHEAT_AXES_INPUTS_FILEPATH)
        growthwheat_facade_ = growthwheat_facade.GrowthWheatFacade(g,
                                                                   growthwheat_ts * hour_to_second_conversion_factor,
                                                                   growthwheat_hiddenzones_inputs_t0,
                                                                   growthwheat_organ_inputs_t0,
                                                                   growthwheat_root_inputs_t0,
                                                                   growthwheat_axes_inputs_t0,
                                                                   shared_organs_inputs_outputs_df,
                                                                   shared_hiddenzones_inputs_outputs_df,
                                                                   shared_elements_inputs_outputs_df,
                                                                   shared_axes_inputs_outputs_df)

        # farquharwheat
        farquharwheat_elements_inputs_t0 = pd.read_csv(FARQUHARWHEAT_INPUTS_FILEPATH)
        farquharwheat_axes_inputs_t0 = pd.read_csv(FARQUHARWHEAT_AXES_INPUTS_FILEPATH)
        # Use the initial version of the photosynthesis sub-model (as in Barillot et al. 2016, and in Gauthier et al. 2020)
        # try increase the 'Sna_Vcmax25' to improve the efficiency of the photosynthesis
        # decrease the 'DELTA_CONVERGENCE' for refine the convergency
        # adapt the PARAM_TEMP for actual meteorlogical data for rice.
        update_parameters_farquharwheat = {
        'SurfacicProteins': False, 'NSC_Retroinhibition': False,
        'DELTA_CONVERGENCE': 0.000001,
        #################### 2023/10/27 parameter adjustment based on information from kano san ##########################################
        # 'PARAM_TEMP': {'deltaHa': {'TPU': 47 + 26}, 'deltaHd': {'TPU': 152.3 + 3}, 'Tref': 298.15-5 },
        # 'Ap_A': 3 * 0.15, 'Vomax_A': 0.5 * 300, 'Aj_A' : 2,
        
        #################### 2023/11/02 parameter adaption based on kano san information, and use Ac constrain rather than Ap #################
        #################### 2023/12/04 add modification of {'deltaHa':'Jmax'} and {'deltaHd':'Jmax'} to correct the temperature repsonse of Ap #########
        # 'PARAM_TEMP': {'deltaHa': {'Vc_max': 89.7 + 8, 'Jmax': 48.9 + 5}, 'deltaHd':{'Vc_max': 149.3 + 5, 'Jmax': 152.3+1}, 'Tref': 298.15 + 5},
        # 'PARAM_N': {'S_surfacic_nitrogen': { 'Vc_max25': 84.965 - 62}},
        }  

        farquharwheat_facade_ = farquharwheat_facade.FarquharWheatFacade(g,
                                                                         farquharwheat_elements_inputs_t0,
                                                                         farquharwheat_axes_inputs_t0,
                                                                         shared_elements_inputs_outputs_df,
                                                                         update_parameters=update_parameters_farquharwheat)

        # elongwheat # Only for temperature related computations
        elongwheat_hiddenzones_inputs_t0 = pd.read_csv(ELONGWHEAT_HZ_INPUTS_FILEPATH)
        elongwheat_elements_inputs_t0 = pd.read_csv(ELONGWHEAT_ELEMENTS_INPUTS_FILEPATH)
        elongwheat_axes_inputs_t0 = pd.read_csv(ELONGWHEAT_AXES_INPUTS_FILEPATH)

        elongwheat_facade_ = elongwheat_facade.ElongWheatFacade(g,
                                                                elongwheat_ts * hour_to_second_conversion_factor,
                                                                elongwheat_axes_inputs_t0,
                                                                elongwheat_hiddenzones_inputs_t0,
                                                                elongwheat_elements_inputs_t0,
                                                                shared_axes_inputs_outputs_df,
                                                                shared_hiddenzones_inputs_outputs_df,
                                                                shared_elements_inputs_outputs_df,
                                                                adel_wheat, option_static=True)
        # cnwheat
        cnwheat_organs_inputs_t0 = pd.read_csv(CNWHEAT_ORGANS_INPUTS_FILEPATH)
        cnwheat_hiddenzones_inputs_t0 = pd.read_csv(CNWHEAT_HIDDENZONE_INPUTS_FILEPATH)
        cnwheat_elements_inputs_t0 = pd.read_csv(CNWHEAT_ELEMENTS_INPUTS_FILEPATH)
        cnwheat_soils_inputs_t0 = pd.read_csv(CNWHEAT_SOILS_INPUTS_FILEPATH)
        # zhao memo: lower the root K_AMINO_ACIDS_EXPORT/K_NITRATE_EXPORT (from 25*3E-5/25*1E-6 -> 1E-6/1E-7) in attempt to lower the N content in phloem
        # set offset to -9 in (cn)model.modified_Arrhenius_equation to acclerate the grain growth (i.e. the age_from_flowering) a bit compared to -11.5
        # increase 'SIGMA_AMINO_ACIDS' (1e-07 -> 1) to accelerate the N translocation in blade.
        # update_cnwheat_parameters = {'roots': {'K_AMINO_ACIDS_EXPORT':  1E-6, #1E-4, #25*3E-5, #5E-4, 
                                               # 'K_NITRATE_EXPORT': 1E-7, #1E-6, #25*1E-6, #5E-6,  
                                               # }, 
                                     # 'PhotosyntheticOrgan': {'VMAX_SFRUCTAN_POT': 0, 
                                                             # 'SIGMA_AMINO_ACIDS': 1,
                                                             # },
                                     # }
        update_cnwheat_parameters = {
            # 'PhotosyntheticOrgan': {'VMAX_SFRUCTAN_POT': 0, },
            # 'grains': {'FILLING_INIT': 240*3600, 'FILLING_END':(240+540)*3600},
            ########### 2024/06/15 calibrated ######################################
            'grains': {'FILLING_INIT': 240*3600, 'FILLING_END':(240+540)*3600}, #'VMAX_STARCH':0.25},
            'roots': {'K_AMINO_ACIDS_EXPORT': 3E-9, 'K_NITRATE_EXPORT': 1E-9, }, #'N_EXUDATION_MAX':1, },
            # 'PhotosyntheticOrgan': {'VMAX_SFRUCTAN_POT': 0, 'SIGMA_SUCROSE': 1e-01}#, 'SIGMA_AMINO_ACIDS': 1e-2},
            #######################################################################
            }

        cnwheat_facade_ = cnwheat_facade.CNWheatFacade(g,
                                                       cnwheat_ts * hour_to_second_conversion_factor,
                                                       CULM_DENSITY,
                                                       update_cnwheat_parameters,
                                                       cnwheat_organs_inputs_t0,
                                                       cnwheat_hiddenzones_inputs_t0,
                                                       cnwheat_elements_inputs_t0,
                                                       cnwheat_soils_inputs_t0,
                                                       shared_axes_inputs_outputs_df,
                                                       shared_organs_inputs_outputs_df,
                                                       shared_hiddenzones_inputs_outputs_df,
                                                       shared_elements_inputs_outputs_df,
                                                       shared_soils_inputs_outputs_df)
                                                       
        # define organs for which the variable 'max_proteins' is fixed
        # zhao: elements in the 'forced_max_protein_elements' is allowed to senesce.
        # forced_max_protein_elements = {(1, 'MS', 9, 'blade', 'LeafElement1'), (1, 'MS', 10, 'blade', 'LeafElement1'), (1, 'MS', 11, 'blade', 'LeafElement1'), (2, 'MS', 9, 'blade', 'LeafElement1'),
                                       # (2, 'MS', 10, 'blade', 'LeafElement1'), (2, 'MS', 11, 'blade', 'LeafElement1')}
        # zhao: also add internode to senesce                               
        forced_max_protein_elements = {(1, 'MS', 9, 'blade', 'LeafElement1'), (1, 'MS', 10, 'blade', 'LeafElement1'), (1, 'MS', 11, 'blade', 'LeafElement1')}

        # zhao: add internode length manully, and reset the length for the peduncle and ear, for these are wiredly updated.
        g.property('length')[184] = .11
        g.property('length')[168] = .12
        g.property('length')[152] = .081
        g.property('length')[136] = .079
        g.property('length')[203] = record_peduncle_length
        g.property('length')[209] = record_ear_length
        g.property('position')[1] = (0,0,0)
        g = adel_wheat.update_geometry(g)

        # manually set the 'width' according to the measured values on the element level
        g.property('width')[197] = 0.0178
        g.property('width')[167+14] = 0.0174
        g.property('width')[151+14] = 0.0166
        g.property('width')[135+14] = 0.0142
        # g.property('shape_max_width')[183+11] = 0.0105
        # g.property('shape_max_width')[167+11] = 0.00925
        # g.property('shape_max_width')[151+11] = 0.00775
        # g.property('shape_max_width')[135+11] = 0.00850
        # sheath
        # g.property('width')[183+9] = 0.00330
        # g.property('width')[167+9] = 0.00410
        # g.property('width')[151+9] = 0.00565
        # g.property('width')[135+9] = 0.00620
        # g.property('diameter')[183+6] = 0.00330
        # g.property('diameter')[167+6] = 0.00410
        # g.property('diameter')[151+6] = 0.00565
        # g.property('diameter')[135+6] = 0.00620
        # internode
        # g.property('width')[183+4] = 0.00330
        # g.property('width')[167+4] = 0.00410
        # g.property('width')[151+4] = 0.00565
        # g.property('width')[135+4] = 0.00620
        # g.property('diameter')[183+1] = 0.00330
        # g.property('diameter')[167+1] = 0.00410
        # g.property('diameter')[151+1] = 0.00565
        # g.property('diameter')[135+1] = 0.00620
        
        # zhao: set the max protein leaf as 0, because its 'update_max_protein' is also True, senesce (should) not happen to it
        ############ 2024/5/16 calibrated ######################################
        g.property('max_proteins')[167+14] = 0 # the second leaf
        ########################################################################
        # g.property('max_proteins')[197] = 0 # the flag leaf
        
        # define the start and the end of the whole simulation (in hours)
        # start_time = 252   # (h)
        # stop_time = start_time + 35*24  # stop_time=252+35*24=1092 the span for actual measurement is 35 days
        start_time = 0
        
       
        # # manually set the interpolation time interval
        # calculate_PARa_by_interploation_df.start_t = 0
        # calculate_PARa_by_interploation_df.ripen_t = 25*24  # the middle ripen time is set as the 25th day.

        # define lists of dataframes to store the inputs and the outputs of the models at each step.
        axes_all_data_list = []
        organs_all_data_list = []  # organs which belong to axes: roots, phloem, grains
        elements_all_data_list = []
        soils_all_data_list = []

        all_simulation_steps = []  # to store the steps of the simulation

        # run the simulators
        current_time_of_the_system = time.time()

        # zhao: animation settings
        # Viewer.animation(True)
        # radii = 1.4
        # theta = -24  # azimuth angle (Y axis -> X axis)
        # phi = 63     # elevation angle
        # radii = 1.3
        # theta = -27
        # phi = 60
        # position_vector = Vector3(Vector3.Spherical(radii, radians(theta), radians(phi)))
        # Viewer.camera.setPerspective()
        # Viewer.camera.setPosition(position_vector)
        # Viewer.camera.lookAt(Vector3(0,0,0.6))
        # Viewer.frameGL.setBgColor(170,170,170)  # default value is (170,170,170)
        # canopy_g = adel_wheat.duplicated(g) # zhao: copy the single plant to create a canopy
        try:
            for t_elongwheat in range(start_time, stop_time, elongwheat_ts):  # Only to compute temperature related variable
                # run ElongWheat
                print('t elongwheat is {}'.format(t_elongwheat))
                Tair, Tsoil = meteo.loc[t_elongwheat, ['air_temperature', 'air_temperature']]
                elongwheat_facade_.run(Tair, Tsoil, option_static=True)

                for t_senescwheat in range(t_elongwheat, t_elongwheat + elongwheat_ts, senescwheat_ts):
                    # run SenescWheat
                    print('t senescwheat is {}'.format(t_senescwheat))
                    # zhao: forcelly set the max_protein of blade in phytomer 10 as 0 to activate senesce
                    if t_senescwheat > 288:
                        # g.node(151+14).max_proteins = 0
                        ########################### 2024/5/16 calibrated ############################
                        g.node(197).max_proteins = 0 # flag leaf, set on the senescence
                        forced_max_protein_elements = {(1, 'MS', 9, 'blade', 'LeafElement1'), (1, 'MS', 10, 'blade', 'LeafElement1'), (1, 'MS', 11, 'blade', 'LeafElement1'), (1, 'MS', 12, 'blade', 'LeafElement1')}
                        #############################################################################
                        
                    senescwheat_facade_.run(forced_max_protein_elements, postflowering_stages=True, option_static=False)
                    
                    ############# zhao: save the snapshot for animation ######################
                    if t_senescwheat % 24 == 0:
                        logger.info('t:{}'.format(t_senescwheat))
                        g = display_seneced_element(g)
                        # canopy_g = propogate_plant_properties(g, canopy_g, 'is_green')
                        # adel_wheat.plot(canopy_g)
                        # Viewer.saveSnapshot('./animation/{:d}.png'.format(t_senescwheat//24))
                        # Viewer.stop()
                    ######################################################
                    # Test for fully senesced shoot tissues  #TODO: Make the model to work even if the whole shoot is dead but the roots are alived
                    if sum(senescwheat_facade_._shared_elements_inputs_outputs_df['green_area']) <= 0.25E-6:
                        break

                    for t_growthwheat in range(t_senescwheat, t_senescwheat + senescwheat_ts, growthwheat_ts):
                        # run GrowthWheat
                        print('t growthwheat is {}'.format(t_growthwheat))
                        growthwheat_facade_.run(postflowering_stages=True)

                        for t_farquharwheat in range(t_growthwheat, t_growthwheat + growthwheat_ts, farquharwheat_ts):
                            # get the meteo of the current step
                            Tair, ambient_CO2, RH, Ur, PARi = meteo.loc[t_farquharwheat, ['air_temperature', 'ambient_CO2', 'humidity', 'Wind', 'PARi']]
                            # get PARa for current step
                            ## zhao: try interpolation between real values, the function must couple with the inputs_Eabs_interpolation.csv file.
                            aggregated_PARa = calculate_PARa_from_df(g, Eabs_df, PARi, multiple_sources=False)
                            # aggregated_PARa = calculate_PARa_by_interploation_df(g, Eabs_df, t_farquharwheat, PARi)
                            
                            print('t caribu is {}'.format(t_farquharwheat))
                            # caribu_facade_.run(energy=PARi,sun_sky_option='sky')
                            caribu_facade_.update_shared_MTG({'PARa': aggregated_PARa})
                            caribu_facade_.update_shared_dataframes({'PARa': aggregated_PARa})
                            # run FarquharWheat
                            print('t farquhar is {}'.format(t_farquharwheat))
                            farquharwheat_facade_.run(Tair, ambient_CO2, RH, Ur)

                            for t_cnwheat in range(t_farquharwheat, t_farquharwheat + farquharwheat_ts, cnwheat_ts):
                                Tair, Tsoil = meteo.loc[t_cnwheat, ['air_temperature', 'air_temperature']]
                                # run CNWheat
                                print('t cnwheat is {}'.format(t_cnwheat))
                                cnwheat_facade_.run(Tair=Tair, Tsoil=Tsoil)
                                # The following print statement is for debugging the updated cn module paramter. 
                                # It should be put here because the update not be executed until the _initialize_model in run function
                                # print('modified cn parameter {} is {}'.format('VMAX_RGR', cnwheat_facade_.population.plants[0].axes[0].grains.PARAMETERS.VMAX_RGR))

                                # append the inputs and outputs at current step to global lists
                                all_simulation_steps.append(t_cnwheat)
                                axes_all_data_list.append(shared_axes_inputs_outputs_df.copy())
                                organs_all_data_list.append(shared_organs_inputs_outputs_df.copy())
                                elements_all_data_list.append(shared_elements_inputs_outputs_df.copy())
                                soils_all_data_list.append(shared_soils_inputs_outputs_df.copy())

                else:
                    continue
                break

        except Exception as ex:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message, fname, exc_tb.tb_lineno)

        finally:
            execution_time = int(time.time() - current_time_of_the_system)
            print('\n', 'Simulation run in ', str(datetime.timedelta(seconds=execution_time)))
            
            adel_wheat.save(g)

            # write all inputs and outputs to CSV files
            all_axes_inputs_outputs = pd.concat(axes_all_data_list, keys=all_simulation_steps)
            all_axes_inputs_outputs.reset_index(0, inplace=True)
            all_axes_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
            all_axes_inputs_outputs.to_csv(AXES_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

            all_organs_inputs_outputs = pd.concat(organs_all_data_list, keys=all_simulation_steps)
            all_organs_inputs_outputs.reset_index(0, inplace=True)
            all_organs_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
            all_organs_inputs_outputs.to_csv(ORGANS_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

            all_elements_inputs_outputs = pd.concat(elements_all_data_list, keys=all_simulation_steps)
            all_elements_inputs_outputs.reset_index(0, inplace=True)
            all_elements_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
            all_elements_inputs_outputs.to_csv(ELEMENTS_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

            all_soils_inputs_outputs = pd.concat(soils_all_data_list, keys=all_simulation_steps)
            all_soils_inputs_outputs.reset_index(0, inplace=True)
            all_soils_inputs_outputs.rename({'level_0': 't'}, axis=1, inplace=True)
            all_soils_inputs_outputs.to_csv(SOILS_STATES_FILEPATH, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION))

    # -POST-PROCESSING##

    if make_graphs:
        generate_graphs()


def generate_graphs():

    # POST PROCESSINGS

    states_df_dict = {}
    for states_filepath in (AXES_STATES_FILEPATH,
                            ORGANS_STATES_FILEPATH,
                            ELEMENTS_STATES_FILEPATH,
                            SOILS_STATES_FILEPATH):
        # assert states_filepaths were not opened during simulation run meaning that other filenames were saved
        path, filename = os.path.split(states_filepath)
        filename = os.path.splitext(filename)[0]
        newfilename = 'ACTUAL_{}.csv'.format(filename)
        newpath = os.path.join(path, newfilename)
        assert not os.path.isfile(newpath), \
            "File {} was saved because {} was opened during simulation run. Rename it before running postprocessing".format(newfilename, states_filepath)

        # Retrieve outputs dataframes from precedent simulation run
        states_df = pd.read_csv(states_filepath)
        states_file_basename = os.path.basename(states_filepath).split('.')[0]
        states_df_dict[states_file_basename] = states_df
    time_grid = states_df_dict['elements_states']['t']
    delta_t = (time_grid.unique()[1] - time_grid.unique()[0]) * HOUR_TO_SECOND_CONVERSION_FACTOR

    # run the postprocessing
    axes_postprocessing_file_basename = os.path.basename(AXES_POSTPROCESSING_FILEPATH).split('.')[0]
    organs_postprocessing_file_basename = os.path.basename(ORGANS_POSTPROCESSING_FILEPATH).split('.')[0]
    elements_postprocessing_file_basename = os.path.basename(ELEMENTS_POSTPROCESSING_FILEPATH).split('.')[0]
    soils_postprocessing_file_basename = os.path.basename(SOILS_POSTPROCESSING_FILEPATH).split('.')[0]
    postprocessing_df_dict = {}
    (postprocessing_df_dict[axes_postprocessing_file_basename],
     _,
     postprocessing_df_dict[organs_postprocessing_file_basename],
     postprocessing_df_dict[elements_postprocessing_file_basename],
     postprocessing_df_dict[soils_postprocessing_file_basename]) \
        = cnwheat_facade.CNWheatFacade.postprocessing(axes_outputs_df=states_df_dict[os.path.basename(AXES_STATES_FILEPATH).split('.')[0]],
                                                      organs_outputs_df=states_df_dict[os.path.basename(ORGANS_STATES_FILEPATH).split('.')[0]],
                                                      hiddenzone_outputs_df=None,
                                                      elements_outputs_df=states_df_dict[os.path.basename(ELEMENTS_STATES_FILEPATH).split('.')[0]],
                                                      soils_outputs_df=states_df_dict[os.path.basename(SOILS_STATES_FILEPATH).split('.')[0]],
                                                      delta_t=delta_t)

    # save the postprocessing to disk
    for postprocessing_file_basename, postprocessing_filepath in ((axes_postprocessing_file_basename, AXES_POSTPROCESSING_FILEPATH),
                                                                  (organs_postprocessing_file_basename, ORGANS_POSTPROCESSING_FILEPATH),
                                                                  (elements_postprocessing_file_basename, ELEMENTS_POSTPROCESSING_FILEPATH),
                                                                  (soils_postprocessing_file_basename, SOILS_POSTPROCESSING_FILEPATH)):
        postprocessing_df_dict[postprocessing_file_basename].to_csv(postprocessing_filepath, na_rep='NA', index=False, float_format='%.{}f'.format(INPUTS_OUTPUTS_PRECISION + 5))

    # - GRAPHS

    # Retrieve last computed post-processing dataframes
    axes_postprocessing_file_basename = os.path.basename(AXES_POSTPROCESSING_FILEPATH).split('.')[0]
    organs_postprocessing_file_basename = os.path.basename(ORGANS_POSTPROCESSING_FILEPATH).split('.')[0]
    elements_postprocessing_file_basename = os.path.basename(ELEMENTS_POSTPROCESSING_FILEPATH).split('.')[0]
    soils_postprocessing_file_basename = os.path.basename(SOILS_POSTPROCESSING_FILEPATH).split('.')[0]
    postprocessing_df_dict = {}
    for (postprocessing_filepath, postprocessing_file_basename) in ((AXES_POSTPROCESSING_FILEPATH, axes_postprocessing_file_basename),
                                                                    (ORGANS_POSTPROCESSING_FILEPATH, organs_postprocessing_file_basename),
                                                                    (ELEMENTS_POSTPROCESSING_FILEPATH, elements_postprocessing_file_basename),
                                                                    (SOILS_POSTPROCESSING_FILEPATH, soils_postprocessing_file_basename)):
        postprocessing_df = pd.read_csv(postprocessing_filepath)
        postprocessing_df_dict[postprocessing_file_basename] = postprocessing_df

    # Generate graphs
    cnwheat_facade.CNWheatFacade.graphs(axes_postprocessing_df=postprocessing_df_dict[axes_postprocessing_file_basename],
                                        hiddenzones_postprocessing_df=None,
                                        organs_postprocessing_df=postprocessing_df_dict[organs_postprocessing_file_basename],
                                        elements_postprocessing_df=postprocessing_df_dict[elements_postprocessing_file_basename],
                                        soils_postprocessing_df=postprocessing_df_dict[soils_postprocessing_file_basename],
                                        graphs_dirpath=GRAPHS_DIRPATH)

    #
    # x_name = 't'
    # x_label='Time (Hour)'
    #
    # # 1) Photosynthetic organs
    # ph_elements_output_df = pd.read_csv(ELEMENTS_STATES_FILEPATH)
    #
    # graph_variables_ph_elements = {'PARa': u'Absorbed PAR (ｵmol m$^{-2}$ s$^{-1}$)', 'Ag': u'Gross photosynthesis (ｵmol m$^{-2}$ s$^{-1}$)','An': u'Net photosynthesis (ｵmol m$^{-2}$ s$^{-1}$)', 'Tr':u'Organ surfacic transpiration rate (mmol H$_{2}$0 m$^{-2}$ s$^{-1}$)', 'Transpiration':u'Organ transpiration rate (mmol H$_{2}$0 s$^{-1}$)', 'Rd': u'Mitochondrial respiration rate of organ in light (ｵmol C h$^{-1}$)', 'Ts': u'Temperature surface (ｰC)', 'gs': u'Conductance stomatique (mol m$^{-2}$ s$^{-1}$)',
    #                    'Conc_TriosesP': u'[TriosesP] (ｵmol g$^{-1}$ mstruct)', 'Conc_Starch':u'[Starch] (ｵmol g$^{-1}$ mstruct)', 'Conc_Sucrose':u'[Sucrose] (ｵmol g$^{-1}$ mstruct)', 'Conc_Fructan':u'[Fructan] (ｵmol g$^{-1}$ mstruct)',
    #                    'Conc_Nitrates': u'[Nitrates] (ｵmol g$^{-1}$ mstruct)', 'Conc_Amino_Acids': u'[Amino_Acids] (ｵmol g$^{-1}$ mstruct)', 'Conc_Proteins': u'[Proteins] (g g$^{-1}$ mstruct)',
    #                    'Nitrates_import': u'Total nitrates imported (ｵmol h$^{-1}$)', 'Amino_Acids_import': u'Total amino acids imported (ｵmol N h$^{-1}$)',
    #                    'S_Amino_Acids': u'[Rate of amino acids synthesis] (ｵmol N g$^{-1}$ mstruct h$^{-1}$)', 'S_Proteins': u'Rate of protein synthesis (ｵmol N g$^{-1}$ mstruct h$^{-1}$)', 'D_Proteins': u'Rate of protein degradation (ｵmol N g$^{-1}$ mstruct h$^{-1}$)', 'k_proteins': u'Relative rate of protein degradation (s$^{-1}$)',
    #                    'Loading_Sucrose': u'Loading Sucrose (ｵmol C sucrose h$^{-1}$)', 'Loading_Amino_Acids': u'Loading Amino acids (ｵmol N amino acids h$^{-1}$)',
    #                    'green_area': u'Green area (m$^{2}$)', 'R_phloem_loading': u'Respiration phloem loading (ｵmol C h$^{-1}$)', 'R_Nnit_red': u'Respiration nitrate reduction (ｵmol C h$^{-1}$)', 'R_residual': u'Respiration residual (ｵmol C h$^{-1}$)', 'R_maintenance': u'Respiration residual (ｵmol C h$^{-1}$)',
    #                    'mstruct': u'Structural mass (g)', 'Nstruct': u'Structural N mass (g)',
    #                    'Conc_cytokinins':u'[cytokinins] (UA g$^{-1}$ mstruct)', 'D_cytokinins':u'Cytokinin degradation (UA g$^{-1}$ mstruct)', 'cytokinins_import':u'Cytokinin import (UA)'}
    #
    #
    # for org_ph in (['blade'], ['sheath'], ['internode'], ['peduncle', 'ear']):
    #     for variable_name, variable_label in graph_variables_ph_elements.iteritems():
    #         graph_name = variable_name + '_' + '_'.join(org_ph) + '.PNG'
    #         cnwheat_tools.plot_cnwheat_ouputs(ph_elements_output_df,
    #                       x_name = x_name,
    #                       y_name = variable_name,
    #                       x_label=x_label,
    #                       y_label=variable_label,
    #                       filters={'organ': org_ph},
    #                       plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
    #                       explicit_label=False)
    #
    # # 2) Roots, grains and phloem
    # organs_output_df = pd.read_csv(ORGANS_STATES_FILEPATH)
    #
    # graph_variables_organs = {'Conc_Sucrose':u'[Sucrose] (ｵmol g$^{-1}$ mstruct)', 'Dry_Mass':'Dry mass (g)',
    #                     'Conc_Nitrates': u'[Nitrates] (ｵmol g$^{-1}$ mstruct)', 'Conc_Amino_Acids':u'[Amino Acids] (ｵmol g$^{-1}$ mstruct)', 'Proteins_N_Mass': u'[N Proteins] (g)',
    #                     'Uptake_Nitrates':u'Nitrates uptake (ｵmol h$^{-1}$)', 'Unloading_Sucrose':u'Unloaded sucrose (ｵmol C g$^{-1}$ mstruct h$^{-1}$)', 'Unloading_Amino_Acids':u'Unloaded Amino Acids (ｵmol N AA g$^{-1}$ mstruct h$^{-1}$)',
    #                     'S_Amino_Acids': u'Rate of amino acids synthesis (ｵmol N g$^{-1}$ mstruct h$^{-1}$)', 'S_Proteins': u'Rate of protein synthesis (ｵmol N h$^{-1}$)', 'Export_Nitrates': u'Total export of nitrates (ｵmol N h$^{-1}$)', 'Export_Amino_Acids': u'Total export of Amino acids (ｵmol N h$^{-1}$)',
    #                     'R_Nnit_upt': u'Respiration nitrates uptake (ｵmol C h$^{-1}$)', 'R_Nnit_red': u'Respiration nitrate reduction (ｵmol C h$^{-1}$)', 'R_residual': u'Respiration residual (ｵmol C h$^{-1}$)', 'R_maintenance': u'Respiration residual (ｵmol C h$^{-1}$)',
    #                     'R_grain_growth_struct': u'Respiration grain structural growth (ｵmol C h$^{-1}$)', 'R_grain_growth_starch': u'Respiration grain starch growth (ｵmol C h$^{-1}$)',
    #                     'R_growth': u'Growth respiration of roots (ｵmol C h$^{-1}$)', 'mstruct': u'Structural mass (g)', 'rate_mstruct_death': u'Rate of structural mass death (g)',
    #                     'C_exudation': u'Carbon lost by root exudation (ｵmol C g$^{-1}$ mstruct h$^{-1}$', 'N_exudation': u'Nitrogen lost by root exudation (ｵmol N g$^{-1}$ mstruct h$^{-1}$',
    #                     'Conc_cytokinins':u'[cytokinins] (UA g$^{-1}$ mstruct)', 'S_cytokinins':u'Rate of cytokinins synthesis (UA g$^{-1}$ mstruct)', 'Export_cytokinins': 'Export of cytokinins from roots (UA h$^{-1}$)',
    #                     'HATS_LATS': u'Potential uptake (ｵmol h$^{-1}$)' , 'regul_transpiration':'Regulating transpiration function'}
    #
    # for org in (['roots'], ['grains'], ['phloem']):
    #     for variable_name, variable_label in graph_variables_organs.iteritems():
    #         graph_name = variable_name + '_' + '_'.join(org) + '.PNG'
    #         cnwheat_tools.plot_cnwheat_ouputs(organs_output_df,
    #                       x_name = x_name,
    #                       y_name = variable_name,
    #                       x_label=x_label,
    #                       y_label=variable_label,
    #                       filters={'organ': org},
    #                       plot_filepath=os.path.join(GRAPHS_DIRPATH, graph_name),
    #                       explicit_label=False)
    #
    # # 3) Soil
    # soil_output_df = pd.read_csv(SOILS_STATES_FILEPATH)
    #
    # fig, (ax1) = plt.subplots(1)
    # conc_nitrates_soil = soil_output_df['Conc_Nitrates_Soil']*14E-6
    # ax1.plot(soil_output_df['t'], conc_nitrates_soil)
    # ax1.set_ylabel(u'[Nitrates] (g m$^{-3}$)')
    # ax1.set_xlabel('Time from flowering (hour)')
    # ax1.set_title = 'Conc Nitrates Soil'
    # plt.savefig(os.path.join(GRAPHS_DIRPATH, 'Conc_Nitrates_Soil.PNG'), format='PNG', bbox_inches='tight')
    # plt.close()


if __name__ == '__main__':
    main(743, run_simu=True, make_graphs=True)
    # main(20, run_simu=True, make_graphs=True)
