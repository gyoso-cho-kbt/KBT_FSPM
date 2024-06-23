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