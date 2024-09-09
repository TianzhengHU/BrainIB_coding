import pandas as pd
import numpy as np

def do_the_map():
    # Load the data from the Excel file
    BrainNet_index_map_path = '/Users/hutianzheng/Desktop/Brain_IB/Visualization_and_data/statistic_data/BSNIP/BrainNet_index_map.xlsx'
    BSNIP_new_id = '/Users/hutianzheng/Desktop/Brain_IB/Visualization_and_data/GroupICA/BSNIP_new_id.xlsx'


    # Read the sheets into dataframes
    GroupICA_105 = pd.read_excel(BSNIP_new_id, sheet_name='Sheet1')
    # Read the sheets into dataframes
    BrainNet_116 = pd.read_excel(BrainNet_index_map_path, sheet_name='Sheet1')

    # Extract the coordinates
    nodes_105 = GroupICA_105[['x', 'y', 'z']].values
    nodes_116 = BrainNet_116[['x', 'y', 'z']].values
    # Extract the indices and coordinates
    nodes_105_indices = GroupICA_105['ID105'].values
    nodes_116_indices = BrainNet_116['index'].values
    nodes_116_names1 = BrainNet_116['name'].values
    nodes_116_names2 = BrainNet_116['name2'].values

    def euclidean_distance(point1, point2):
        return np.linalg.norm(point1 - point2)


    # Store the results
    results = []
    save_most_map_node = []
    for i, red_point in enumerate(nodes_105):
        distances = []
        for green_point in nodes_116:
            distance = euclidean_distance(red_point, green_point)
            distances.append(distance)
        # Get the indices of the 3 smallest distances
        closest_indices = np.argsort(distances)[:3]
        # closest_indices = [nodes_116_indices[closest_indices_in_distances]]
        closest_distances = [distances[j] for j in closest_indices]
        closest_green_points = nodes_116[closest_indices]
        closest_green_indices = nodes_116_indices[closest_indices]
        closest_green_name1 = nodes_116_names1[closest_indices]
        closest_green_name2 = nodes_116_names2[closest_indices]

        results.append((nodes_105_indices[i], red_point, closest_green_indices, closest_green_points, closest_distances))
        save_most_map_node.append({
            'Red Point': nodes_105[i],
            'Closest Green Point 1': closest_green_indices[0],
            'Distance 1': closest_distances[0],
            'Name 1': closest_green_name1[0] +"("+closest_green_name2[0]+")",
            'Closest Green Point 2': closest_green_indices[1],
            'Distance 2': closest_distances[1],
            'Name 2': closest_green_name1[1] + "(" + closest_green_name2[1] + ")",
            'Closest Green Point 3': closest_green_indices[2],
            'Distance 3': closest_distances[2],
            'Name 3': closest_green_name1[2] + "(" + closest_green_name2[2] + ")",
        })
    # Print the results
    for red_index, red_point, closest_green_indices, closest_green_points, closest_distances in results:
        print(f"Red point {red_index} ({red_point}):")
        for j in range(3):
            print(f"    Green point {closest_green_indices[j]} ({closest_green_points[j]}): Distance = {closest_distances[j]}")

    # Create a DataFrame from the results
    results_df = pd.DataFrame(save_most_map_node)
    # Save the results to an Excel file
    output_file_path = '/Users/hutianzheng/Desktop/Brain_IB/Visualization_and_data/GroupICA/closest_map_points.xlsx'
    results_df.to_excel(output_file_path, index=False)

    # Print the results (optional)
    for red_name, row in results_df.iterrows():
        print(f"Red point '{row['Red Point']}':")
        print(f"    Closest Green point '{row['Closest Green Point 1']}' at distance: {row['Distance 1']}")


# ---------before line: create the map------------

# now if we have any list of id index for the 105 aal, map it to 116 AAL



def mapping_list_to_AAl_116(target_list):
    map_file_path = "/Users/hutianzheng/Desktop/Brain_IB/DATA/BSNIP/ICNs_v2.xlsx"
    id_map = pd.read_excel(map_file_path, sheet_name='Sheet1')
    filtered_df = id_map[id_map['id in model'].isin(target_list)]
    # Get corresponding Category B values
    mapped_categories = filtered_df[['id in model', 'map to BrainNet AAL nodes','Name in AAL']]

    sorted_combinations = mapped_categories.sort_values(by='map to BrainNet AAL nodes', ascending=True)
    index_float_list = sorted_combinations['map to BrainNet AAL nodes'].to_list()
    index_list = [int(x) for x in index_float_list]

    name_list = sorted_combinations['Name in AAL'].to_list()
    return sorted_combinations, index_list, name_list



target_list = [101, 94, 100, 92, 95, 102, 96, 97, 101, 98, 97, 97, 92, 6, 97, 85, 85, 7, 86, 102]
sorted_combinations, index_list, name_list = mapping_list_to_AAl_116(target_list)

print(index_list)