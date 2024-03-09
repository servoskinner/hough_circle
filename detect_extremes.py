import numpy as np

def area_traverse(array, x0, y0, markup=None, leap_size=1):
    '''
    Traverse enclosed grid area, return its width, height, and center coordinates
    '''
    if markup is None:
        markup = np.zeros_like(array)

    x_max, x_min, y_max, y_min = x0-1, x0, y0-1, y0
    center = np.array([0, 0])
    count = 0

    # leap_size determines the radius of a cell's neighbourhood in which other cells are considered connected to it;
    step_options = np.array([[hor, ver] for hor in range(-leap_size, leap_size+1) for ver in range(-leap_size, leap_size+1)])
    # traverse the grid
    if markup[x0, y0] != 0:
        return -1, -1
    neighbours = [np.array([x0, y0])]
    markup[x0, y0] = 1

    area_class = array[x0, y0]

    while len(neighbours) > 0:
        position = neighbours.pop(0)
        # process current cell
        count += 1
        center += position
        x_max = max(position[0],  x_max)
        x_min = min(position[0],  x_min)
        y_max = max(position[1],  y_max)
        y_min = min(position[1],  y_min)
        # shortlist next step options
        options = [position + delta for delta in step_options]
        for option in options:
            # check if option is legal
            if np.all(option >= 0) and np.all(option < array.shape) \
                and markup[tuple(option)] == 0 and array[tuple(option)] == area_class:
                # add to queues
                neighbours.append(option)
                markup[tuple(option)] = 1

    height = x_max-x_min+1
    width  = y_max-y_min+1
    return height, width, center/count if count != 0 else center

def to_bit_array(array, threshold=0.5, normalize=True):
    array_norm = np.copy(array)
    if normalize:
        arr_max, arr_min = np.max(array), np.min(array)
        if arr_max == arr_min: # division by zero follows
            return np.zeros_like(array)
        array_norm = (array_norm - arr_min)/(arr_max - arr_min)

    array_norm = np.where(array_norm >= threshold, 1, 0)
    return array_norm

def detect_extreme_points(array, threshold=0.8, normalize=False, max_cluster_size=1, leap_size=1):
    '''
    Detect peaks on 2d array.
    '''
    traversal_progress = np.zeros_like(array)
    
    array_norm = to_bit_array(array, threshold=threshold, normalize=normalize)
    # detect clusters, reject them if they are too large and therefore ambiguous
    extrema = []
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            if array_norm[x, y] != 0 and traversal_progress[x, y] == 0:
                # traverse the cluster to determine its size and center point
                cluster_height, cluster_width, center = area_traverse(array_norm, x, y, markup=traversal_progress,
                                                                      leap_size=leap_size)
                if max(cluster_height, cluster_width) <= max_cluster_size:
                    extrema.append(center + np.array([0.5, 0.5])) # grid origin adjustment - (0, 0) would be top left pixel's
                                                                  #  top left edge, not its center
    return extrema

