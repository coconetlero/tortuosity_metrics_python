
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import re        



def load_float_curve_from_txt_file(file_path):
    """
    Load a curve from a txt file
    """
    curve = np.loadtxt(file_path, dtype=np.float32, delimiter=',')
    return curve


def load_float_curves_from_txt_files(folder_path):
    """
    Load all curves from a folder containing txt files
    """
    curves = []
    filenames = []
    filtered_filenames = [item for item in os.listdir(folder_path) if not item.startswith('._')]
    filtered_filenames = sorted(filtered_filenames)
    for filename in filtered_filenames:
        file_path = os.path.join(folder_path, filename)
        curve = load_float_curve_from_txt_file(file_path)
        curves.append(curve)
        filenames.append(filename)
    
    return curves, filenames



def load_pixelated_curve_from_txt_file(file_path, delimiter=' '):
    """
    Load a curve from a txt file
    """
    points = np.loadtxt(file_path, dtype=int, delimiter=delimiter)
    unique_rows, idx = np.unique(points, axis=0, return_index=True)
    pixelated_curve = unique_rows[np.argsort(idx)]
    return pixelated_curve


def load_pixelated_curves_from_txt_files(folder_path):
    """
    Load all curves from a folder containing txt files
    """
    curves = []
    filenames = []
    filtered_filenames = [item for item in os.listdir(folder_path) if not item.startswith('._')]
    filtered_filenames = sorted(filtered_filenames)
    for filename in filtered_filenames:
        file_path = os.path.join(folder_path, filename)
        curve = load_pixelated_curve_from_txt_file(file_path)
        curves.append(curve)
        filenames.append(filename)
    
    return curves, filenames



def find_and_load_pixelated_curve_from_image(binary_image):
    """
    Traces an open curve of 1-pixel thickness in a binary image [0, 255].

    Parameters:
        binary_image: 2D np.ndarray with values 0 and 255

    Returns:
        list of (x, y) tuples with the ordered coordinates of the curve,
        from one endpoint to the other. Returns an empty list if no
        curve is found.
    """
    img = (binary_image == 255)
    rows, cols = img.shape

    # Offsets for the 8 neighbors
    neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]

    def get_neighbors(r, c):
        """Returns list of (r, c) neighbor coordinates that belong to the curve."""
        neighbors = []
        for dr, dc in neighbor_offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and img[nr, nc]:
                neighbors.append((nr, nc))
        return neighbors

    # 1. Find all curve pixels
    curve_coords = np.argwhere(img)
    if len(curve_coords) == 0:
        return []

    # 2. Find an endpoint: a pixel with exactly 1 neighbor
    endpoint = None
    for r, c in curve_coords:
        r, c = int(r), int(c)
        if len(get_neighbors(r, c)) == 1:
            endpoint = (r, c)
            break

    # If no clear endpoint (closed curve or noise), take any point
    if endpoint is None:
        endpoint = tuple(int(v) for v in curve_coords[0])

    # 3. Traverse the curve starting from the endpoint
    visited = set()
    path = []

    current = endpoint
    previous = None

    while current is not None:
        visited.add(current)
        path.append(current)

        neighbors = get_neighbors(*current)
        # Discard already visited neighbors
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if len(unvisited_neighbors) == 0:
            # Reached the other endpoint (or got stuck)
            break
        elif len(unvisited_neighbors) == 1:
            next_pixel = unvisited_neighbors[0]
        else:
            # Branch point: choose the neighbor closest to the current
            # direction of travel (avoids weird jumps)
            if previous is not None:
                dir_r = current[0] - previous[0]
                dir_c = current[1] - previous[1]
                def score(n):
                    nr, nc = n[0] - current[0], n[1] - current[1]
                    return -(nr * dir_r + nc * dir_c)  # prioritize same direction
                unvisited_neighbors.sort(key=score)
            next_pixel = unvisited_neighbors[0]

        previous = current
        current = next_pixel

    # 4. Convert from (r, c) -> (x, y) where x=col, y=row
    curve_xy = [(c, r) for (r, c) in path]

    return curve_xy



def plot_two_curves(curve1, curve2, plot_title="Curves Comparison", label1='Curve 1', label2='Curve 2'):
    """
    Plot the results of the polynomial fitting
    """
    plt.figure(figsize=(18, 12))
    plt.plot(curve1[:, 0], curve1[:, 1], 'o-', color="darkturquoise", alpha=0.6, markersize=2, linewidth=1, label=label1)
    plt.plot(curve2[:, 0], curve2[:, 1], 'o-', color="crimson", alpha=0.6, markersize=2, linewidth=1, label=label2)


    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()



def display_curve_on_image(image_path, curve):

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError("Could not load image")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    x = curve[:, 0]
    y = curve[:, 1]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.plot(x, y, linewidth=2, color='red', alpha=0.5)   # curve overlay
    plt.scatter(x, y, s=5, color='yellow', alpha=0.5)     # optional: show sample points
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def display_curve_on_image_2(image_path, curve1, curve2):
    
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError("Could not load image")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    x = curve1[:, 0]
    y = curve1[:, 1]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.plot(curve2[:, 0], curve2[:, 1], 'o-', color="lawngreen",
     alpha=0.5, markersize=2, linewidth=1, label="pixel")
    plt.plot(x, y, linewidth=2, color='red', alpha=0.5)   # curve overlay
    plt.scatter(x, y, s=5, color='yellow', alpha=0.5)     # optional: show sample points
    

    plt.axis('off')
    plt.tight_layout()
    plt.show()