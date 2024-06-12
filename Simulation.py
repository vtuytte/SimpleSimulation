import numpy as np
import cv2
from numba import jit
import numpy as np

def init_simulation(height, width, n_populations):
    """
    Initializes a simulation by creating population and food matrices.

    Parameters:
    height (int): The height of the population matrix.
    width (int): The width of the population matrix.
    n_populations (int): The number of populations in the simulation.

    Returns:
    population_matrix (ndarray): The population matrix with shape (height, width, n_populations).
    food_matrix (ndarray): The food matrix with shape (height, width).
    """
    population_matrix = np.zeros((height, width, n_populations), dtype=np.float32)
    food_matrix = np.zeros((height, width), dtype=np.float32)
    return population_matrix, food_matrix

@jit(target_backend='cuda', nopython=True)
def weighted_choice(choices, probabilities):
    """
    Randomly selects an item from the given choices based on their corresponding probabilities.

    Args:
        choices (list): A list of items to choose from.
        probabilities (list): A list of probabilities corresponding to each item in the choices list.

    Returns:
        The selected item based on the weighted random choice.

    """
    cumulative_probabilities = np.cumsum(probabilities)
    random_choice = np.random.rand()
    for i, cumulative_probability in enumerate(cumulative_probabilities):
        if random_choice < cumulative_probability:
            return choices[i]
    return choices[-1]

@jit(target_backend='cuda', nopython=True)
def update_simulation(height, width, n_populations, population_matrix, food_matrix, exploration_rate, food_eat_rate):
    """
    Update the simulation by performing one iteration of the population dynamics.

    Args:
        height (int): The height of the simulation grid.
        width (int): The width of the simulation grid.
        n_populations (int): The number of populations in the simulation.
        population_matrix (ndarray): The matrix representing the population distribution on the grid.
        food_matrix (ndarray): The matrix representing the food distribution on the grid.
        exploration_rate (float): The exploration rate for choosing nearby cells.
        food_eat_rate (float): The rate at which food is consumed by the population.

    Returns:
        tuple: A tuple containing the updated population matrix and food matrix.
    """
    winning_matrix = np.zeros((height, width), dtype=np.int32)
    # Check which population wins and update the population matrix depending on who won and how much food is available
    for i in range(height):
        for j in range(width):
            if np.sum(population_matrix[i, j]) > 0:
                winning_matrix[i,j] = weighted_choice(np.arange(n_populations), population_matrix[i, j]/np.sum(population_matrix[i, j]))
                new_population = np.sum(population_matrix[i, j])
                eaten_food = min(food_matrix[i,j], new_population) * food_eat_rate
                population_matrix[i, j] = np.zeros(n_populations)
                population_matrix[i, j, winning_matrix[i,j]] = new_population + eaten_food
                food_matrix[i,j] -= eaten_food
    
    # Update the population matrix by moving population to nearby cells
    for i in range(height):
        for j in range(width):
            if population_matrix[i,j, winning_matrix[i,j]] > 0.1:
                nearby_cells = list()
                if i > 0:
                    nearby_cells.append((i-1, j))
                if i < height - 1:
                    nearby_cells.append((i+1, j))
                if j > 0:
                    nearby_cells.append((i, j-1))
                if j < width - 1:
                    nearby_cells.append((i, j+1))
                nearby_cells = np.asarray(nearby_cells)
                nearby_incentive = np.array([np.sum(population_matrix[cell[0], cell[1]]) + 2 * food_matrix[cell[0],cell[1]]  for cell in nearby_cells]) + np.random.rand(len(nearby_cells)) * exploration_rate
                if np.sum(nearby_incentive) == 0:
                    chosen_cell = nearby_cells[np.random.choice(len(nearby_cells))]
                else:
                    cell_index = weighted_choice(np.arange(len(nearby_cells)), nearby_incentive/np.sum(nearby_incentive))
                    chosen_cell = nearby_cells[cell_index]
                population_matrix[chosen_cell[0], chosen_cell[1], winning_matrix[i,j]] += 0.5 * population_matrix[i, j, winning_matrix[i,j]]
                population_matrix[i, j, winning_matrix[i,j]] -= 0.5 * population_matrix[i, j, winning_matrix[i,j]]

    return population_matrix, food_matrix

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function that handles events when the mouse is clicked or released.

    Parameters:
        event: The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN)
        x (int): The x-coordinate of the mouse click
        y (int): The y-coordinate of the mouse click
        flags: Additional flags passed by OpenCV
        param: Additional parameters passed by OpenCV
    """

    global population_matrix, food_matrix, left_button_down, right_button_down, radius, selected

    if event == cv2.EVENT_LBUTTONDOWN or left_button_down:
        # add food/population in radius to mouse click
        if selected == 0:
            food_matrix[y-radius:y+radius, x-radius:x+radius] += 1
        else:
            population_matrix[y-radius:y+radius, x-radius:x+radius, selected-1] += 1
        left_button_down = True
    
    if event == cv2.EVENT_LBUTTONUP:
        left_button_down = False
    
    if event == cv2.EVENT_RBUTTONDOWN or right_button_down:
        # remove food/population in radius to mouse click
        if selected == 0:
            food_matrix[y-radius:y+radius, x-radius:x+radius] = 0
        else:
            population_matrix[y-radius:y+radius, x-radius:x+radius, selected-1] = 0
        right_button_down = True
    
    if event == cv2.EVENT_RBUTTONUP:
        right_button_down = False

def render_simulation(height, width, population_matrix, n_populations, population_colors, food_matrix, food_color=(0, 255, 0)):
    """
    Renders the simulation by displaying the population and food matrices.

    Args:
        height (int): The height of the display matrix.
        width (int): The width of the display matrix.
        population_matrix (ndarray): The population matrix containing population data.
        n_populations (int): The number of populations.
        population_colors (list): A list of RGB color tuples for each population.
        food_matrix (ndarray): The food matrix containing food data.
        food_color (tuple, optional): The RGB color tuple for food. Defaults to (0, 255, 0).
    """
    display_matrix = np.zeros((height, width, 3), dtype=np.uint8)
    ppm = np.clip(population_matrix, 0, 1)
    for k in range(n_populations):
        display_matrix += np.uint8(ppm[:, :, k][:, :, np.newaxis] * population_colors[k])
    display_matrix += np.uint8(food_matrix[:, :, np.newaxis] * food_color)
    cv2.imshow('Population', display_matrix)

def run_simulation(height, width, n_populations, population_colors, RESCALEFACTOR=1):
    """
    Run the simulation.

    Args:
        height (int): The height of the simulation window.
        width (int): The width of the simulation window.
        n_populations (int): The number of populations in the simulation.
        population_colors (list): A list of colors for each population.
        RESCALEFACTOR (int, optional): The factor to scale the simulation window. Defaults to 1.
    """
    WINDOWNAME = 'Population'
    trackbar_offset = 125
    cv2.namedWindow(WINDOWNAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOWNAME, height*RESCALEFACTOR + trackbar_offset, width*RESCALEFACTOR)
    cv2.setMouseCallback(WINDOWNAME, mouse_callback)
    cv2.createTrackbar('Radius', WINDOWNAME, 0, 50, lambda x: globals().update({'radius': x}))
    cv2.createTrackbar('Selected', WINDOWNAME, 0, n_populations, lambda x: globals().update({'selected': x}))
    cv2.createTrackbar('Pause', WINDOWNAME, 0, 1, lambda x: globals().update({'paused': x}))
    cv2.createTrackbar('Exploration Rate', WINDOWNAME, 0, 1000, lambda x: globals().update({'exploration_rate': x/100}))
    cv2.createTrackbar('Food Eat Rate', WINDOWNAME, 0, 100, lambda x: globals().update({'food_eat_rate': x/100}))
    global population_matrix, food_matrix, left_button_down, right_button_down, paused, selected, radius, exploration_rate, food_eat_rate
    left_button_down, right_button_down, paused, selected, radius, exploration_rate, food_eat_rate = False, False, False, 0, 0, 0.5, 0.1 
    population_matrix, food_matrix = init_simulation(height, width, n_populations)
    while True:
        if not paused:
            population_matrix, food_matrix = update_simulation(height, width, n_populations, population_matrix, food_matrix, exploration_rate, food_eat_rate)
        render_simulation(height, width, population_matrix, n_populations, population_colors, food_matrix)
        if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty(WINDOWNAME, cv2.WND_PROP_VISIBLE) <1:
            break

if __name__ == '__main__':
    run_simulation(250, 250, 2, [(0, 0, 255), (255, 0, 0)], 4)