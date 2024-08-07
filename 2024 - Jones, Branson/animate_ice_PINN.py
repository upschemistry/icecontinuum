import f90nml
import keras
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from pint import UnitRegistry; AssignQuantity = UnitRegistry().Quantity
import reference_solution as refsol


# Read in GI parameters
inputfile = "GI parameters - Reference limit cycle (for testing).nml"
GI=f90nml.read(inputfile)['GI']
nx_crystal = GI['nx_crystal']
L = GI['L']

# Define x values for plotting
X_QLC = np.linspace(-L,L,nx_crystal)

# Define time constants
NUM_T_STEPS = 51
RUNTIME = 50

TITLE_DICT = {0: "Ntot", 1: "Nqll", 2: "N-ice"}

def animate_refsol(index):
    """
    Animates reference solution.

    Args:
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
    """
    REFERENCE_SOLUTION = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS)

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Reference "+TITLE_DICT[index]) # set axes

    line = ax.plot(X_QLC, X_QLC, 'b', linewidth=1.0, label="Reference Value", zorder=0)[0]
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        ref_y = REFERENCE_SOLUTION[index][frame]

        # Update y-axis limits to fit the data
        ax.set_ylim(np.min(ref_y), np.max(ref_y))

        # set t values to graph for this frame    
        line.set_ydata(ref_y)

        return (line)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=50)
    # show me the money
    plt.show()

def animate_PINN(model_name, index):
    """
    Animates PINN model output.

    Args:
        model_name: String local file path of model
        index: 0 for Ntot, 1 for Nqll, 2 for N-ice
    """
    # load network
    model = keras.models.load_model(model_name)

    # Define x and t points
    # X_QLC defined above
    t_points = np.linspace(0, RUNTIME, NUM_T_STEPS)
    x, t = np.meshgrid(X_QLC, t_points)
    x_flat = x.flatten()
    t_flat = t.flatten()

    # Create the input array for the network
    input_points = np.vstack((x_flat, t_flat)).T

    # Get predictions from the network
    pred = model.predict(input_points)
    Ntot_pred = pred[:, 0].reshape((NUM_T_STEPS, 320))
    Nqll_pred = pred[:, 1].reshape((NUM_T_STEPS, 320))
    Nice_pred = Ntot_pred - Nqll_pred

    # Stack predictions to match expected output shape
    network_solution = np.stack([Ntot_pred, Nqll_pred, Nice_pred], axis=0)

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Predicted "+TITLE_DICT[index]) # set axes

    line = ax.plot(X_QLC, X_QLC, 'r', linewidth=1.0, label="Network Prediction", zorder=0)[0]
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        net_y = network_solution[index][frame]

        # Update y-axis limits to fit the data
        ax.set_ylim(np.min(net_y), np.max(net_y))

        # set t values to graph for this frame    
        line.set_ydata(net_y)

        return (line)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=50)
    # show me the money
    plt.show()


def animate_both(model_name, index):
    # load network
    model = keras.models.load_model(model_name)
    
    # Define x and t points
    # X_QLC defined above
    t_points = np.linspace(0, RUNTIME, NUM_T_STEPS)
    x, t = np.meshgrid(X_QLC, t_points)
    x_flat = x.flatten()
    t_flat = t.flatten()

    # Create the input array for the network
    input_points = np.vstack((x_flat, t_flat)).T

    # Get predictions from the network
    pred = model.predict(input_points)
    Ntot_pred = pred[:, 0].reshape((NUM_T_STEPS, 320))
    Nqll_pred = pred[:, 1].reshape((NUM_T_STEPS, 320))
    Nice_pred = Ntot_pred - Nqll_pred

    # Stack predictions to match expected output shape
    network_solution = np.stack([Ntot_pred, Nqll_pred, Nice_pred], axis=0)

    # Generate expected output
    REFERENCE_SOLUTION = refsol.generate_reference_solution(runtime=RUNTIME, num_steps=NUM_T_STEPS)

    # set up graph
    fig, ax = plt.subplots()
    ax.set(xlim=[-L, L], xlabel="x (micrometers)", ylabel="Number of Ice Layers", title="Comparing PINN to Reference: "+TITLE_DICT[index]) # set axes

    # Create lines for prediction and reference values
    lines = ax.plot(X_QLC, X_QLC, 'r--', label="Network Prediction", zorder=2)  # Initialize with placeholder data
    lines.append(ax.plot(X_QLC, X_QLC, 'b', linewidth=1.0, label="Reference Value", zorder=0)[0])
    ax.legend()

    # list of frames to iterate through (a list of unique times)
    frames = np.arange(NUM_T_STEPS)

    # update method to be passed into animator
    def update(frame):

        # select y values that correspond to the frame in question
        net_y = network_solution[index][frame]
        ref_y = REFERENCE_SOLUTION[index][frame]

        # Update y-axis limits to fit the data
        ax.set_ylim(np.min([np.min(net_y), np.min(ref_y)]), np.max([np.max(ref_y), np.max(ref_y)]))

        # set t values to graph for this frame    
        lines[0].set_ydata(net_y)
        lines[1].set_ydata(ref_y)

        return (lines)

    # animation!
    animation = ani.FuncAnimation(fig=fig, func=update, frames=frames, interval=50)
    # show me the money
    plt.show()
    