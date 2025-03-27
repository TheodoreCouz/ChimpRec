import matplotlib.pyplot as plt
import numpy as np
import math

def weight(bbox_size):

    def h(x):
        return 0.4 * math.tanh(15*x - 0.4) + 0.5
    
    return 1/h(bbox_size + 0.035) - 0.112

def plot_function(f):
    # Generate x values in the range [0, 1]
    x_values = np.linspace(0, 1, 1000)
    color_GT = (34/255, 87/255, 236/255)
    fill_color = (34/255, 87/255, 236/255, 0.2)
    
    # Compute y values using the weight function
    y_values = [f(x) for x in x_values]
    
    y_intersection = f(0)

    # Plot the function
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label="weight(bbox_size)", color=color_GT)

    plt.fill_between(x_values, y_values, color=fill_color, alpha=0.2)

    # Add a horizontal line at y_intersection
    plt.axhline(y_intersection, color="red", linestyle="--", label=f"y(0) = {y_intersection:.3f}")

    # Labels and Title
    plt.xlabel("Bounding Box Size (Normalided)")
    plt.ylabel("IoU Weight")
    plt.title("IoU weight as a function of the normalised predicted bbox size")

    # Add Axes
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # X-axis
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Y-axis

    # Legend and Grid
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/iou_weight.svg", format="svg")
    plt.show()

# Call the function to plot
plot_function(weight)
