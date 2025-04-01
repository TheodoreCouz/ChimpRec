import sys
sys.path.append("C:\\Users\\Theo\\Documents\\Unif\\ChimpRec\\Code")

from chimplib.metric import *

def plot_weight(f):
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
    plt.xlabel("Bounding Box Size (Normalised)")
    plt.ylabel("IoU Weight")
    plt.title("default IoU weight as a function of the normalised predicted bbox size (a=1.6; b=18)")

    # Add Axes
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # X-axis
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # Y-axis

    # Legend and Grid
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/iou_weight.svg", format="svg")
    plt.show()

df_builtin = pd.read_csv("./PR_v8s_built_in.csv")
df_custom = pd.read_csv("./PR_v8s_custom.csv")

P_builtin, R_builtin = df_builtin["Precision"].tolist(), df_builtin["Recall"].tolist()
P_custom, R_custom = df_custom["Precision"].tolist(), df_custom["Recall"].tolist()

def plot(builtin, custom, colors, metric):

    c1, c2 = colors

    # Create the plot
    t_values = [i/20 for i in range(1, 20, 1)]
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, builtin, label=f"Built in {metric}", color=c1, marker='o', linestyle='-')
    plt.plot(t_values, custom, label=f"Custom Precision {metric}", color=c2, marker='s', linestyle='-')

    # Labels and title
    plt.xlabel("Confidence threshold")
    plt.ylabel(metric)
    plt.title(f"Built in VS Custom {metric} as a function of the confidence threshold (iou_t = {0.6}) (body detection)")
    plt.legend()
    plt.grid(True)

    # Set x-axis ticks to show all t values
    plt.xticks(t_values, labels=[str(t) for t in t_values])  # Ensuring each t is explicitly labeled

    # Display the plot
    plt.savefig(f"plots/{metric}_comparison.svg", format="svg")
    plt.show()

# plot(P_builtin, P_custom, ("mediumvioletred", "rebeccapurple"), "Precision")
# plot(R_builtin, R_custom, ("steelblue", "navy"), "Recall")

"""
Precision and recall as a function of the confidence threshold
"""
def PR_conft_plot():
    iterators = [i/20 for i in range(1, 20, 1)]

    data = dict()
    n_pred_list = []

    for i in iterators:
        pred = predict(model_path, test_set, i)
        results = extract_metrics(GT, pred, t=t)
        tp, fp, fn = results.values()
        if (tp+fp) == 0: precision = 0
        else: precision=tp/(tp+fp)

        if (tp+fn) == 0: recall = 0
        else: recall=tp/(tp+fn)
        data[i] = (precision, recall)
        n_pred = 0
        for values in pred.values():
            n_pred += len(values)
        n_pred_list.append(n_pred)