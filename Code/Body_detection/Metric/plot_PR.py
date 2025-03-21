import pandas as pd
import matplotlib.pyplot as plt

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

plot(P_builtin, P_custom, ("mediumvioletred", "rebeccapurple"), "Precision")
plot(R_builtin, R_custom, ("steelblue", "navy"), "Recall")


