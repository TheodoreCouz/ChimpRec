import sys
sys.path.append("PATH TO /Code")

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

# df_builtin = pd.read_csv("./PR_v8s_built_in.csv")
# df_custom = pd.read_csv("./PR_v8s_custom.csv")

# P_builtin, R_builtin = df_builtin["Precision"].tolist(), df_builtin["Recall"].tolist()
# P_custom, R_custom = df_custom["Precision"].tolist(), df_custom["Recall"].tolist()

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

def plot_PR(data, out_path):
    # Sort data by t values
    sorted_data = sorted(data.items())  
    t_values, data_extracted = zip(*sorted_data)

    # Extract precision and recall
    precision_values, recall_values, n_pred_list = zip(*data_extracted)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, precision_values, label="Precision", color="mediumvioletred", marker='o', linestyle='-')
    plt.plot(t_values, recall_values, label="Recall", color="rebeccapurple", marker='s', linestyle='-')

    # Labels and title
    plt.xlabel("Confidence threshold")
    plt.ylabel("Value")
    plt.title(f"Precision and Recall as a function of the merging threshold (w_iou_t = {0.6}, conf_t = {0.35}) (body detection)")
    plt.legend()
    plt.grid(True)

    # Set x-axis ticks to show all t values
    plt.xticks(t_values, labels=[str(t) for t in t_values])  # Ensuring each t is explicitly labeled

    # Display the plot
    plt.savefig(f"{out_path}/PR.svg", format="svg")
    plt.show()

def plot_proportion(GT, data, out_path = ""):
    # Sort data by t values
    sorted_data = sorted(data.items())  
    t_values, data_extracted = zip(*sorted_data)

    # Extract precision and recall
    precision_values, recall_values, n_pred_list = zip(*data_extracted)

    counter_GT = 0
    for i in GT.values(): counter_GT += len(i)

    ratios = [i/counter_GT for i in n_pred_list]

    # Generate x values: merging thresholds (evenly spaced from 0.05 to 0.95)
    x_val = np.linspace(0.05, 0.95, len(ratios))

    # Plot the curve
    plt.figure(figsize=(10, 5))
    plt.plot(x_val, ratios, marker='o', linestyle='-', color='steelblue')

    # Add horizontal reference line at y = 1
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2)

    # Labels and title
    plt.xlabel("Merging threshold")
    plt.ylabel("#pred / #ground truth")
    plt.title("# predicted bboxes over # ground truth bboxes as a function of the merging threshold (body detection)")

    # Grid and display
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{out_path}/proportions.svg", format="svg")
    plt.show()

"""
Precision and recall as a function of the confidence threshold
"""
def PR_conft_plot(model_path, images, labels, t_iou, out_path = "NONE"):
    iterators = [i/20 for i in range(1, 20, 1)]

    data = dict()

    GT = extract_ground_truth(labels, images)

    for i in iterators:
        pred = predict(model_path, images, i)
        results = extract_metrics(GT, pred, t=t_iou)
        tp, fp, fn = results.values()
        if (tp+fp) == 0: precision = 0
        else: precision=tp/(tp+fp)

        if (tp+fn) == 0: recall = 0
        else: recall=tp/(tp+fn)

        n_pred = 0
        for values in pred.values(): n_pred += len(values)
        data[i] = (precision, recall, n_pred)
    
    plot_PR(data, out_path)
    plot_proportion(GT, data, out_path)

    if out_path != "NONE":
        df = pd.DataFrame.from_dict(data, orient="index", columns=["Precision", "Recall", "N_PRED"])
        df.index.name = "Threshold"
        df.to_csv(f"{out_path}/results_PR.csv")
    
def plot_merging_threshold(model_path, images, labels, t_iou,  out_path = "NONE"):
        iterators = [i/20 for i in range(1, 20, 1)]

        data = dict()
        n_pred_list = []

        GT = extract_ground_truth(labels, images)
        predictions = predict(model_path, images, 0.35)

        for i in iterators:
            merged = merge_boxes(predictions, i)
            results = extract_metrics(GT, merged, t=t_iou)
            tp, fp, fn = results.values()
            if (tp+fp) == 0: precision = 0
            else: precision=tp/(tp+fp)

            if (tp+fn) == 0: recall = 0
            else: recall=tp/(tp+fn)

            n_pred = 0
            for values in merged.values(): n_pred += len(values)
            data[i] = (precision, recall, n_pred)

        plot_PR(data, out_path)
        plot_proportion(GT, data, out_path)

        if out_path != "NONE":
            df = pd.DataFrame.from_dict(data, orient="index", columns=["Precision", "Recall", "N_PRED"])
            df.index.name = "MERGING_R"
            df.to_csv(f"{out_path}/results_merging_bboxes.csv")

def plot_iou_comparison(model_path, yaml_file, images, labels, t_iou,  out_path = "NONE"):
    iterators = [i/20 for i in range(1, 20, 1)]
    builtin_recall = []
    builtin_precision = []

    model = YOLO(model_path)

    for i in iterators:
        # Evaluate the model
        results = model.val(data=yaml_file, conf=i)
        precision, recall = results.box.mp, results.box.mr
        builtin_precision.append(float(precision))
        builtin_recall.append(float(recall))

    #-----------------------------------------
    GT = extract_ground_truth(labels, images)

    iou_recall = []
    iou_precision = []

    w_iou_recall = []
    w_iou_precision = []
    for i in iterators:
        pred = predict(model_path, images, i)
        results_iou = extract_metrics(GT, pred, t=t_iou, score_fct=iou)
        results_w_iou = extract_metrics(GT, pred, t=t_iou, score_fct=weighted_iou)

        tp_iou, fp_iou, fn_iou = results_iou.values()
        tp_w_iou, fp_w_iou, fn_w_iou = results_w_iou.values()

        if (tp_iou+fp_iou) == 0: precision_iou = 0
        else: precision_iou=tp_iou/(tp_iou+fp_iou)

        if (tp_iou+fn_iou) == 0: recall_iou = 0
        else: recall_iou=tp_iou/(tp_iou+fn_iou)

        iou_recall.append(recall_iou)
        iou_precision.append(precision_iou)

        if (tp_w_iou+fp_w_iou) == 0: precision_w_iou = 0
        else: precision_w_iou=tp_w_iou/(tp_w_iou+fp_w_iou)

        if (tp_w_iou+fn_w_iou) == 0: recall_w_iou = 0
        else: recall_w_iou=tp_w_iou/(tp_w_iou+fn_w_iou)

        w_iou_recall.append(recall_w_iou)
        w_iou_precision.append(precision_w_iou)

    df = pd.DataFrame({
        'CONF_T': iterators,
        'RECALL_BI': builtin_recall,
        'PRECISION_BI': builtin_precision,
        'RECALL_IOU': iou_recall,
        'PRECISION_IOU': iou_precision,
        'RECALL_W_IOU': w_iou_recall,
        'PRECISION_W_IOU': w_iou_precision
    })

    df.to_csv(f"{out_path}/data.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(iterators, iou_precision, marker='x', label='Precision (IOU)', color="steelblue", linestyle='-')
    plt.plot(iterators, w_iou_precision, marker='s', label='Precision (weighted IOU)', color="navy", linestyle='-')
    plt.plot(iterators, builtin_precision, marker='o', label='Precision (built in - ultralytics)', color="mediumvioletred", linestyle='-')

    plt.xlabel('Confidence threshold')
    plt.ylabel('Precision')
    plt.title('Precision vs confidence threshold - three ways to compute the iou score (iou_t = 0.6) (body detection)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_path}/precision.svg", format = "svg")
    plt.show()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterators, iou_recall, marker='x', label='Recall (IOU)', color="steelblue", linestyle='-')
    plt.plot(iterators, w_iou_recall, marker='s', label='Recall (weighted IOU)', color="navy", linestyle='-')
    plt.plot(iterators, builtin_recall, marker='o', label='Recall (built in - ultralytics)', color="mediumvioletred", linestyle='-')

    plt.xlabel('Confidence threshold')
    plt.ylabel('Recall')
    plt.title('Recall vs confidence threshold - three ways to compute the iou score (iou_t = 0.6) (body detection)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_path}/recall.svg", format = "svg")
    plt.show()