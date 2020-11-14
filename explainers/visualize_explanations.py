import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc, font_manager
import seaborn as sns
import itertools
import glob


def bar_chart_explanation(tokenized_text, values, class_to_explain, pred):
    values = np.array(values)
    plt.figure(figsize=(12, 6))

    colors = ["green" if x > 0 else "red" for x in values]
    plt.bar([*range(len(values))], values, color=colors)
    plt.xticks(np.arange(len(tokenized_text)), tokenized_text, fontsize=15)
    plt.yticks(fontsize=15)
    plt.axhline(y=0, color='black', linestyle='dashed')
    title = f"Predicted class: {class_to_explain} ({pred:.2f} %)"
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.show()


def text_box_explanation(raw, values):
    values = np.array(values)
    fixed_y = 0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    Yy = 5
    plt.xlim((0, Yy))
    plt.ylim((0.4, 0.6))
    threshold = sum(abs(values)) * 0.01
    h = [x if abs(x) > threshold else 0 for x in values]
    h /= np.sum(np.abs(h))
    show_box = [["green", "mediumaquamarine"][int(x * 10 < 1)] if abs(x) > threshold and x > 0 else
                ["red", "tomato"][int(x * 10 > -1)] if abs(x) > threshold else "white" for x in h]
    text_color = ["white" if abs(x) > threshold and x > 0 else "black" for x in values]
    coord = []
    for i, word in enumerate(raw):
        x = 0 if not coord else coord[-1][1]
        t = plt.text(x=x, y=fixed_y, s=word, ha="center", va="center", size=20, rotation=0., color=text_color[i],
                     bbox=dict(boxstyle="square", ec="white", fc=show_box[i], ))
        tt = t.get_window_extent(renderer=fig.canvas.get_renderer())
        transf = ax.transData.inverted()
        d = tt.transformed(transf)
        f = (d.x0, d.x1, d.y0, d.y1)
        diff_x = d.x1 - d.x0 + 0.01

        if not coord:
            t.set_position((diff_x / 2, fixed_y))
        elif x + diff_x < Yy:
            t.set_position((x + diff_x / 2, fixed_y))
        else:
            fixed_y -= 0.1
            t.set_position((diff_x / 2, fixed_y))
        # print(t.get_position())
        # print(f)
        # print(diff_x)
        coord.append((t.get_position(), x + diff_x))
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def determine_graph_width(max_word, max_length):
    return max_word * max_length * 0.15

def joint_visualization(tokenized_text, values, class_to_explain, pred, i):
    ## first plot.

    # if len(tokenized_text) > 10:
    #     font_size = 10
    # else:
    #     font_size = 12

    font_size = 14#20 - (len(tokenized_text) * 0.8)
    font_properties = {'family': 'serif', 'serif': ['Computer Modern Roman'],
                   'weight': 'normal', 'size': font_size}
    
    font_manager.FontProperties(family='Computer Modern Roman', style='normal',
                            size=font_size, weight='normal', stretch='normal')
    # rc('text', usetex=True)
    rc('font', **font_properties)
    
    sns.set_style('whitegrid')
    plt.rcParams["figure.figsize"] = (determine_graph_width(max_word=len(max(tokenized_text, key=len)), max_length=len(tokenized_text)), 6)
    fig, ax = plt.subplots(1, 1)
    values = np.array(values)
    # colors = sns.color_palette("coolwarm", len(values))
    # out_indices = np.argsort(values)[::1]
    # print(out_indices)
    # print(values)
    # colors = [colors[x] for x in out_indices]

    perc_pos, perc_neg = 0, 0
    for xx in [[x for x in values if x > 0], [x for x in values if x <= 0]]:
        try:
            if all([x > 0 for x in xx]):
                perc_pos = np.percentile(xx, 50)
            elif all([x <= 0 for x in xx]):
                perc_neg = np.percentile(xx, 50)
        except IndexError:
            pass

    colors = [["mediumaquamarine", "green"][int(x > perc_pos)] if x > 0 else
                       ["red", "tomato"][int(x > perc_neg)] for x in values]
    # text_color = ["white" if abs(x) > threshold and x > 0 else "black" for x in values]
    # colors = ["green" if x > 0 else "red" for x in values]

    plt.bar([*range(len(values))], values, color=colors, edgecolor="black", alpha=0.6)
    ax.set_xticks([*range(len(values))])
    # sns.barplot([*range(len(values))], values, color = colors, edgecolor = "black", alpha = 0.6)
    # print(ax.get_xticklabels(which="both"))
    ax.set_xticklabels(tokenized_text)
    colors_ticks = colors
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors_ticks):
        bbox = dict(boxstyle="round", ec="black", fc=tickcolor, alpha=0.2)
        plt.setp(ticklabel, bbox=bbox)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha="right")
    plt.axhline(y=0, color='black', linestyle='dashed')
    pred *= 100
    title = f"Predicted class: {class_to_explain} ({pred:.2f} %)"
    fig.suptitle(title)
    plt.ylabel("Impact on model output")
    pname = "".join(tokenized_text[0:3])
    # plt.show()
    plt.savefig(f"figures/our_vis_{i}", dpi=300)
    print(f"DONE figures/our_vis_{i}")
