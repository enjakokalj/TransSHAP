import matplotlib.pyplot as plt
import numpy as np


def bar_chart_explanation(tokenized_text, values, class_to_explain, pred):
    values = np.array(values)
    plt.figure(figsize=(12, 6))
    
#     val_max = np.max(values)
#     f=[]
#     for i, x in enumerate(str(val_max)):
#         if x in "0.":
#             f.append(x)
#         else:
#             f.append(str(int(x)+1))
#             break
#     plt.ylim(top=float("".join(f)))
    
#     val_min = np.min(values)
#     if val_min < 0:
#         f=[]
#         for i, x in enumerate(str(val_min)):
#             if x in "0.":
#                 f.append(x)
#             else:
#                 f.append(str(int(x)-1))
#                 break
#         plt.ylim(bottom=float("".join(f)))
#     else:
#         plt.ylim(bottom=0)
    
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
    Yy = 2
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
        t = plt.text(x=x, y=fixed_y, s=word, ha="center", va="center", size=35, rotation=0., color=text_color[i],
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
