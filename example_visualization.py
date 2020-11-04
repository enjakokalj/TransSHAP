import numpy as np
from explainers import visualize_explanations

words_impact = [
    [["Being", "offensive", "isn't", "illegal", "you", "idiot"],
    [-0.1, 0.6, -0.05, 0.35, 0.01, 0.45],
    [0.05, 0.35, 0.6]],
]

### Our visualization
for text, contribution_values, f in words_impact:
    class_to_explain = ["Positive", "Neutral", "Negative"][int(np.argmax(f))]
    prediction_probability = f[np.argmax(f)] * 100
    visualize_explanations.bar_chart_explanation(text, contribution_values, class_to_explain, prediction_probability)
    visualize_explanations.text_box_explanation(text, contribution_values)