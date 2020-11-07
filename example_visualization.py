from explainers import visualize_explanations
from nltk.tokenize import TweetTokenizer
import pandas as pd

def split_string(string):
    data_raw = TweetTokenizer().tokenize(string)
    data_raw = [x for x in data_raw if x not in ".,:;'"]
    return data_raw

t1 = "Why is it that 'fire sauce' isn't made with any real fire? Seems like false advertising." #neutral
t2 = "Being offensive isn't illegal you idiot." #negative
t3 = "Loving the first days of summer! <3" #positive
t4 = "I hate when people put lol when we are having a serious talk ."   #negative
t5 = "People are complaining about lack of talent. Takes a lot of talent to ignore objectification and degradation #MissAmerica" #neutral
t6 = "Shit has been way too fucked up, for way too long. What the fuck is happening" #negative
t7 = "@Lukc5SOS bc you're super awesome😉" #positive
t8 = "RT @JakeBoys: This show is amazing! @teenhoot family are insane 😍" #positive
t9 = "So kiss me and smile for me 😊💗 http://t.co/VsRs8KUmOP"

texts = [t2, t3, t4, t7, t8, t9]
texts_ = [split_string(x) for x in texts]

# m = len(max(texts_, key=len))
# texts_all = []
# for i, x in enumerate(texts_):
#     contr = [random.random()*[-1, 1][int(random.random() > 0.5)] for _ in range(len(x))]
#     prob = []
#     for _ in range(3):
#         if len(prob) < 2:
#             prob.append(random.uniform(0.0, 1-sum(prob)))
#         else:
#             prob.append(1-sum(prob))
#     print(sum(prob))
#     p = prob.index(max(prob))
#     t = [texts[i], p, prob[p], *contr]
#     diff = m - len(contr)
#     texts_all.append(t + [0.0]*diff)
#
# pd.DataFrame(texts_all, columns=["text", "pred", "prob", *range(13)]).to_csv("data.csv")

data = pd.read_csv("data.csv").iloc[:, 1:]
for i in range(data.shape[0]):
    d = data.iloc[i, :]
    text, class_to_explain, prediction_probability = split_string(d["text"]), \
                                                     ["Positive", "Neutral", "Negative"][d["pred"]], d["prob"]
    contribution_values = list(d[3:3 + len(text)])
    # visualize_explanations.bar_chart_explanation(text, contribution_values, class_to_explain, prediction_probability)
    # visualize_explanations.text_box_explanation(text, contribution_values)
    visualize_explanations.joint_visualization(text, contribution_values, class_to_explain, prediction_probability, i)
