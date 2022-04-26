# -*- coding: utf-8 -*-


import json
import numpy as np
from keras.models import load_model

from att import Attention
from albert_zh.extract_feature import BertVector
load_model = load_model("event_type.h5", custom_objects={"Attention": Attention})

# 预测语句
text="甲乙两位同学用围棋子做游戏如图所示现轮到黑棋下子黑棋下一子后白棋再下一子使黑棋的个棋子组成轴对称图形白棋的个棋子也成轴对称图形则下列下子方法不正确的是说明棋子的位置用数对表示如点在"
#text="二次函数的图象与轴有两个交点其中一个交点为那么另一个交点坐标为"
#text="甲乙两名运动员在某项测试中的次成绩的茎叶图如图所示分别表示甲乙两名运动员这项测试成绩的平均数分别表示甲乙两名运动员这项测试成绩的标准差则有"
text = text.replace("\n", "").replace("\r", "").replace("\t", "")

labels = []

bert_model = BertVector(pooling_strategy="NONE", max_seq_len=200)

# 将句子转换成向量
vec = bert_model.encode([text])["encodes"][0]
x_train = np.array([vec])

# 模型预测
predicted = load_model.predict(x_train)[0]

indices = [i for i in range(len(predicted)) if predicted[i] > 0.2]

with open("event_type.json", "r", encoding="utf-8") as g:
    movie_genres = json.loads(g.read())

print("预测语句: %s" % text)
print("预测事件类型: %s" % "|".join([movie_genres[index] for index in indices]))

