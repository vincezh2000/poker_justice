#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, jsonify, render_template

from PIL import Image

#################################
# 1. 花色 & 点数映射
#################################
suit_map = {'S': 0, 'C': 1, 'D': 2, 'H': 3}
rank_map = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
    '7': 5, '8': 6, '9': 7, 'T': 8,
    'J': 9, 'Q': 10, 'K': 11, 'A': 12
}

# 反向映射: 索引 -> 字符
idx2suit = {v: k for k, v in suit_map.items()}   # 0->"S",1->"C",2->"D",3->"H"
idx2rank = {v: k for k, v in rank_map.items()}   # 0->"2",1->"3",...,12->"A"

#################################
# 2. 定义模型结构(与训练保持一致)
#################################
class CardNetResnet(nn.Module):
    """
    ResNet18 微调: 去掉原 fc, 换成 suit(4类) 和 rank(13类) 两个输出.
    """
    def __init__(self, freeze_layers=True):
        super(CardNetResnet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.fc_suit = nn.Linear(in_features, 4)
        self.fc_rank = nn.Linear(in_features, 13)
        
        if freeze_layers:
            # 仅保留 layer4 可训练 (推理时其实无关)
            for name, param in self.resnet.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False

    def forward(self, x):
        features = self.resnet(x)   # [B, in_features]
        suit_out = self.fc_suit(features)
        rank_out = self.fc_rank(features)
        return suit_out, rank_out

#################################
# 3. 推理的图像预处理
#################################
transform_infer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    # 如果训练时用了 Normalize(...)，请在此加同样的 Normalize
])

def predict_single_image(model, image, device):
    """
    给定一张 PIL.Image，返回预测的 (suit_char, rank_char)
    """
    image_tensor = transform_infer(image).unsqueeze(0).to(device)
    with torch.no_grad():
        suit_out, rank_out = model(image_tensor)
        suit_idx = suit_out.argmax(dim=1).item()  # 0~3
        rank_idx = rank_out.argmax(dim=1).item()  # 0~12
    suit_char = idx2suit[suit_idx]  # "S"/"C"/"D"/"H"
    rank_char = idx2rank[rank_idx]  # "2"~"A"
    return suit_char, rank_char


#################################
# 4. 批量预测
#################################
def predict_folder(model, folder_path, device):
    """
    对文件夹内所有 png/jpg/jpeg 图片执行推理，返回如 ["3C","TS","AH"]...
    按文件名排序，确保结果顺序稳定；也可以自行定义排序方式。
    """
    results = []
    all_files = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png','.jpg','.jpeg'))]
    all_files.sort()
    for fname in all_files:
        full_path = os.path.join(folder_path, fname)
        img = Image.open(full_path).convert('RGB')
        s, r = predict_single_image(model, img, device)
        card_str = r + s  # e.g. "3C"  "AH"
        results.append(card_str)
    return results


#################################
# 5. 解析与比较德州扑克牌型
#################################
rank_char_to_value = {
    '2':2, '3':3, '4':4, '5':5, '6':6,
    '7':7, '8':8, '9':9, 'T':10,
    'J':11, 'Q':12, 'K':13, 'A':14
}

def evaluate_five_cards(cards_5):
    """
    5张牌 -> 返回 (category, [若干tie-breaker信息]) 做牌型比较
    """
    rank_vals = sorted([rank_char_to_value[c[:-1]] for c in cards_5], reverse=True)
    suits = [c[-1] for c in cards_5]
    is_flush = (len(set(suits)) == 1)

    def check_straight(vals):
        if all(vals[i] - vals[i+1] == 1 for i in range(len(vals)-1)):
            return True, vals[0]
        # A2345 特判
        if vals == [14,5,4,3,2]:
            return True, 5
        return False, None

    is_straight, top_straight = check_straight(rank_vals)

    from collections import Counter
    c = Counter(rank_vals)
    freq_sort = sorted(((cnt, rv) for rv,cnt in c.items()), key=lambda x: (x[0], x[1]), reverse=True)

    # 同花顺
    if is_flush and is_straight:
        return (9, top_straight)
    # 四条
    if freq_sort[0][0] == 4:
        return (8, freq_sort[0][1], freq_sort[1][1])
    # 葫芦(3+2)
    if freq_sort[0][0] == 3 and freq_sort[1][0] == 2:
        return (7, freq_sort[0][1], freq_sort[1][1])
    # 同花
    if is_flush:
        return (6,) + tuple(rank_vals)
    # 顺子
    if is_straight:
        return (5, top_straight)
    # 三条
    if freq_sort[0][0] == 3:
        three_val = freq_sort[0][1]
        kickers = sorted((freq_sort[1][1], freq_sort[2][1]), reverse=True)
        return (4, three_val) + tuple(kickers)
    # 两对
    if freq_sort[0][0] == 2 and freq_sort[1][0] == 2:
        pair1 = freq_sort[0][1]
        pair2 = freq_sort[1][1]
        kicker = freq_sort[2][1]
        high_pair = max(pair1, pair2)
        low_pair = min(pair1, pair2)
        return (3, high_pair, low_pair, kicker)
    # 一对
    if freq_sort[0][0] == 2:
        pair_val = freq_sort[0][1]
        kickers = sorted([freq_sort[1][1], freq_sort[2][1], freq_sort[3][1]], reverse=True)
        return (2, pair_val) + tuple(kickers)
    # 高牌
    return (1,) + tuple(rank_vals)

def best_5_of_7(cards_7):
    """
    在7张牌里选出最好的5张组合 (Texas Hold'em经典)
    返回 (best_category_tuple, best5cards)
    """
    best_rank = None
    best_combo = None
    for combo in itertools.combinations(cards_7, 5):
        rank_info = evaluate_five_cards(combo)
        if (best_rank is None) or (rank_info > best_rank):
            best_rank = rank_info
            best_combo = combo
    return best_rank, best_combo

def compare_two_hands(common_cards, left_cards, right_cards):
    """
    比较左手牌 vs 右手牌谁更大
    """
    left_7 = common_cards + left_cards
    right_7 = common_cards + right_cards

    left_rank, left_best = best_5_of_7(left_7)
    right_rank, right_best = best_5_of_7(right_7)

    if left_rank > right_rank:
        return "left", left_rank, right_rank, left_best, right_best
    elif left_rank < right_rank:
        return "right", left_rank, right_rank, left_best, right_best
    else:
        return "tie", left_rank, right_rank, left_best, right_best


#################################
# 6. Flask 构建
#################################
app = Flask(__name__)

# 6.1 一次性加载模型
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")

model_global = CardNetResnet(freeze_layers=False)
model_path = "cardnet_resnet18.pth"

if os.path.isfile(model_path):
    st = torch.load(model_path, map_location=device)
    model_global.load_state_dict(st)
    print(f"[INFO] Model loaded from {model_path}")
else:
    print(f"[WARN] Model file {model_path} not found; predictions may fail.")

model_global.to(device)
model_global.eval()

# 6.2 首页: 返回前端页面 (假设你有 templates/index.html)
@app.route("/")
def index():
    return render_template("index.html")

# 6.3 核心接口: 一次性预测公共牌 & 左手牌 & 右手牌, 决定胜负
@app.route("/predict_all")
def predict_all():
    """
    读取:
      src/common (5张)
      src/hand_l (2张)
      src/hand_r (2张)
    批量推理, 并根据德州扑克规则对比哪边更大
    返回 JSON.
    """
    try:
        common_folder = os.path.join("src", "common")
        left_folder   = os.path.join("src", "hand_l")
        right_folder  = os.path.join("src", "hand_r")

        common_cards = predict_folder(model_global, common_folder, device)  # 5张
        left_cards   = predict_folder(model_global, left_folder, device)    # 2张
        right_cards  = predict_folder(model_global, right_folder, device)   # 2张

        if len(common_cards) != 5:
            return jsonify({"status":"error","msg":"公共牌必须5张, 实际:"+str(len(common_cards))}),400
        if len(left_cards) != 2:
            return jsonify({"status":"error","msg":"左手牌必须2张, 实际:"+str(len(left_cards))}),400
        if len(right_cards) != 2:
            return jsonify({"status":"error","msg":"右手牌必须2张, 实际:"+str(len(right_cards))}),400

        # 比较牌型
        winner, left_rank, right_rank, left_best5, right_best5 = compare_two_hands(common_cards, left_cards, right_cards)

        left_rank_str = f"{left_rank}"
        right_rank_str = f"{right_rank}"

        resp = {
            "status": "ok",
            "common": common_cards,   # list of 5
            "left": left_cards,       # list of 2
            "right": right_cards,     # list of 2
            "winner": winner,
            "left_best5": list(left_best5),
            "right_best5": list(right_best5),
            "left_rank_info": left_rank_str,
            "right_rank_info": right_rank_str
        }
        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"status":"error","msg":str(e)}),500

#################################
# 7. 运行
#################################
if __name__ == "__main__":
    # 默认 http://127.0.0.1:5000
    # 如果想让局域网访问可改成 host="0.0.0.0"
    app.run(debug=True)
