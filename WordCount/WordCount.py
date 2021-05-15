import numpy as np
from mpi4py import MPI
import jieba
import random
import math
import sys


def WordCount_P(local_data, local_text):
    """并行部分的WordCount"""
    for i in local_text:
        if i in stopword:
            continue
        elif i in local_data.keys():
            local_data[i] += 1
        else:
            local_data[i] = 1
    return local_data


def MergeDict(Dicts):
    """字典合并 TODO: 合并的并行实现"""
    Keys = []
    ans = {}
    for i in Dicts:
        Keys.extend(i.keys())
    for i in Keys:
        ans[i] = 0
    for i in Keys:
        for j in Dicts:
            if i in j.keys():
                ans[i] += j[i]
    return ans


def WashText(text):
    """如果是中文需要手动分词并清洗，不是则直接按空格分词"""
    test = random.choices(text, k=10)
    flag = 0
    for i in test:
        if "\u4e00" <= i <= "\u9fa5":
            flag = 1
            break
    if flag == 1:
        text = jieba.lcut(text)
    else:
        text.split(" ")
    return text


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

text = None
local_size = None
stopword = None
local_text = []
local_data = {}

K = int(sys.argv[1])

with open("stopword.txt", "r") as f:
    stopword = f.read().split("\n")
    stopword.extend(["\n", "\u3000"])

if rank == 0:

    with open("text.txt", "r") as f:
        text = f.read()
    text=WashText(text)
    print(f"加载完成，预览：\n{text[:100]}")

    len_text = len(text)
    local_size = math.ceil(len_text / size)
    for i in range(size):
        start = i * local_size
        end = (i + 1) * local_size
        local_text.append(text[start:end])

local_text = comm.scatter(local_text, root=0)
local_data = WordCount_P(local_data, local_text)
local_data = comm.gather(local_data, root=0)

if rank == 0:
    MergeDict(local_data)

    sorted_dict = sorted(
        local_data[0].items(), key=lambda kv: (kv[1], kv[0]), reverse=True
    )
    for i in range(K):
        print(sorted_dict[i])

    with open("output.txt", "w+") as f:
        for i in sorted_dict:
            f.write(f"{i[0]},{i[1]}\n")
