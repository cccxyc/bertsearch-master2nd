import os
from pprint import pprint
import argparse
import json
import torch
import numpy as np
from roformer import RoFormerForCausalLM, RoFormerConfig
from transformers import BertTokenizer
import pandas as pd
from flask import Flask, render_template, jsonify, request
from elasticsearch import Elasticsearch
#from bert_serving.client import BertClient
from sentence_transformers import SentenceTransformer, util
from elastic_transport import ObjectApiResponse

SEARCH_SIZE = 10
INDEX_NAME = "bd-vector-ik-lin-2"
app = Flask(__name__)
APlist = []
RRlist = []
def read_txt_file(file_path):
    data = []
    file = open(file_path, 'r',encoding='utf-8')  # 打开文件
    file_data = file.readlines()  # 读取所有行
    for row in file_data:
        data.append(row.split('\n')[0])
    return data



class Model(object):
    def __init__(self, pretrain_model_path):
        # 模型配置
        pretrained_model = pretrain_model_path

        # 建立分词器
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

        # 加载模型
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        config = RoFormerConfig.from_pretrained(pretrained_model)
        config.is_decoder = True
        config.eos_token_id = self.tokenizer.sep_token_id
        config.pooler_activation = "linear"
        self.model = RoFormerForCausalLM.from_pretrained(pretrained_model, config=config)
        self.model.to(self.device)
        self.model.eval()

    def encoder_predict(self, r):
        inputs2 = self.tokenizer(r, padding=True, return_tensors="pt")
        with torch.no_grad():
            inputs2.to(self.device)
            outputs = self.model(**inputs2)
            Z = outputs.pooler_output.cpu().numpy()
        return Z



@app.route('/')
def index():
    return render_template('index-bd.html')




def cal_metric (path):
    # 加载本地模型

    client = Elasticsearch('http://localhost:9200')

    # model = SentenceTransformer('../example/sbert-chinese-qmc-domain')
    model = SentenceTransformer('../example/sbert_base_chinese_softmaxloss_BinaryClassificationEvaluator')
    # 从网页端获得请求
    # query = request.args.get('q')
    files = os.listdir(path)
    mpList = []
    mrList = []
    for f in files:
        query = f.split(".")[0]

        # 计算网页输入的句向量
        # query_vector = bc.encoder_predict([query])[0].tolist()
        query_vector = model.encode(query).tolist()

        # 根据 语义相似分数+字面相似分数 计算查询分数
        script_query = {

            "script_score": {
                "query": {
                    "match": {
                        "title": query
                    }


                },
                "script": {
                    "source": " 0.1 * _score + cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                    "params": {"query_vector": query_vector}

                }
            },

        }

        # script_query = {
        #     "function_score": {
        #         "query": {
        #             "match": {
        #
        #                     "title": query,
        #
        #
        #
        #             }
        #         },
        #         "functions": [
        #             {
        #                 "filter": {
        #                     "match_all": {
        #                         "boost": 1
        #                     }
        #                 },
        #                 "script_score": {
        #                     "script": {
        #                         "source": "Math.max(cosineSimilarity(params.query_vector, 'text_vector') + 1.0,1)",
        #                         "params": {"query_vector": query_vector},
        #                         "lang": "painless"
        #                     }
        #                 }
        #             }
        #         ],
        #         "boost": 1,
        #         "boost_mode": "sum",
        #         "score_mode": "sum"
        #     }
        #
        #     #
        #
        #     # }
        # }

        response = client.search(
            index=INDEX_NAME,
            body={
                "size": SEARCH_SIZE,
                "query": script_query,
                "_source": {"includes": ["title", "answer","text_vector"]}
            }
        )
        print(script_query)
        print(query)
        print(type(query_vector))
        print(response)
        print(type(response["took"]))
        print(len(response["hits"]["hits"]))
        rightNum = 0
        rankIndex = 0
        curPList = []
        for i in range(10):
            # print(response["hits"]["hits"][i]["_source"]["title"])
            rankIndex += 1
            corpus_sentences = read_txt_file('../example/eval_txt/'+query+'.txt')
            # corpus_embeddings = bc.encoder_predict(corpus_sentences)
            corpus_embeddings = model.encode(corpus_sentences)
            query_embedding = response["hits"]["hits"][i]["_source"]["text_vector"]
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=5)

            print("\n\n======================\n\n")
            print("Query:", response["hits"]["hits"][i]["_source"]["title"])

            print("\nTop 5 most similar sentences in corpus:")

            for score, idx in zip(top_results[0], top_results[1]):
                print(corpus_sentences[idx], "(Score: {:.4f})".format(score))

            if top_results[0][0]>0.75:
                rightNum +=1
                p = float(rightNum) / rankIndex
                curPList.append(p)

        if len(curPList) > 0:
            RRlist.append(curPList[0])
            APlist.append(float(sum(curPList)) / len(curPList))
        i = i + 1
        print("curPList",curPList)
        if len(curPList)>0:
            print("MAP::  ", curPList[0])
            mpList.append(curPList[0])
            print("MRR::  ",float(sum(curPList)) / 10)
            mrList.append(float(sum(curPList)) / 10)
        else:
            mpList.append(0)
            mrList.append(0)

        print("mpList:::   ",mpList)
        print("MRR::  ", float(sum(mpList)) / 10)
        print("mrList:::   ", mrList)
        print("MAP::  ", float(sum(curPList)) / 10)
if __name__ == '__main__':
    # app.run(host="localhost", port=5000)
    # cal_metric("../example/eval_txt")
    cal_metric("../example/eval_txt")