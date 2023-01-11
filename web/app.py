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

SEARCH_SIZE = 20
INDEX_NAME = "bd-vector-ik-liao"
app = Flask(__name__)


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



@app.route('/search')
def analyzer():
    # 加载本地模型
    # bc = Model("../example/roformer_chinese_sim_char_base")
    client = Elasticsearch('http://localhost:9200')

    model = SentenceTransformer('../example/distilbert-base-uncased-2023-01-07_12-26-36')
    # 从网页端获得请求
    query = request.args.get('q')

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
    #
    script_query = {



                "match": {
                    "title": query
                }



    }
    # script_query = {
    #     "function_score": {
    #         "query": {
    #             "multi_match": {
    #                 "title": query
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
    #                         "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
    #                         "params": {"query_vector": query_vector},
    #                         "lang": "painless"
    #                     }
    #                 }
    #             }
    #         ],
    #         "boost": 10,
    #         "boost_mode":"multiply",
    #         "score_mode": "sum"
    #     }
    #
    #
    #     #
    #
    #     #}
    # }



    # response = client.knn_search(
    #     index=INDEX_NAME,
    #     body={
    #         "knn": {
    #             "field": "text_vector",
    #             "query_vector": query_vector,
    #             "k": 10,
    #             "num_candidates": 10
    #
    #         },
    #         # "fields": ["title"]
    #         "_source": {"includes": ["title", "answer"]}
    #     }
    # )
    response = client.search(
        index=INDEX_NAME,
        body={
            "size": SEARCH_SIZE,
            "query": script_query,
            "_source": {"includes": ["title", "answer"]}
        }
    )
    print(script_query)
    print(query)
    print(type(query_vector))
    print(response)
    print(type(response["took"]))
    print(len(response["hits"]["hits"]))
    for i in range(10):
        print(response["hits"]["hits"][i]["_source"]["title"])
        i = i+1

    return jsonify(response.raw)


if __name__ == '__main__':
    app.run(host="localhost", port=5000)
