"""
Example script to create elasticsearch documents.
"""
import argparse
import json
import torch
import numpy as np
from roformer import RoFormerForCausalLM, RoFormerConfig
from transformers import BertTokenizer
import pandas as pd
#from bert_serving.client import BertClient



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

bc = Model("./roformer_chinese_sim_char_base")
# = BertClient(output_fmt='list')


def create_document(doc, emb, index_name):
    return {

        'text': doc['text'],
        'title': doc['title'],
        'text_vector': emb
    }


def load_dataset(path):
    docs = []
    df = pd.read_csv(path)
    for row in df.iterrows():
        series = row[1]
        doc = {
            'title': series.Title,
            'text': series.Description
        }
        docs.append(doc)
    return docs


def bulk_predict(docs, batch_size=256):
    """Predict bert embeddings."""
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i: i+batch_size]
        embeddings = bc.encoder_predict([doc['text'] for doc in batch_docs])

        for emb in embeddings:
            yield emb


def main(args):
    docs = load_dataset(args.data)
    with open(args.save, 'w') as f:
        for doc, emb in zip(docs, bulk_predict(docs)):
            emb1 = emb.tolist()
            print(len(emb1))
            d = create_document(doc, emb1, args.index_name)
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating elasticsearch documents.')
    parser.add_argument('--data',default="example.csv", help='data for creating documents.')
    parser.add_argument('--save', default='documents5.json', help='created documents.')
    parser.add_argument('--index_name', default='jobsearch', help='Elasticsearch index name.')
    args = parser.parse_args()
    main(args)
