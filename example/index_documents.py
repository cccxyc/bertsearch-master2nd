"""
Example script to index elasticsearch documents.
"""
import argparse
import json

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


def load_dataset(path):
    with open(path,encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main(args):
    client = Elasticsearch('http://localhost:9200')
    docs = load_dataset(args.data)
    bulk(client, docs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indexing elasticsearch documents.')
    parser.add_argument('--data', default='documents-bd-op-1-lin-2.json', help='Elasticsearch documents.')
    args = parser.parse_args()
    main(args)
