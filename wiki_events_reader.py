import pandas as pd
import os
import logging
from tqdm import tqdm
import webanno_tsv as tsv


def read_wiki_events_from_jsonl(path):
    logging.info("Reading from {}".format(path))
    if not os.path.exists(path):
        logging.error("Cannot read the dataset from your directory \n"
                      "Please check again!")
    train_df = pd.read_json(path + '/train.jsonl', lines=True)
    dev_df = pd.read_json(path + '/dev.jsonl', lines=True)
    test_df = pd.read_json(path + '/test.jsonl', lines=True)
    train = train_df.to_dict('records')
    dev = dev_df.to_dict('records')
    test = test_df.to_dict('records')
    return train, dev, test
    pass


def parse_to_tsv(dataset, save_path: str):
    if save_path.endswith('/'):
        save_path = save_path[:-1]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    layer_defs = [('webanno.custom.WikiEventType', ['value'])]
    for document in tqdm(dataset):
        event_sent_dict = {}
        for event in document['event_mentions']:
            if event['trigger']['sent_idx'] not in event_sent_dict:
                event_sent_dict[event['trigger']['sent_idx']] = [event]
            else:
                event_sent_dict[event['trigger']['sent_idx']] += [event]
        sent_sum_len = []
        last_sent_len = 0
        for sent_idx, sentence in enumerate(document['sentences']):
            last_sent_len += len(sentence[0])
            sent_sum_len.append(last_sent_len)
            if sent_idx not in event_sent_dict:
                continue
            else:
                tokens_sentence = [[item[0] for item in sentence[0]]]
                doc = tsv.Document.from_token_lists(tokens_sentence)
                annotations = []
                for event in event_sent_dict[sent_idx]:
                    start = event['trigger']['start'] - sent_sum_len[sent_idx]
                    end = event['trigger']['end'] - sent_sum_len[sent_idx]
                    annotations.append(tsv.Annotation(doc.tokens[start:end],
                                                      layer='webanno.custom.WikiEventType',
                                                      field='value',
                                                      label=event['event_type']))
                doc = tsv.replace(doc, annotations=annotations, layer_defs=layer_defs)
                for event in event_sent_dict[sent_idx]:
                    if not os.path.exists('{}/{}'.format(save_path, event['event_type'])):
                        os.mkdir('{}/{}'.format(save_path, event['event_type']))
                    with open("{}/{}/{}.tsv".format(save_path, event['event_type'], event['id']), "w") as f:
                        f.write(doc.tsv())


if __name__ == "__main__":
    train, dev, test = read_wiki_events_from_jsonl('/home/tapati/event_extraction/WikiEventDataset')
    print(train[10].keys())
    print(train[10]['doc_id'])
    print(train[10]['tokens'])
    print(train[10]['sentences'])
    print(train[10]['text'])
    print(train[10]['event_mentions'])
    print("")

    parse_to_tsv(train, '/home/tapati/event_extraction/WikiEventDataset/tsv')
    parse_to_tsv(dev, '/home/tapati/event_extraction/WikiEventDataset/tsv')
    parse_to_tsv(test, '/home/tapati/event_extraction/WikiEventDataset/tsv')