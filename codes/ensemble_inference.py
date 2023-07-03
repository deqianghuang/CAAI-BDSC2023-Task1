import argparse
import logging
import torch
import os
from model import KGEModel
import ipdb
import json
import pandas as pd
from collections import defaultdict as ddict
import numpy as np
import heapq


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def save_to_json(file_name, file_dict):
    file = open(file_name, 'w', encoding='utf-8')
    file.write(json.dumps(file_dict))
    file.close()
    return


def get_pred(args, model_name, query):
    
    if os.path.exists(f"{model_name}/scores.npy"):
        logging.info(f"Prediction for {model_name} already exists. Skipping...")
        return
    
    model = load_model(args, model_name)
    
    logging.info('Start inference...')
    
    if args.cuda:
        model = model.cuda()
    
    model.eval()
    
    scores = []
    idx = 0
    while idx < len(query):
        next_idx = min(idx + args.test_batch_size, len(query))
        
        batch_query = torch.tensor(query[idx: next_idx], dtype=torch.long)
        tail_batch = torch.stack([torch.arange(args.nentity) for _ in range(batch_query.shape[0])], dim=0)
        
        # batch_query = batch_query.repeat(1, args.nentity).reshape(-1, 2)
        # all_entities = all_entities.repeat(next_idx - idx)
        
        # batch_triple = torch.cat([batch_query, all_entities.unsqueeze(1)],dim=1)
        
        if args.cuda:
            # batch_triple = batch_triple.cuda()
            batch_query = batch_query.cuda()
            tail_batch = tail_batch.cuda()
        
        score = model((batch_query, tail_batch), "tail-batch").reshape(-1, args.nentity).cpu().detach()
        
        # batch_triple[:, 1] += args.nrelation
        # ipdb.set_trace()
        # score += model(batch_triple[:, [2, 1, 0]], "single").reshape(-1, args.nentity).cpu().detach()
        # score /= 2
        
        scores.append(score)
        
        idx = next_idx
        if idx % 100 == 0:
            logging.info(f"Predicting: {idx} / {len(query)}")
    
    model = model.cpu()
    scores = torch.cat(scores, dim=0).numpy()
    np.save(f"{args.init_checkpoint}/scores.npy", scores)
    logging.info(f"Successfully save model prediction to {model_name}/scores.npy!")
    
    return scores


def ensemble_predict(args, model_list, query):
    for model_name in model_list:
        try:
            args.cuda = True
            get_pred(args, model_name, query)
        except RuntimeError as e:
            args.cuda = False
            logging.info("OOM! Set cuda to False for inference...")
            get_pred(args, model_name, query)


def get_candidate_voter_list(args, model_list, query, triple2idx, all_true_triples, id2entity):
    score = 0
    for model_name in model_list:
        model_pred = np.load(f"{model_name}/scores.npy")
        score += sigmoid(model_pred)
    
    score /= len(model_list)
    
    if args.use_bayes_weight:
        rel_bayes_weight = np.load("data/BDSC/rel_weight.npy", allow_pickle=True).item()
        query_rel = np.array(query)[:, 1]
        for i, qr in enumerate(query_rel):
            score[i] *= rel_bayes_weight[qr]
    
    if args.filter:
        logging.info("Masking Scores...")
        mask = np.zeros((score.shape[1], )) > 0
        
        train_and_valid_ent = [triple[0] for triple in all_true_triples] + [triple[2] for triple in all_true_triples]
        test_ent = [q[0] for q in query]
        all_ent = train_and_valid_ent + test_ent
        all_ent = set(all_ent)
        
        for i in range(mask.shape[0]):
            mask[i] = False if i in all_ent else True
        score[:, mask] = -np.inf

    facts = ddict(set)
    
    for (s, r, o) in all_true_triples:
        facts[(s, r)].add(o)
    
    submission = []
    
    for i in range(len(query)):
        s, r = query[i]
        train_list = np.array(list(facts[(s,r)]), dtype=np.int64)
        pred = score[i]
        pred[train_list] = -np.inf
        pred_dict = {idx: score for idx, score in enumerate(pred)}
        candidate_voter_dict = heapq.nlargest(5, pred_dict.items(), key=lambda kv :kv[1])
        candidate_voter_list = [id2entity[k] for k,v in candidate_voter_dict]
        submission.append({
                'triple_id': '{:04d}'.format(triple2idx[(s,r)]),
                'candidate_voter_list': candidate_voter_list
            })
        if (i + 1) % 100 == 0:
            logging.info(f"Submission Generating: {i + 1} / {len(query)}")
    logging.info(f"Successfully generate {args.exp}_preliminary_submission !")
    
    save_to_json(f"{args.exp}_final_submission.json", submission)


def prepare_test_query(args, entity2id, relation2id):
    df = pd.read_json("{}/target_event_final_test_info.json".format(args.data_path))
    records = df.to_dict('records')
    
    triple2idx = dict()
    query = dict()
    for line in records:
        triple_id = int(line['triple_id'])
        sub, rel = line['inviter_id'], line['event_id']
        sub_id, rel_id = entity2id[sub], relation2id[rel]
        query[(sub_id, rel_id)] = set()
        triple2idx[(sub_id, rel_id)] = triple_id
    return list(query.keys()), triple2idx


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--data_path', type=str, default="data/BDSC")
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    
    parser.add_argument('-cpu', '--cpu_num', default=16, type=int)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--use_bayes_weight', action='store_true')
    parser.add_argument('--filter', action='store_true')
    
    
    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.gamma = argparse_dict['gamma']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.triple_relation_embedding = argparse_dict['triple_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.triplere_u = argparse_dict['triplere_u']

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    log_file = os.path.join(f'{args.exp}_ensemble_generate_submission.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)


def load_model(args, init_checkpoint):
    args.init_checkpoint = init_checkpoint
    override_config(args)
    checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'), map_location='cpu')
    logging.info('Loading checkpoint %s...' % args.init_checkpoint)
    
    kge_model = KGEModel(
    model_name=args.model,
    nentity=args.nentity,
    nrelation=args.nrelation,
    hidden_dim=args.hidden_dim,
    gamma=args.gamma,
    double_entity_embedding=args.double_entity_embedding,
    double_relation_embedding=args.double_relation_embedding,
    triple_relation_embedding=args.triple_relation_embedding
)
    
    kge_model.load_state_dict(checkpoint['model_state_dict'])
    
    return kge_model


def inference(args):
    
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    id2entity = {v: k for k, v in entity2id.items()}
    
    args.nentity = len(entity2id)
    args.nrelation = len(relation2id)
    
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % args.nentity)
    logging.info('#relation: %d' % args.nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
   # ipdb.set_trace()
    #All true triples
    all_true_triples = train_triples + valid_triples
    
    model_list = ["models/HAKE_221s","models/RotatE_221s","models/RotatE_141s","models/HAKE_141s","models/HAKE_141s","models/HAKE_221","models/HAKE_511"]#,"models/HAKE_BDSC_221old","models/RotatE_BDSC_141","models/HAKE_BDSC_141","models/RotatE_BDSC_221old","models/HAKE_BDSC_221old","models/RotatE_BDSC_1","models/HAKE_BDSC_1",] # , "models/HAKE_BDSC_0"]
    
    query, triple2idx = prepare_test_query(args, entity2id=entity2id, relation2id=relation2id)
    
    ensemble_predict(args, model_list, query)
    
    get_candidate_voter_list(args, model_list, query, triple2idx, all_true_triples, id2entity)
    


if __name__ == "__main__":
    args = parse_args()
    inference(args)