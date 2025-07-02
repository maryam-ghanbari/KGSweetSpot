import pandas as pd
from conceptnet import merged_relations, merged_relations_clean, relation_text
from tqdm import tqdm
import argparse
import scipy
import networkx as nx
import json


concept2id = None
id2concept = None
relation2id = None
id2relation = None

cpnet = None
cpnet_all = None
cpnet_simple = None

pruned_graph = '/content/qagnn-main/data/cpnet/conceptnet.en.pruned.graph'
vocab_graph = '/content/qagnn-main/data/cpnet/concept.txt'
df_statement = pd.read_json('/content/qagnn-main/data/csqa/statement/dev.statement.jsonl', lines=True)
df_grounded = pd.read_json('/content/qagnn-main/data/csqa/grounded/dev.grounded.jsonl', lines=True)
pickleFile = open("/content/qagnn-main/data/csqa/graph/dev.graph.adj.pk", "rb")
test_pruned_graph_df = pd.read_pickle(pickleFile)


def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation

    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations_clean
    relation2id = {r: i for i, r in enumerate(id2relation)}


def find_knowledge_set_multi_relation(adj_matrix, concepts_list, qmask, amask, id2concept, id2relation, serialization, scope):
    # 1) reshape to (R, N, N)
    A = adj_matrix.toarray() if hasattr(adj_matrix, "toarray") else adj_matrix
    N = len(concepts_list)
    R = len(id2relation)
    A = A.reshape(R, N, N)

    # 2) precompute index sets
    qc_idx = [i for i,b in enumerate(qmask) if b]
    ac_idx = [i for i,b in enumerate(amask) if b]
    ex_idx = [i for i in range(N) if i not in qc_idx + ac_idx]

    # helper to turn a (i → j via r) into text
    def emit_triple(i, r, j, out_dict):
        path = f"{id2concept[concepts_list[i]]} {id2relation[r]} {id2concept[concepts_list[j]]}"
        out_dict[path] = path

    def emit_path(i, r1, k, r2, j, out_dict, swap_second=False):
        """
        Emit a full 2-hop path. By default it’s
            extra → ac  for the second hop,
        but if swap_second=True it’ll do
            ac → extra
        """
        # first hop is always i → k
        t1 = (
            f"{id2concept[concepts_list[i]]} "
            f"{id2relation[r1]} "
            f"{id2concept[concepts_list[k]]}"
        )
        # second hop can be k → j or j → k
        if not swap_second:
            t2 = (
                f"{id2concept[concepts_list[k]]} "
                f"{id2relation[r2]} "
                f"{id2concept[concepts_list[j]]}"
            )
        else:
            t2 = (
                f"{id2concept[concepts_list[j]]} "
                f"{id2relation[r2]} "
                f"{id2concept[concepts_list[k]]}"
            )
        path = f"{t1}.{t2}"
        out_dict[path] = path


    knowledge_set = {}

    # --- 1-hop: direct qc→ac and ac→qc ---
    if scope in ("1hop","both"):
        for r in range(R):
            M = A[r]
            for i in qc_idx:
                for j in ac_idx:
                    if M[i, j]:
                        emit_triple(i, r, j, knowledge_set)
                        
            for i in ac_idx:
                for j in qc_idx:
                    if M[i, j]:
                        emit_triple(i, r, j, knowledge_set)

    # --- 2-hop via each extra node, four patterns ---
    if scope in ("2hop","both"):
        for r1 in range(R):
            M1 = A[r1]
            for r2 in range(R):
                M2 = A[r2]

                # Forward→Forward: qc→ex→ac
                for i in qc_idx:
                    for k in ex_idx:
                        if not M1[i, k]:
                            continue
                        for j in ac_idx:
                            if not M2[k, j]:
                                continue
                            if serialization == "triple":
                                emit_triple(i,  r1, k, knowledge_set)
                                emit_triple(k,  r2, j, knowledge_set)
                            else:
                                emit_path(i, r1, k, r2, j, knowledge_set, swap_second=False)

                # Forward→Reverse: qc→ex & ac→ex
                for i in qc_idx:
                    for k in ex_idx:
                        if not M1[i, k]:
                            continue
                        for j in ac_idx:
                            if not M2[j, k]:
                                continue
                            if serialization == "triple":
                                emit_triple(i,  r1, k, knowledge_set)
                                emit_triple(j,  r2, k, knowledge_set)
                            else:
                                emit_path(i, r1, k, r2, j, knowledge_set, swap_second=True)

                # Reverse→Forward: ex→qc & ex→ac
                for k in ex_idx:
                    for i in qc_idx:
                        if not M1[k, i]:
                            continue
                        for j in ac_idx:
                            if not M2[k, j]:
                                continue
                            if serialization == "triple":
                                emit_triple(k, r1, i, knowledge_set)
                                emit_triple(k, r2, j, knowledge_set)
                            else:
                                emit_path(k, r1, i, r2, j, knowledge_set, swap_second=False)

                # Reverse→Reverse: ex→qc & ac→ex
                for k in ex_idx:
                    for i in qc_idx:
                        if not M1[k, i]:
                            continue
                        for j in ac_idx:
                            if not M2[j, k]:
                                continue
                            if serialization == "triple":
                                emit_triple(k, r1, i, knowledge_set)
                                emit_triple(j, r2, k, knowledge_set)
                            else:
                                emit_path(k, r1, i, r2, j, knowledge_set, swap_second=True)
    return knowledge_set


def get_knowledge_statements_relevance_scoring(ground_statement_object, serialization, scope):
    
    global concept2id, id2concept, relation2id, id2relation, cpnet_simple, cpnet
    if any(x is None for x in [concept2id, id2concept, relation2id, id2relation]):
        load_resources(vocab_graph)
    
    concepts_list = ground_statement_object["concepts"]
    qmask = ground_statement_object["qmask"]
    amask = ground_statement_object["amask"]
    adj_matrix = ground_statement_object["adj"]

    # Extract the knowledge paths, passing the mappings to convert IDs to text
    knowledge_set = find_knowledge_set_multi_relation(
        adj_matrix, concepts_list, qmask, amask, id2concept, id2relation, serialization, scope
    )
    return knowledge_set


def get_knowledge_set_relevance_scoring(serialization, scope):
    knowledge_set = {}
    for index_statement in tqdm(range(len(df_statement))):
        question_knowledge_set = {}
        for index_statement_option in df_statement.iloc[index_statement]["statements"]:
            # Only process statements labeled as true
            # if not index_statement_option["label"]:
            #     continue  

            # Only process statements labeled as false
            # if index_statement_option["label"]:
            #     continue  
            statement = index_statement_option["statement"]
            statements_set = {}
            for index_ground in range(len(df_grounded)):
                if df_grounded.iloc[index_ground]["sent"] == statement:
                    statements_set = get_knowledge_statements_relevance_scoring(test_pruned_graph_df[index_ground], serialization, scope)
                    break
            question_knowledge_set.update(statements_set)

        question_knowledge_set_array = [value for key, value in question_knowledge_set.items() if value != '']
        knowledge_set[df_statement.iloc[index_statement]["question"]["stem"]] = question_knowledge_set_array

    return knowledge_set

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument('--serialization', default=['triple'], choices=['triple', 'path'])
    parser.add_argument('--scope', default=['both'], choices=['1hop', '2hop', 'both'])

    args = parser.parse_args()

    load_resources(vocab_graph)
    knowledge_dict = get_knowledge_set_relevance_scoring(args.serialization, args.scope)

    output_list = []
    for _, stmt_row in tqdm(df_statement.iterrows(), total=len(df_statement)):
        stem       = stmt_row["question"]["stem"]
        cands      = [ ch["text"] for ch in stmt_row["question"]["choices"] ]
        answer     = cands[ord(stmt_row["answerKey"]) - ord("A")]
        knowledges = knowledge_dict.get(stem, [])

        output_list.append({
            "query":      stem,
            "cands":      cands,
            "answer":     answer,
            "knowledges": knowledges
        })

    with open(args.output, "w") as f:
        json.dump(output_list, f, indent=2)

if __name__ == '__main__':
    main()
