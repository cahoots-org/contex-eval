"""Replay Contex's hybrid fusion offline on its real vector + BM25 rankings.

Pulls each query's top-100 vector ranking (pgvector, exact via ef_search/iterative scan) and
top-100 BM25 ranking (ParadeDB, the same SQL Contex runs) straight from the Contex Postgres,
then scores RRF fusion variants — so one publish answers "which fusion change moved recall?"

Usage:  python scripts/fusion_replay.py <project> [k]
Needs:  the eval stack's postgres container (contexeval-postgres-1) and data/questions.jsonl.
"""
import json
import random
import subprocess
import sys
from collections import defaultdict

from sentence_transformers import SentenceTransformer

from contexeval import config

PG = ["docker", "exec", "-i", "contexeval-postgres-1", "psql", "-U", "contex", "-d", "contex",
      "-tA", "-F", "\t"]


def pull(project, qs):
    """{'V'|'L': {qi: [para_id, ...]}} — Contex's vector and lexical top-100 per query."""
    model = SentenceTransformer(config.EMBED_MODEL, device="cpu")  # same model Contex embeds with
    vecs = model.encode([q["question"] for q in qs], normalize_embeddings=True)
    sql = ["set hnsw.ef_search=200; set hnsw.iterative_scan=strict_order;"
           "set paradedb.planner_warnings='off';"]
    for i, (q, v) in enumerate(zip(qs, vecs)):
        lit = "'[" + ",".join(f"{x:.7f}" for x in v) + "]'"
        qt = q["question"].replace("'", "''")
        sql.append(f"select 'V',{i},node_key from (select node_key from embeddings "
                   f"where project_id='{project}' order by embedding <=> {lit} limit 100) t;")
        sql.append(f"select 'L',{i},node_key from (select node_key, paradedb.score(id) s "
                   f"from embeddings where project_id='{project}' and (description @@@ '{qt}' "
                   f"or data_original @@@ '{qt}') order by s desc limit 100) t;")
    out = subprocess.run(PG, input="\n".join(sql), capture_output=True, text=True, check=True).stdout
    ranks = {"V": defaultdict(list), "L": defaultdict(list)}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] in ranks:
            ranks[parts[0]][int(parts[1])].append(parts[2].split(".", 1)[0])
    return ranks


def rrf(lists, k=60):
    s = defaultdict(float)
    for lst in lists:
        for r, x in enumerate(lst):
            s[x] += 1 / (k + r + 1)
    return sorted(s, key=lambda x: -s[x])


def boot_ci(a, b, n=10000, seed=13):
    d = [x - y for x, y in zip(a, b)]
    rnd = random.Random(seed)
    ms = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), ms[int(0.025 * n)], ms[int(0.975 * n)]


def main(project, k=10):
    qs = [json.loads(line) for line in open(config.QUESTIONS_PATH)]
    R = pull(project, qs)

    def recall(fuse):
        out = []
        for i, q in enumerate(qs):
            gold = set(q["gold_para_ids"])
            out.append(len(set(fuse(R["V"][i], R["L"][i])[:k]) & gold) / len(gold))
        return out

    vec = recall(lambda v, l: v)
    print(f"vector only            recall@{k}={sum(vec)/len(vec):.3f}")
    bm = recall(lambda v, l: l)
    print(f"bm25 only (ParadeDB)   recall@{k}={sum(bm)/len(bm):.3f}")
    by_pool = {}
    for p in sorted({k, 10, 20, 30, 50, 100}):
        by_pool[p] = recall(lambda v, l, p=p: rrf([v[:p], l[:p]]))
        print(f"RRF depth {p:3d}          recall@{k}={sum(by_pool[p])/len(by_pool[p]):.3f}")
    drop = recall(lambda v, l: [x for x in rrf([v, l]) if x in set(v)])
    print(f"RRF depth 100, no lexical-only hits  recall@{k}={sum(drop)/len(drop):.3f}")

    print(f"\npaired bootstrap 95% CI (10k resamples):")
    for name, a, b in [(f"depth{k} - depth100", by_pool[k], by_pool[100]),
                       (f"depth{k} - vector", by_pool[k], vec),
                       ("depth100 - vector", by_pool[100], vec)]:
        print(f"  {name:20s} mean=%+.3f CI=[%+.3f,%+.3f]" % boot_ci(a, b))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)
