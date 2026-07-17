"""
GraphRAG retrieval, adapted to GraphRAG-Bench's novel_questions.json.

IMPORTANT -- what this dataset actually is:
  This file is a QA EVALUATION SET, not a corpus + graph. Its real columns are:
      id, question, question_type, evidence, answer
  There are no (subject, relation, object) triples in it, and no pre-chunked
  corpus. `evidence` holds the supporting passage(s) used to answer each
  question -- that's the closest thing to retrievable "content" it ships with.

What this script does:
  1. Loads novel_questions.json as `questions_df` (id, question, question_type,
     evidence, answer).
  2. Builds a `children_df` corpus OUT OF the evidence passages themselves
     (one row per evidence snippet, across all questions) -- so you have
     something concrete to vector-search against and test the mechanics on.
  3. Provides an OPTIONAL triples_df hook: if you have (or build) real
     entity/relation triples over this corpus, plug them in and graph
     traversal activates. Without it, retrieval still works -- it just runs
     as vector-search-only (graph_expand simply returns no extra entities).
  4. Adds an evaluation loop: for each question, run retrieval, check whether
     the retrieved chunks overlap with that question's own gold evidence
     (a simple recall proxy) so you can see whether retrieval is working.
"""

import re
import numpy as np
import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------------
# Step 1: load the dataset
# ---------------------------------------------------------------------------

def load_questions_df(path_or_url="hf://datasets/GraphRAG-Bench/GraphRAG-Bench/Datasets/Questions/novel_questions.json"):
    df = pd.read_json(path_or_url)
    return df


# ---------------------------------------------------------------------------
# Step 2: build a searchable corpus out of the evidence passages
# ---------------------------------------------------------------------------

def build_children_df_from_evidence(questions_df):
    """
    `evidence` may be a single string or a list of strings depending on the
    row -- handled defensively either way. Each individual evidence passage
    becomes one row in children_df, tagged with which question it originally
    supports (useful for the evaluation step below).
    """
    rows = []
    counter = 0

    for row in questions_df.itertuples(index=False):
        evidence = row.evidence
        if evidence is None:
            continue

        # normalize to a list of passages regardless of source format
        if isinstance(evidence, str):
            passages = [evidence]
        elif isinstance(evidence, (list, tuple, np.ndarray)):
            passages = [str(e) for e in evidence]
        else:
            passages = [str(evidence)]

        for passage in passages:
            passage = passage.strip()
            if not passage:
                continue
            rows.append({
                "child_id": f"c{counter}",
                "parent_id": f"q{row.id}",       # groups evidence by originating question
                "heading": getattr(row, "question_type", None),
                "content": passage,
                "source_question_id": row.id,     # used only for evaluation, not retrieval
            })
            counter += 1

    children_df = pd.DataFrame(rows).drop_duplicates(subset="content").reset_index(drop=True)
    return children_df


# ---------------------------------------------------------------------------
# Step 3: graph (optional -- empty/no-op if you have no triples yet)
# ---------------------------------------------------------------------------

def build_graph_from_df(triples_df):
    """
    triples_df columns: subject, relation, object, source_chunk
    If you don't have real triples yet, pass an empty DataFrame with those
    columns -- graph_expand will simply return no additional entities and
    graph_rag_retrieve_df degrades gracefully to vector-search-only.
    """
    graph = nx.MultiDiGraph()
    if triples_df is None or triples_df.empty:
        return graph

    for row in triples_df.itertuples(index=False):
        for entity in (row.subject, row.object):
            if graph.has_node(entity):
                graph.nodes[entity]["mentioned_in"].add(row.source_chunk)
            else:
                graph.add_node(entity, mentioned_in={row.source_chunk})
        graph.add_edge(row.subject, row.object, relation=row.relation, source_chunk=row.source_chunk)

    return graph


def entities_in_chunk(graph, child_id):
    return [n for n, data in graph.nodes(data=True) if child_id in data.get("mentioned_in", set())]


def graph_expand(graph, seed_entities, hops=1):
    if graph.number_of_nodes() == 0:
        return set()
    undirected = graph.to_undirected()
    expanded = set(seed_entities)
    frontier = set(seed_entities)
    for _ in range(hops):
        next_frontier = set()
        for entity in frontier:
            if entity in undirected:
                next_frontier.update(undirected.neighbors(entity))
        expanded.update(next_frontier)
        frontier = next_frontier
    return expanded


# ---------------------------------------------------------------------------
# Step 4: vector search
# ---------------------------------------------------------------------------

def vector_search_df(query, children_df, embedding_model, top_k=3):
    query_vec = np.array(embedding_model.embed_query(query))
    query_vec /= np.linalg.norm(query_vec)

    def score(text):
        vec = np.array(embedding_model.embed_query(text))
        vec /= np.linalg.norm(vec)
        return float(np.dot(query_vec, vec))

    scored_df = children_df.copy()
    scored_df["similarity"] = scored_df["content"].apply(score)
    return scored_df.sort_values("similarity", ascending=False).head(top_k)


# ---------------------------------------------------------------------------
# Step 5: full hybrid retrieval
# ---------------------------------------------------------------------------

def graph_rag_retrieve_df(query, children_df, graph, embedding_model, top_k=3, hops=1):
    seed_chunks_df = vector_search_df(query, children_df, embedding_model, top_k=top_k)
    seed_child_ids = set(seed_chunks_df["child_id"])

    seed_entities = set()
    for cid in seed_child_ids:
        seed_entities.update(entities_in_chunk(graph, cid))

    expanded_entities = graph_expand(graph, seed_entities, hops=hops)

    graph_child_ids = set()
    for entity in expanded_entities:
        if entity in graph.nodes:
            graph_child_ids.update(graph.nodes[entity].get("mentioned_in", set()))

    all_child_ids = seed_child_ids | graph_child_ids
    context_df = children_df[children_df["child_id"].isin(all_child_ids)].copy()
    context_df["via"] = context_df["child_id"].apply(
        lambda cid: "vector" if cid in seed_child_ids else "graph"
    )

    return {
        "seed_chunks_df": seed_chunks_df,
        "seed_entities": seed_entities,
        "expanded_entities": expanded_entities,
        "context_df": context_df,
    }


# ---------------------------------------------------------------------------
# Step 6: quick evaluation -- did retrieval find each question's own evidence?
# ---------------------------------------------------------------------------

def evaluate_retrieval(questions_df, children_df, graph, embedding_model,
                        top_k=3, hops=1, n_questions=20):
    """
    For each of the first n_questions questions, run retrieval using the
    question text as the query, then check whether the retrieved context_df
    includes at least one chunk that originated from that same question's
    own evidence (source_question_id match) -- a simple, cheap recall proxy.
    """
    results = []
    sample = questions_df.head(n_questions)

    for row in sample.itertuples(index=False):
        result = graph_rag_retrieve_df(
            query=row.question,
            children_df=children_df,
            graph=graph,
            embedding_model=embedding_model,
            top_k=top_k,
            hops=hops,
        )
        retrieved_source_qids = set(result["context_df"]["source_question_id"])
        hit = row.id in retrieved_source_qids

        results.append({
            "id": row.id,
            "question": row.question,
            "question_type": getattr(row, "question_type", None),
            "hit": hit,
            "num_retrieved": len(result["context_df"]),
        })

    eval_df = pd.DataFrame(results)
    print(f"\nRecall proxy: {eval_df['hit'].mean():.2%} "
          f"({eval_df['hit'].sum()}/{len(eval_df)} questions retrieved their own evidence)")
    return eval_df


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[1/4] Loading novel_questions.json...")
    questions_df = load_questions_df()
    print(f"Loaded {len(questions_df)} questions. Columns: {questions_df.columns.tolist()}\n")

    print("[2/4] Building corpus from evidence passages...")
    children_df = build_children_df_from_evidence(questions_df)
    print(f"Built {len(children_df)} evidence-derived chunks.\n")

    print("[3/4] Building graph (empty for now -- plug in real triples_df if you have them)...")
    empty_triples_df = pd.DataFrame(columns=["subject", "relation", "object", "source_chunk"])
    graph = build_graph_from_df(empty_triples_df)
    print(f"Graph: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relations "
          f"(0 expected until you supply real triples).\n")

    print("[4/4] Running retrieval + evaluation on a sample of questions...")
    from langchain_huggingface import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    eval_df = evaluate_retrieval(
        questions_df, children_df, graph, embedding_model,
        top_k=3, hops=1, n_questions=20
    )
    print(eval_df[["id", "question_type", "hit", "num_retrieved"]])
