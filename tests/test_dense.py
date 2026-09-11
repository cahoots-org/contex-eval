from contexeval.retrievers.dense import DenseRetriever

CORPUS = [
    {"para_id": "cats", "title": "Cats", "text": "Cats are small feline animals kept as pets."},
    {"para_id": "cars", "title": "Cars", "text": "Cars are motor vehicles for road transport."},
]

def test_dense_semantic_match():
    r = DenseRetriever(CORPUS, k=1).retrieve("a pet kitten")
    assert r.para_ids == ["cats"]
