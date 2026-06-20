"""
NNS Competition – Algorithm Template
======================================
Implement your nearest neighbor search algorithm by filling in the three
methods below.  Do not rename the class or the methods.

Interface contract
------------------
fit(train, **index_params)
    Receives all training vectors as a float32 NumPy array of shape
    (n_train, dim), plus any keyword arguments declared under
    index_params in your scenarios.yaml.
    Build your index here.  Return value is ignored.
    A fresh Algorithm() instance is created for every scenario, so
    state never leaks between scenarios.

query(query, k, **query_params)
    Receives a SINGLE query vector as a float32 NumPy array of shape
    (dim,) and the number of neighbors k to retrieve.
    Must return a 1-D integer array of length k with the indices
    (into the training set) of the k nearest neighbors, in any order.
    Called once per query; wall-clock time is measured individually.

get_n_distances() -> int
    Returns the CUMULATIVE number of distance computations performed
    since this instance was created.  The harness calls this
    after all queries have finished.
    You are responsible for incrementing an internal counter inside query().

Rules
-----
* No index construction is allowed inside query().
* No I/O, network access, or subprocess calls during either phase.
"""

import numpy as np
import ANNarchy as ann


class Algorithm:

    def __init__(self):
        self._algo = "hnsw"

    def fit(self, train: np.ndarray, **index_params) -> None:
        self._algo = index_params.pop("algorithm", "hnsw")
        if self._algo == "ivf":
            ann.ivf_fit(train, **index_params)
        else:
            ann.hnsw_fit(train, **index_params)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        if self._algo == "ivf":
            return ann.ivf_query(query, k, **query_params)
        else:
            return ann.hnsw_query(query, k, **query_params)

    def get_n_distances(self) -> int:
        return ann.total_distances_count()


# DO NOT SUBMIT CODE BELOW. ONLY FOR LOCAL TESTING.
if __name__ == "__main__":
    alg = Algorithm()
    test = np.random.rand(100, 10).astype(np.float32)

    print("--- IVF ---")
    alg.fit(test, algorithm="ivf", nlist=5, niter=3)
    print(alg.query(test[0], 5, nprobe=2))
    print("distances:", alg.get_n_distances())

    print("--- HNSW ---")
    alg2 = Algorithm()
    alg2.fit(test, algorithm="hnsw", M=8, ef_construction=20)
    print(alg2.query(test[0], 5, ef_search=20))
    print("distances:", alg2.get_n_distances())
