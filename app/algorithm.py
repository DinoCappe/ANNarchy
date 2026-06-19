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
        self._n_distances = 0   # cumulative distance counter – update in query()

    def fit(self, train: np.ndarray, **index_params) -> None:
        """
        Build your index over the training vectors.

        Parameters
        ----------
        train         : np.ndarray, shape (n_train, dim), dtype float32
        **index_params: arbitrary kwargs from scenarios.yaml -> index_params
                        e.g. ef_construction=200, M=32
        """
        ann.fit(train, **index_params)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        """
        Return the k nearest neighbors for a single query vector.

        Parameters
        ----------
        query         : np.ndarray, shape (dim,), dtype float32
        k             : int -- number of neighbors to retrieve
        **query_params: arbitrary kwargs from scenarios.yaml -> query_params
                        e.g. ef_search=50

        Returns
        -------
        neighbors : np.ndarray, shape (k,), dtype int32
        """
        return ann.query(query, k, **query_params)

    def get_n_distances(self) -> int:
        """
        Return the total number of distance computations performed so far.
        The harness calls this after fit() and again after all queries.
        """
        return ann.total_distances_count()
    
# DO NOT SUBMIT CODE BELOW. ONLY FOR LOCAL TESTING.
if __name__ == "__main__":
    alg = Algorithm()
    print(alg.get_n_distances())
    test = np.random.rand(10, 10).astype(np.float32)
    alg.fit(test, 2)
    print(alg.query(test, 5))