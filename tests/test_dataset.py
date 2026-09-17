from genrec.data import iterative_k_core, make_synthetic_dataset

def test_iterative_k_core_removes_cascading_low_degree_nodes():
    rows = [("u1", "a", 1, 1)] * 5 + [("u1", "b", 2, 2)] * 4 + [("u2", "b", 3, 3)]
    filtered, _ = iterative_k_core(rows, 5, 5)
    assert len(filtered) == 5
    assert {row[1] for row in filtered} == {"a"}

def test_synthetic_split_has_leave_two_out_examples():
    data = make_synthetic_dataset(users=3, length=5)
    assert all(len(sequence) == 3 for sequence in data.train_sequences)
    assert len(data.valid_examples) == len(data.test_examples) == 3
