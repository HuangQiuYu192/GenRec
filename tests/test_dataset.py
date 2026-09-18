from genrec.data import from_sequences, make_synthetic_dataset, read_sequences, write_sequences

def test_portable_sequence_format_round_trip(tmp_path):
    sequences = {"user_a": ["item_1", "item_2", "item_3"]}
    path = tmp_path / "interactions.txt"; write_sequences(path, sequences)
    assert read_sequences(path) == sequences
    assert from_sequences(sequences, "test").num_items == 4

def test_synthetic_split_has_leave_two_out_examples():
    data = make_synthetic_dataset(users=3, length=5)
    assert all(len(sequence) == 3 for sequence in data.train_sequences)
    assert len(data.valid_examples) == len(data.test_examples) == 3
