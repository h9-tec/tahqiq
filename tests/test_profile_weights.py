from tahqiq.profile import weight_bytes

GB = 10**9


def test_one_copy_of_weights():
    files = [("model.safetensors", 3 * GB), ("model.fp32-00001-of-00002.safetensors", 5 * GB), ("model.fp32-00002-of-00002.safetensors", 1 * GB),
             ("pytorch_model.bin", 3 * GB), ("flax_model.msgpack", 3 * GB)]
    assert weight_bytes(files) == 3 * GB


def test_training_checkpoints_and_duplicates_ignored():
    files = [("model.safetensors", 2 * GB), ("modelx.safetensors", 2 * GB), ("checkpoint-500/model.safetensors", 2 * GB),
             ("checkpoint-500/optimizer.pt", 4 * GB), ("training_args.bin", 1000)]
    assert weight_bytes(files) == 2 * GB


def test_sharded_and_consolidated():
    files = [("consolidated.safetensors", 48 * GB), ("model-00001-of-00002.safetensors", 30 * GB), ("model-00002-of-00002.safetensors", 18 * GB)]
    assert weight_bytes(files) == 48 * GB


def test_mms_language_adapters_and_nemo():
    assert weight_bytes([("model.safetensors", 4 * GB), ("adapter.ara.safetensors", 9 * 10**6), ("adapter.eng.safetensors", 9 * 10**6)]) == 4 * GB
    assert weight_bytes([("stt_ar.nemo", GB // 2)]) == GB // 2
