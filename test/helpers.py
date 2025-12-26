from onnxruntime import InferenceSession


def onnx_forward_on_models(*models, samples, sess_options=None, seed=None):
    assert len(models) > 0

    outputs = []
    for model in models:
        sess = InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"], sess_options=sess_options
        )
        outputs.append(sess.run(None, samples))

    if len(models) == 1:
        return outputs[0]

    return outputs
