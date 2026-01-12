from onnx_quantize import QActivationArgs, QConfig, QuantizationStrategy, QuantType, QWeightArgs
from onnx_quantize.qfunctions import get_qfunction


class TestQLinearFactory:
    def test_qlinear_matmul_factory(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qlinear",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QLinearMatMul"

    def test_qlinear_gemm_factory(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qlinear",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QLinearGemm"


class TestQDQFactory:
    def test_qmatmul_qdq_factory_weights_only(self):
        qconfig = QConfig(weights=QWeightArgs(dtype=QuantType.QInt8), format="qdq")

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightsOnlyQDQ"

    def test_qmatmul_qdq_factory_static_input(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightStaticInputQDQ"

    def test_qmatmul_qdq_factory_static_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightStaticOutputQDQ"

    def test_qmatmul_qdq_factory_static_input_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightStaticInputOutputQDQ"

    def test_qmatmul_qdq_factory_dynamic_input(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightDynamicInputQDQ"

    def test_qmatmul_qdq_factory_dynamic_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightDynamicOutputQDQ"

    def test_qmatmul_qdq_factory_dynamic_input_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightDynamicInputOutputQDQ"

    def test_qmatmul_qdq_factory_grouped(self):
        qconfig = QConfig(
            weights=QWeightArgs(
                dtype=QuantType.QInt8, strategy=QuantizationStrategy.GROUP, group_size=32
            ),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightsOnlyGrouped"

    def test_qmatmul_qdq_factory_grouped_4bits(self):
        qconfig = QConfig(
            weights=QWeightArgs(
                dtype=QuantType.QInt4, strategy=QuantizationStrategy.GROUP, group_size=32
            ),
            format="qdq",
        )

        result = get_qfunction("MatMul", qconfig).__name__
        assert result == "QMatMulWeightsOnlyGrouped"

    # QDQ Gemm factory tests
    def test_qgemm_qdq_factory_weights_only(self):
        qconfig = QConfig(weights=QWeightArgs(dtype=QuantType.QInt8), format="qdq")

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightsOnlyQDQ"

    def test_qgemm_qdq_factory_static_input(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightInputQDQ"

    def test_qgemm_qdq_factory_static_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightOutputQDQ"

    def test_qgemm_qdq_factory_static_input_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            output_activations=QActivationArgs(dtype=QuantType.QInt8, is_static=True),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightInputOutputQDQ"

    def test_qgemm_qdq_factory_dynamic_input(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightDynamicInputQDQ"

    def test_qgemm_qdq_factory_dynamic_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightDynamicOutputQDQ"

    def test_qgemm_qdq_factory_dynamic_input_output(self):
        qconfig = QConfig(
            weights=QWeightArgs(dtype=QuantType.QInt8),
            input_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            output_activations=QActivationArgs(dtype=QuantType.QUInt8, is_static=False),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightDynamicInputOutputQDQ"

    def test_qgemm_qdq_factory_grouped(self):
        qconfig = QConfig(
            weights=QWeightArgs(
                dtype=QuantType.QInt8, strategy=QuantizationStrategy.GROUP, group_size=32
            ),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightsOnlyGrouped"

    def test_qgemm_qdq_factory_grouped_4bits(self):
        qconfig = QConfig(
            weights=QWeightArgs(
                dtype=QuantType.QInt4, strategy=QuantizationStrategy.GROUP, group_size=32
            ),
            format="qdq",
        )

        result = get_qfunction("Gemm", qconfig).__name__
        assert result == "QGemmWeightsOnlyGrouped"
