import code.common.constants as C
import code.fields.harness as harness_fields
import code.fields.loadgen as loadgen_fields
import code.fields.models as model_fields
import code.resnet50.tensorrt.fields as rn50_fields
from nvmitten.constants import Precision

EXPORTS = {
    C.WorkloadSetting(C.HarnessType.Custom, C.AccuracyTarget(0.99), C.PowerSetting.MaxP): {
        model_fields.gpu_batch_size: {
            "resnet50": 256,
        },
        model_fields.precision: Precision.INT8,
        model_fields.input_dtype: Precision.INT8,
        model_fields.input_format: "linear",
        harness_fields.tensor_path: "build/preprocessed_data/imagenet/ResNet50/int8_linear",
        harness_fields.map_path: "data_maps/imagenet/val_map.txt",
        harness_fields.gpu_copy_streams: 4,
        harness_fields.gpu_inference_streams: 4,
        harness_fields.use_graphs: True,
        harness_fields.warmup_duration: 5.0,
        loadgen_fields.performance_sample_count: 1024,
        loadgen_fields.offline_expected_qps: 60000,
    },
}