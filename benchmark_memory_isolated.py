import torch
import torch.nn as nn
import onnxruntime as ort
import resource
import sys
import json
import bitwise_ops
from models import get_model, OptimizedXNORNet, ResidualBinaryLayer
import torch.nn.functional as F

def get_peak_ram():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)

def load_optimized_model():
    model = get_model('xnor', num_classes=10)
    ckpt = torch.load('checkpoints/xnor_kd_cifar10.pth', map_location='cpu', weights_only=True)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def pack_model_weights(model):
    packed_weights = []
    for layer in model.features:
        if isinstance(layer, ResidualBinaryLayer):
            w = layer.conv.weight
            w_bin = torch.where(w >= 0, 1.0, -1.0)
            w_packed = bitwise_ops.pack_tensor(w_bin.float())
            packed_weights.append(w_packed)
    return packed_weights

def bitwise_forward(model, x, packed_weights):
    with torch.no_grad():
        out = F.relu(model.bn1(model.conv1(x)))
        packed_idx = 0
        for layer in model.features:
            if isinstance(layer, ResidualBinaryLayer):
                out_bn = layer.bn(out)
                out_bin = torch.where(out_bn >= 0, 1.0, -1.0)
                out_packed = bitwise_ops.pack_tensor(out_bin)
                w_packed = packed_weights[packed_idx]
                in_channels = layer.conv_in_channels
                conv_out = bitwise_ops.bitwise_conv2d(
                    out_packed, w_packed, in_channels,
                    layer.conv.padding, layer.conv.stride
                )
                conv_out = conv_out * layer.conv.alpha.view(1, -1, 1, 1)
                shortcut_out = layer.shortcut(out)
                if layer.use_pool:
                    shortcut_out = F.max_pool2d(shortcut_out, kernel_size=2, stride=2)
                    conv_out = layer.pool(conv_out)
                out = conv_out + shortcut_out
                packed_idx += 1
            else:
                out = layer(out)
        out = out.view(out.size(0), -1)
        out = model.classifier(out)
        return out

def run():
    target = sys.argv[1]
    batch_size = 128
    inputs = torch.randn(batch_size, 3, 32, 32)
    
    if target == 'sim':
        model = load_optimized_model()
        with torch.no_grad():
            for _ in range(5): _ = model(inputs)
            
    elif target == 'onnx':
        onnx_path = 'checkpoints/xnor_network.onnx'
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(onnx_path, sess_options)
        input_name = session.get_inputs()[0].name
        x_np = inputs.numpy()
        for _ in range(5): _ = session.run(None, {input_name: x_np})
        
    elif target == 'bitwise':
        model = load_optimized_model()
        packed_weights = pack_model_weights(model)
        for layer in model.features:
            if isinstance(layer, ResidualBinaryLayer):
                layer.conv_in_channels = layer.conv.weight.size(1)
                layer.conv.weight = nn.Parameter(torch.empty(0))
        for _ in range(5): _ = bitwise_forward(model, inputs, packed_weights)
        
    print(json.dumps({"peak_mb": get_peak_ram()}))

if __name__ == "__main__":
    run()
