#include <torch/extension.h>
#include <vector>

// Pack binary tensor into uint64_t along the channel dimension
// Assumes input values are -1 or 1
torch::Tensor pack_tensor(torch::Tensor input) {
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(input.device());
  int batch_size = input.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  int packed_channels = (channels + 63) / 64;
  auto packed =
      torch::zeros({batch_size, packed_channels, height, width}, options);

  auto input_ptr = input.data_ptr<float>();
  auto packed_ptr = packed.data_ptr<int64_t>();

  for (int b = 0; b < batch_size; ++b) {
    for (int c = 0; c < channels; ++c) {
      int pc = c / 64;
      int bit = c % 64;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          float val = input_ptr[b * channels * height * width +
                                c * height * width + h * width + w];
          if (val > 0) {
            packed_ptr[b * packed_channels * height * width +
                       pc * height * width + h * width + w] |= (1ULL << bit);
          }
        }
      }
    }
  }
  return packed;
}

// Bitwise XNOR Convolution
torch::Tensor bitwise_conv2d(torch::Tensor input_packed,
                             torch::Tensor weight_packed, int real_in_channels,
                             int padding, int stride) {
  int batch_size = input_packed.size(0);
  int in_pc = input_packed.size(1);
  int in_h = input_packed.size(2);
  int in_w = input_packed.size(3);

  int out_channels = weight_packed.size(0);
  int k_h = weight_packed.size(2);
  int k_w = weight_packed.size(3);

  int out_h = (in_h + 2 * padding - k_h) / stride + 1;
  int out_w = (in_w + 2 * padding - k_w) / stride + 1;

  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat32)
                     .device(input_packed.device());
  auto output = torch::zeros({batch_size, out_channels, out_h, out_w}, options);

  auto in_ptr = input_packed.data_ptr<int64_t>();
  auto w_ptr = weight_packed.data_ptr<int64_t>();
  auto out_ptr = output.data_ptr<float>();

  uint64_t last_pc_mask = (real_in_channels % 64 == 0)
                              ? ~0ULL
                              : (1ULL << (real_in_channels % 64)) - 1;
  int last_pc_bits =
      (real_in_channels % 64 == 0) ? 64 : (real_in_channels % 64);

  for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int64_t acc = 0;
          int64_t bits_processed = 0;

          for (int kh = 0; kh < k_h; ++kh) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= in_h)
              continue;

            for (int kw = 0; kw < k_w; ++kw) {
              int iw = ow * stride - padding + kw;
              if (iw < 0 || iw >= in_w)
                continue;

              // Full packed channels
              int pc = 0;
              for (; pc < in_pc - 1; ++pc) {
                int64_t a = in_ptr[b * in_pc * in_h * in_w + pc * in_h * in_w +
                                   ih * in_w + iw];
                int64_t b_val = w_ptr[oc * in_pc * k_h * k_w + pc * k_h * k_w +
                                      kh * k_w + kw];
                acc += __builtin_popcountll(~(a ^ b_val));
                bits_processed += 64;
              }
              // Last packed channel
              int64_t a = in_ptr[b * in_pc * in_h * in_w + pc * in_h * in_w +
                                 ih * in_w + iw];
              int64_t b_val = w_ptr[oc * in_pc * k_h * k_w + pc * k_h * k_w +
                                    kh * k_w + kw];
              acc += __builtin_popcountll(~(a ^ b_val) & last_pc_mask);
              bits_processed += last_pc_bits;
            }
          }
          out_ptr[b * out_channels * out_h * out_w + oc * out_h * out_w +
                  oh * out_w + ow] = (float)(acc * 2 - bits_processed);
        }
      }
    }
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_tensor", &pack_tensor, "Pack float tensor into int64 tensor");
  m.def("bitwise_conv2d", &bitwise_conv2d, "Bitwise convolution 2D");
}
