#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#if defined(__aarch64__)
#include <arm_neon.h> // ARM NEON intrinsics
#endif
#include <torch/extension.h>
#include <vector>

// Pack binary tensor into uint64_t
// Layout: [N, H, W, PackedC]
torch::Tensor pack_tensor(torch::Tensor input) {
  auto options =
      torch::TensorOptions().dtype(torch::kInt64).device(input.device());
  int batch_size = input.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);

  int packed_channels = (channels + 63) / 64;

  auto packed =
      torch::zeros({batch_size, height, width, packed_channels}, options);

  auto input_ptr = input.data_ptr<float>();
  auto packed_ptr = packed.data_ptr<int64_t>();

  at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
    for (int b = start; b < end; ++b) {
      for (int c = 0; c < channels; ++c) {
        int pc = c / 64;
        int bit = c % 64;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            float val = input_ptr[b * channels * height * width +
                                  c * height * width + h * width + w];
            if (val > 0) {
              // Packed Layout: [B, H, W, PC]
              packed_ptr[b * height * width * packed_channels +
                         h * width * packed_channels + w * packed_channels +
                         pc] |= (1ULL << bit);
            }
          }
        }
      }
    }
  });
  return packed;
}

// Bitwise XNOR Convolution
torch::Tensor bitwise_conv2d(torch::Tensor input_packed,
                             torch::Tensor weight_packed, int real_in_channels,
                             int padding, int stride) {
  int batch_size = input_packed.size(0);
  int in_h = input_packed.size(1);
  int in_w = input_packed.size(2);
  int in_pc = input_packed.size(3);

  int out_channels = weight_packed.size(0);
  int k_h = weight_packed.size(1);
  int k_w = weight_packed.size(2);

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

  // Parallelize over Batch
  at::parallel_for(0, batch_size, 1, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; ++b) {

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

                // Pointers
                const int64_t *in_row_ptr = in_ptr + (b * in_h * in_w * in_pc) +
                                            (ih * in_w * in_pc) + (iw * in_pc);
                const int64_t *w_row_ptr = w_ptr + (oc * k_h * k_w * in_pc) +
                                           (kh * k_w * in_pc) + (kw * in_pc);

                int pc = 0;

#if defined(__aarch64__)
                // SIMD Loop: Process 2 int64 (128 bits) at a time (ARM64 Only)
                for (; pc <= in_pc - 2; pc += 2) {
                  // Load 128-bit vector as bytes directly
                  uint8x16_t v_a = vld1q_u8((const uint8_t *)&in_row_ptr[pc]);
                  uint8x16_t v_b = vld1q_u8((const uint8_t *)&w_row_ptr[pc]);

                  // XOR
                  uint8x16_t xor_v = veorq_u8(v_a, v_b);
                  // NOT (~XOR = XNOR)
                  uint8x16_t xnor_v = vmvnq_u8(xor_v);

                  // Count bits
                  uint8x16_t counts = vcntq_u8(xnor_v);

                  // Sum counts
                  uint16x8_t sum16 = vpaddlq_u8(counts);
                  uint32x4_t sum32 = vpaddlq_u16(sum16);
                  acc += vaddvq_u32(sum32);

                  bits_processed += 128;
                }
#endif

                // Remainder Handling (Scalar) - Acts as full loop for non-ARM
                for (; pc < in_pc - 1; ++pc) {
                  int64_t a = in_row_ptr[pc];
                  int64_t b_val = w_row_ptr[pc];
                  acc += __builtin_popcountll(~(a ^ b_val));
                  bits_processed += 64;
                }

                // Last packed channel (always scalar)
                if (pc < in_pc) {
                  int64_t a = in_row_ptr[pc];
                  int64_t b_val = w_row_ptr[pc];
                  acc += __builtin_popcountll(~(a ^ b_val) & last_pc_mask);
                  bits_processed += last_pc_bits;
                }
              } // kw
            } // kh

            out_ptr[b * out_channels * out_h * out_w + oc * out_h * out_w +
                    oh * out_w + ow] = (float)(acc * 2 - bits_processed);
          } // ow
        } // oh
      } // oc
    } // b
  }); // parallel_for

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_tensor", &pack_tensor, "Pack float tensor");
  m.def("bitwise_conv2d", &bitwise_conv2d, "Bitwise convolution 2D");
}
