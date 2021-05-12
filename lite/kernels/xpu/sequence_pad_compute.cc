// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/xpu/sequence_pad_compute.h"
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/backends/xpu/target_wrapper.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequencePadCompute::Run() {
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& param = this->template Param<param_t>();

  auto* x = param.X;
  CHECK(!x->lod().empty()) << "Input X should have lod data.";
  auto& x_dims = x->dims();
  int dim = x->numel() / x_dims[0];
  auto x_lod = x->lod();
  const auto& x_lod_0 = x_lod[0];
  int seq_num = x_lod_0.size() - 1;
  int max_seq_len = 0;
  for (int i = 0; i < seq_num; ++i) {
    max_seq_len =
        (std::max)(max_seq_len, static_cast<int>(x_lod_0[i + 1] - x_lod_0[i]));
  }

  auto* pad_value = param.PadValue;
  CHECK_EQ(pad_value->numel(), 1)
      << "The numel of pad_value only be 1 for XPU.";
  float pad_value_data = pad_value->data<float>()[0];

  auto* out = param.Out;
  auto* len_t = param.Length;

  int padded_length = param.padded_length;
  int real_padded_length = padded_length;
  if (real_padded_length == -1) {
    real_padded_length = max_seq_len;
  }

  std::vector<int> x_lod_0_int(x_lod_0.size());
  for (int i = 0; i < x_lod_0.size(); i++) {
    x_lod_0_int[i] = static_cast<int>(x_lod_0[i]);
  }
  XPUScratchPadGuard x_lod_0_guard_ =
      TargetWrapperXPU::MallocScratchPad(x_lod_0_int.size() * sizeof(int));
  XPU_CALL(xpu_memcpy(reinterpret_cast<int*>(x_lod_0_guard_->addr_),
                      x_lod_0_int.data(),
                      x_lod_0_int.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int ret =
      xdnn::sequence_pad(ctx.GetRawContext(),
                         x->data<float>(),
                         out->mutable_data<float>(TARGET(kXPU)),
                         reinterpret_cast<const int*>(x_lod_0_guard_->addr_),
                         max_seq_len,
                         seq_num,
                         dim,
                         pad_value_data);
  CHECK_EQ(ret, 0) << "call xdnn::sequence_pad failed!";

  auto* len_data = len_t->template mutable_data<int64_t>();
  for (size_t i = 1; i < x_lod[0].size(); i++) {
    len_data[i - 1] = x_lod[0][i] - x_lod[0][i - 1];
  }
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pad,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequencePadCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindInput("PadValue",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kFloat))})
    .BindOutput("Length",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
