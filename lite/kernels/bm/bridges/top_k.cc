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

#include <bmcompiler_if.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"
#include "lite/operators/topk_v2_op.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

using namespace bmcompiler;

int TopkConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  std::cout<<"active topk"<<std::endl;
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);

  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_name = lite::subgraph::bm::UniqueName(op_type);
  auto input_names = op_info->input_names();
  auto output_names = op_info->output_names();
  for(auto& name:input_names){
      std::cout<<name<<std::endl;
  }
  for(auto& name:output_names){
      std::cout<<name<<std::endl;
  }
//  // input
  auto input_name = op_info->Input("X").front();
  auto input_tensor = scope->FindVar(input_name)->GetMutable<lite::Tensor>();
  auto input_dims = input_tensor->dims();
  const int64_t* input_shape_data = const_cast<const int64_t*>(&input_dims.data()[0]);
  std::cout<<"input dims size: "<<input_dims.size()<<std::endl;
  std::vector<int> i_x_shape_data(input_dims.size());
  std::cout<<"in shape: "<<std::endl;
  for (size_t i = 0; i < input_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(input_shape_data[i]);
    std::cout<<i_x_shape_data[i]<<std::endl;
  }

  auto output_name = op_info->Output("Out").front();
  auto output_tensor = scope->FindVar(output_name)->GetMutable<lite::Tensor>();
  auto output_dims = output_tensor->dims();
  const int64_t* output_shape_data = const_cast<const int64_t*>(&output_dims.data()[0]);
  std::cout<<"output dims size: "<<output_dims.size()<<std::endl;
  std::vector<int> out_shape_data(output_dims.size());
  std::cout<<"out shape: "<<std::endl;
  for (size_t i = 0; i < output_dims.size(); i++) {
    out_shape_data[i] = static_cast<int>(output_shape_data[i]);
    std::cout<<out_shape_data[i]<<std::endl;
  }

  auto indices_name = op_info->Output("Indices").front();
  auto indices_tensor = scope->FindVar(indices_name)->GetMutable<lite::Tensor>();
  auto indices_dims = indices_tensor->dims();
  const int64_t* indices_shape_data = const_cast<const int64_t*>(&indices_dims.data()[0]);
  std::cout<<"indices dims size: "<<indices_dims.size()<<std::endl;
  std::vector<int> index_shape_data(indices_dims.size());
  std::cout<<"indices shape: "<<std::endl;
  for (size_t i = 0; i < indices_dims.size(); i++) {
    index_shape_data[i] = static_cast<int>(indices_shape_data[i]);
    std::cout<<index_shape_data[i]<<std::endl;
  }

  std::cout<<"attr : "<<std::endl;
  for(auto& name:op_info->AttrNames()){
      std::cout<<name<<std::endl;
  }
  int k = op_info->GetAttr<int>("k");
  std::cout<<"k: "<<k<<std::endl;
  //  auto k = op_info->GetAttr("k");

  const int* a[2] = {&out_shape_data[0], &index_shape_data[0]};
  const int dimsize[2] = {(int)output_dims.size(), (int)indices_dims.size()};
  const char* outname[2] = {output_name.c_str(), indices_name.c_str()};
  add_topk_layer(graph->GetCompilerHandle(), 
          const_cast<const int*>(&i_x_shape_data[0]), 
          input_dims.size(), static_cast<const char*>(input_name.c_str()), 
          k, input_dims.size()-1, a, dimsize, 
          outname);

//
//
//
//  auto& param = Param<operators::TopkParam>();
//  const float* x_data = param.X->data<float>();
//  float* out_val = param.Out->mutable_data<float>();
//  auto out_ind = param.Indices->mutable_data<int64_t>();
//  DDim x_dims = param.X->dims();
//  int axis = param.axis;
//  int dim_size = x_dims.size();
//  int k = param.K;
//  if (axis < 0) {
//    axis += dim_size;
//  }
//  if (param.k_is_tensor) {
//    k = param.KTensor->data<int>()[0];
//  }
//  int outer_size = x_dims.count(0, axis);
//  int axis_size = x_dims[axis];
//  int inner_size = x_dims.count(axis + 1, dim_size);
//  int sum_size = axis_size * inner_size;
//  int out_sum_size = k * inner_size;
//  for (int n = 0; n < outer_size; n++) {
//    const float* in_data = x_data + n * sum_size;
//    float* out_data = out_val + n * out_sum_size;
//    int64_t* out_ind_data = out_ind + n * out_sum_size;
//    for (int i = 0; i < inner_size; i++) {
//      std::vector<std::pair<float, int>> vec;
//      for (int j = 0; j < axis_size; j++) {
//        vec.push_back(std::make_pair(in_data[j * inner_size + i], j));
//      }
//      std::partial_sort(vec.begin(), vec.begin() + k, vec.end(), comp_func);
//      for (int j = 0; j < k; j++) {
//        out_data[j * inner_size + i] = vec[j].first;
//        out_ind_data[j * inner_size + i] = vec[j].second;
//      }
//    }
//  }
  return SUCCESS;
}

int DummyConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(top_k, kBM,
              paddle::lite::subgraph::bm::TopkConverter);
REGISTER_SUBGRAPH_BRIDGE(matrix_nms, kBM,
              paddle::lite::subgraph::bm::DummyConverter);
