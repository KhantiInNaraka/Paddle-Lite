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

#include <bmcompiler_defs.h>
#include <bmcompiler_if.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

using namespace bmcompiler;

int FillConstantBatchSizeLikeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  std::cout<<"active fill_constant_batch_size_like"<<std::endl;
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("Input").front();
  auto x = scope->FindTensor(x_name);
  auto x_dims = x->dims();
  const int64_t* input_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
  std::vector<int> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(input_shape_data[i]);
    std::cout<<i_x_shape_data[i]<<std::endl;
  }


  auto out_name = op_info->Output("Out").front();
  auto out = scope->FindTensor(out_name);
  auto out_shape = out->dims().Vectorize();
  std::cout<<"out shape : "<<std::endl;
  for(auto& o:out_shape){
    std::cout<<" "<<o;
  }
  std::cout<<std::endl;

  std::vector<int> ouput_shape_data(out->dims().size());
  for (size_t i = 0; i < out->dims().size(); i++) {
    ouput_shape_data[i] = static_cast<int>(out_shape[i]);
    //if(i==1) ouput_shape_data[i]=1;
    std::cout<<ouput_shape_data[i]<<std::endl;
  }

  auto shape_param = op_info->GetAttr<std::vector<int>>("shape");
  for(auto& o:shape_param){
    std::cout<<" "<<o;
  }
  std::cout<<std::endl;

  auto value = op_info->GetAttr<float>("value");
  std::cout<<"value: "<<value<<std::endl;


  std::string str_trans0 = x_name+"@trans0";
  
  add_reshape_layer_v2(graph->GetCompilerHandle(),static_cast<const char*>(x_name.c_str()),
    const_cast<const int*>(&i_x_shape_data[0]),x_dims.size(),
    static_cast<const char*>(str_trans0.c_str()),const_cast<const int*>(&ouput_shape_data[0]),
    out->dims().size());

 
  std::string str_trans1 = x_name+"@trans1";
  add_shape_ref_layer(graph->GetCompilerHandle(),static_cast<const char*>(str_trans0.c_str()) ,
          const_cast<const int*>(&ouput_shape_data[0]), out->dims().size(), static_cast<const char*>(str_trans1.c_str()));



  add_constant_fill_layer_v2(graph->GetCompilerHandle(), static_cast<const char*>(str_trans1.c_str()),
          static_cast<const char*>(out_name.c_str()),static_cast<const void*>(&value),DTYPE_FP32);

  graph->AddNode(out_name);


  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fill_constant_batch_size_like, kBM,
              paddle::lite::subgraph::bm::FillConstantBatchSizeLikeConverter);
