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
#include <unordered_map>
#include <assert.h>

#define FLOAT_PTR(ptr) (reinterpret_cast<float *>(ptr))
#define DATA(params, idx) FLOAT_PTR(copyData(TENSOR(params, idx)).get())

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

using namespace bmcompiler;

#define PARAMS_IN 0
#define PARAMS_OUT 1
//in_out 0:in  1:out

struct Tensor_Info_BM {
  std::string name;
  const Tensor* t;
  size_t dims_size;
  std::vector<int> i_x_shape_data;
};

static std::vector<Tensor_Info_BM> get_tensor_info(int in_out,
                                         const OpInfo *op_info,
                                         Scope* scope,
                                         const std::string& param_name){
    std::vector<std::string> names;
    std::vector<Tensor_Info_BM> vec_ret;
    (in_out==0)?(names = op_info->Input(param_name.c_str())):(names = op_info->Output(param_name.c_str()));
    for(auto& name: names){
      Tensor_Info_BM tIB;
      std::cout<<name<<std::endl;
      tIB.name = name;
      auto x = scope->FindTensor(name);
      
      auto x_dims = x->dims();
      tIB.dims_size = x_dims.size();
      const int64_t* input_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
      tIB.i_x_shape_data.resize(x_dims.size());
      for (size_t i = 0; i < x_dims.size(); i++) {
        tIB.i_x_shape_data[i] = static_cast<int>(input_shape_data[i]);
        std::cout<<tIB.i_x_shape_data[i]<<std::endl;
      }
      tIB.t = x;
      vec_ret.push_back(tIB);
    }
    return vec_ret;
}

static inline void trans(const float *src, float *dst, int X, int H) {
      for (int i = 0; i < 4; ++i) {
          for (int h = 0; h < H; ++h)
             for (int x = 0; x < X; ++x)
                 dst[x * H + h] = src[h * X + x];
        src += X * H;
        dst += X * H;
     }
 }


int RnnConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  std::cout<<"active rnn"<<std::endl;
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
  std::cout<<"pre state: "<<std::endl;
  auto pre_state = get_tensor_info(PARAMS_IN,op_info,scope,"PreState");
  std::cout<<"weight list: "<<std::endl;
  auto weight_list = get_tensor_info(PARAMS_IN,op_info,scope,"WeightList");

  std::cout<<"out: "<<std::endl;
  auto out_param = get_tensor_info(PARAMS_OUT,op_info,scope,"Out");
  std::cout<<"dropout state: "<<std::endl;
  get_tensor_info(PARAMS_OUT,op_info,scope,"DropoutState");
  std::cout<<"reserve: "<<std::endl;
  get_tensor_info(PARAMS_OUT,op_info,scope,"Reserve");
  std::cout<<"state: "<<std::endl;
  auto out_state_param = get_tensor_info(PARAMS_OUT,op_info,scope,"State");


  auto dropout_prob = op_info->GetAttr<float>("dropout_prob");
  std::cout<<"dropout_prob: "<<dropout_prob<<std::endl;
  auto hidden_size = op_info->GetAttr<int>("hidden_size");
  std::cout<<"hidden_size: "<<hidden_size<<std::endl;
  auto input_size = op_info->GetAttr<int>("input_size");
  std::cout<<"input_size: "<<input_size<<std::endl;
  auto is_bidirec = op_info->GetAttr<bool>("is_bidirec");
  std::cout<<"is_bidirec: "<<is_bidirec<<std::endl;
  auto is_test = op_info->GetAttr<bool>("is_test");
  std::cout<<"is_test: "<<is_test<<std::endl;
  auto mode = op_info->GetAttr<std::string>("mode");
  std::cout<<"mode: "<<mode<<std::endl;
  auto num_layers = op_info->GetAttr<int>("num_layers");
  std::cout<<"num_layers: "<<num_layers<<std::endl;
  // auto op_role = op_info->GetAttr<int>("op_role");
  // std::cout<<"op_role: "<<op_role<<std::endl;

  int D = is_bidirec?2:1;
  int H = hidden_size;
  int X = input_size;
  bool has_bias = true;
  float *weight = new float[D * 4 * H * (X + H) +
  D * 4 * H * (D * H + H) * (num_layers - 1)];
  float *bias = has_bias ?  new float[D * 4 * H * 2 * num_layers] : nullptr;

  auto weight_len = weight_list.size();
  assert(static_cast<int>(weight_len) == (has_bias ? 2 : 1) * (is_bidirec ? 2 : 1) * 2 * num_layers);

  float * witer = weight;
  
  for(int i=0;i<weight_len/(has_bias ? 2 : 1);i++){
    auto x = weight_list[i].t;
    auto x_dims = x->dims();
    const int64_t* input_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
    std::vector<int> i_x_shape_data(x_dims.size());
    auto data = x->data<float>();
    // unsigned long long tensor_size = 1;
    for (size_t i = 0; i < x_dims.size(); i++) {
      i_x_shape_data[i] = static_cast<int>(input_shape_data[i]);
      std::cout<<i_x_shape_data[i]<<" : "<<std::endl;
      //tensor_size *= i_x_shape_data[i];
    }
    
    //float* new_weight = new float[4*hidden_size * i_x_shape_data[1]];
    //memcpy(new_weight, data + hidden_size*i_x_shape_data[1], sizeof(float)*2*hidden_size*i_x_shape_data[1]);
    //memcpy(new_weight + 2*hidden_size*i_x_shape_data[1], data, sizeof(float)*hidden_size * i_x_shape_data[1]);
    //memcpy(new_weight + 3*hidden_size*i_x_shape_data[1], data + 3*hidden_size*i_x_shape_data[1], sizeof(float)*hidden_size*i_x_shape_data[1]);
    //memcpy(new_weight, data, sizeof(float) * 2 * hidden_size * i_x_shape_data[1]);
    //memcpy(new_weight + 2*hidden_size*i_x_shape_data[1], data + 3*hidden_size*i_x_shape_data[1], sizeof(float)*hidden_size*i_x_shape_data[1]);
    //memcpy(new_weight+3*hidden_size*i_x_shape_data[1], data+2*hidden_size*i_x_shape_data[1], sizeof(float)*hidden_size*i_x_shape_data[1]);
    //trans(new_weight, witer, i_x_shape_data[1], hidden_size);

    trans(data, witer, i_x_shape_data[1], hidden_size);
    witer += 4*hidden_size*i_x_shape_data[1];
    //delete [] new_weight;
  }

  float * biter = bias;
  for(int i=num_layers*(has_bias ? 2 : 1)*D;i<weight_len;i++){
    auto x = weight_list[i].t;
    auto x_dims = x->dims();
    auto data = x->data<float>();

    //float* new_bias = new float[4*hidden_size];
    //memcpy(new_bias, data + hidden_size, sizeof(float)*2*hidden_size);
    //memcpy(new_bias + 2*hidden_size, data, sizeof(float)*hidden_size);
    //memcpy(new_bias+3*hidden_size, data+3*hidden_size, sizeof(float)*hidden_size);
    //memcpy(new_bias, data, 2*hidden_size*sizeof(float));
    //memcpy(new_bias+2*hidden_size, data+3*hidden_size, sizeof(float)*hidden_size);
    //memcpy(new_bias+3*hidden_size, data+2*hidden_size, sizeof(float)*hidden_size);
    //memcpy(biter,new_bias,4*hidden_size*sizeof(float));

    memcpy(biter,data,4*hidden_size*sizeof(float));
    biter += 4*hidden_size;
    //delete [] new_bias;
  }

#if 0
  for(auto& weight:weight_list){
    auto x = weight.t;
    auto x_dims = x->dims();
    const int64_t* input_shape_data = const_cast<const int64_t*>(&x_dims.data()[0]);
    std::vector<int> i_x_shape_data(x_dims.size());
    //auto data = x->data<float>();
    unsigned long long tensor_size = 1;
    for (size_t i = 0; i < x_dims.size(); i++) {
      i_x_shape_data[i] = static_cast<int>(input_shape_data[i]);
      std::cout<<i_x_shape_data[i]<<" : "<<std::endl;
      tensor_size *= i_x_shape_data[i];
    }

    // for(int i=0;i<tensor_size;i++){
    //   std::cout<<" ~~"<<data[i];
    // }

    // std::cout<<std::endl;
  }
#endif
  std::string node_name = "bitmain_happy_sky";
  add_pytorch_lstm_layer(graph->GetCompilerHandle(),
                      const_cast<const int*>(&i_x_shape_data[0]),x_dims.size(),x_name.c_str(),
                      const_cast<const int*>(&pre_state[0].i_x_shape_data[0]),pre_state[0].dims_size,pre_state[0].name.c_str(),
                      const_cast<const int*>(&pre_state[1].i_x_shape_data[0]),pre_state[1].dims_size,pre_state[1].name.c_str(),
                      const_cast<const int*>(&out_param[0].i_x_shape_data[0]),out_param[0].dims_size,out_param[0].name.c_str(),
                      const_cast<const int*>(&out_state_param[0].i_x_shape_data[0]),out_state_param[0].dims_size,out_state_param[0].name.c_str(),
                      const_cast<const int*>(&out_state_param[1].i_x_shape_data[0]),out_state_param[1].dims_size,out_state_param[1].name.c_str(),
                      weight,bias,is_bidirec,false,num_layers,
                      node_name.c_str());

  delete [] weight;
  weight = nullptr;
  if(bias!=nullptr){
    delete [] bias;
    bias = nullptr;
  }




  std::cout<<"rnn final out: "<<op_info->Output("Out").front()<<std::endl;

  graph->AddNode(op_info->Output("Out").front());
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(rnn, kBM,
              paddle::lite::subgraph::bm::RnnConverter);
//REGISTER_SUBGRAPH_BRIDGE(fill_constant_batch_size_like, kBM,
//              paddle::lite::subgraph::bm::DummyConverter);
