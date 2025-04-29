
#include "../common/include.h"
#include "../compiler_frontend/logging.h"
#include "../cuda_kernels/include.h"
#include "../mangler/scope_struct.h"
#include "../mma/general.h"
#include "../networks/modules/include.h"
#include "../tensor/tensor_struct.h"
#include "include.h"

std::vector<Tensor *> todo_backward_tensors;
std::map<std::string, float *> NamedParamGrads;

std::map<std::string, std::function<void(float *, float, float *, float *, float *, std::string)>> backward_functions;




inline void HandleLeafGradient(Tensor *back_node, float *device_dy, std::string tensor_name, bool from_custom) {
  float dims_prod = back_node->dims_prod;
  if (!from_custom)
  {
    if(tensor_name!="")
    {
      if(var_to_grad.count(tensor_name)>0)
      {   
        float *acc_y = var_to_grad[tensor_name];
        cpp_tensor_tensor_add(acc_y, device_dy, dims_prod);
        to_pool(dims_prod, acc_y, "dy of leaf");
      } else
        var_to_grad[tensor_name] = device_dy;
    }
    to_pool(dims_prod, device_dy, "dy of leaf");
  }
  
  to_pool(dims_prod, back_node->tensor_ptr, "leaf tensor"); 
  // std::cout << "Adding " << tensor_name << ".\n";
  if (!back_node->is_last_version)
    to_free_tensor(back_node);
}




inline void Acquire_Simple_Derivative(float *&d_ptr, float size, int op, bool from_custom, std::string parent) {
  if (op==add_op)
    return;
  // std::string from = "dx of "+ std::to_string(op);

  std::string from = "dx of " + parent;
 
  int grid_size, block_size; 
  CalculateGridAndBlockSizes(size, grid_size, block_size);

  d_ptr = get_from_pool(0, size, from);
  //TODO: remove this set to zero to improve performance (then, adjust gather op dx to be set to zero)
  set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(d_ptr, size);
}


inline void Acquire_Weight_Gradient(float *&d_ptr, float size, std::string param_name, int op, bool from_custom) {
  if (op==hadamard_op||op==add_op)
    return;

  int grid_size, block_size; 
  CalculateGridAndBlockSizes(size, grid_size, block_size);
  
  if (NamedParamGrads[param_name]==nullptr)
  {
    float *new_grad_ptr;
    
    new_grad_ptr = get_from_pool(0, size, "weight grad pointer");
    set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(new_grad_ptr, size);
    NamedParamGrads[param_name] = new_grad_ptr;
  }
  d_ptr = NamedParamGrads[param_name];
}



inline void Alloc_Child_Nodes_Derivatives(Tensor* back_node, float*& d_lhs, float*& d_rhs, size_t lhs_size, size_t rhs_size, int op, bool from_custom) {

  if (back_node->L_Node)
  {
    if (!back_node->L_Node->weight)
    {
      if(op!=add_op && op!=scalar_add_op && !from_custom && op!=broadcast_lastdim_add_op && back_node->L_Node->op != detach_op)
      {
        std::string parent = back_node->scopeless_name;
        Acquire_Simple_Derivative(d_lhs, lhs_size, op, from_custom, parent);
      }
    }
    else
      Acquire_Weight_Gradient(d_lhs, lhs_size, back_node->L_Node->name, op, from_custom);
  }


  if(back_node->R_Node!=nullptr&&!in_int(op, loss_ops))
  {
    if (!back_node->R_Node->weight)
    {
      if(!in_int(op, weightless_ops) && !from_custom && back_node->R_Node->op != detach_op && op!=add_op)
        Acquire_Simple_Derivative(d_rhs, rhs_size, op, from_custom, "rhs");
    }
    else  
      Acquire_Weight_Gradient(d_rhs, rhs_size, back_node->R_Node->name, op, from_custom);
  }
}




void TraversePreOrder(Tensor *back_node, float *device_dy, bool from_custom, int parent_op)
{
  if(back_node==nullptr)
    return;

  int op=back_node->op;
  std::string tensor_name, param_name, bias_name;
  float *rhs, *d_lhs, *d_rhs;
  d_lhs=nullptr;
  d_rhs=nullptr;
  float dims_prod = back_node->dims_prod;

  

  if(!in_int(op, gradless_ops))
  {

    //std::cout << "\nTraversing: " << back_node->name << "/" << back_node->scopeless_name << ", op: " << back_node->op << ", parent_op: " << parent_op << ", leaf: " << back_node->leaf << ", weight: " << back_node->weight << "\n";
    if(device_dy==nullptr && !in_int(op, loss_ops) && !from_custom)
    {
    
      tensor_name = back_node->scopeless_name;
      CleanTree_Backprop(back_node);
      std::string _err = "dy derivate is null at the backward mode with op "+std::to_string(op) + " for tensor " + tensor_name;
      // LogError(_err);
      return;
    }

    if (back_node->weight) // dw is updated by pointer
      return;
    
    tensor_name = back_node->scopeless_name;
    if (back_node->leaf)
    {
      HandleLeafGradient(back_node, device_dy, tensor_name, from_custom);
      return;
    }

    from_custom = from_custom || (in_int(op, custom_ops));




    float lhs_size, rhs_size;
    float *lhs, *out;
    d_rhs=nullptr;
    rhs=nullptr;
    

    

    tensor_name = back_node->L_Node->scopeless_name;
    lhs = back_node->L_Node->tensor_ptr;
    out = back_node->tensor_ptr;
    lhs_size = back_node->L_Node->dims_prod;

    if(back_node->R_Node!=nullptr)
    {
      param_name  = back_node->R_Node->name;
      rhs = back_node->R_Node->tensor_ptr;
      rhs_size = back_node->R_Node->dims_prod;
    }


 
    
    Alloc_Child_Nodes_Derivatives(back_node, d_lhs, d_rhs, lhs_size, rhs_size, op, from_custom);
  




    switch (op)
    {
      // Simple Leaf Nodes Ops
      case scalar_add_op:
        d_lhs = device_dy;
        break;
      case scalar_mult_op:
        scalarmult_backward(d_lhs, device_dy, back_node->scalar, lhs_size); //todo: This one may be wrong
        break;
      case mult_op:
        matmul_backward(back_node->L_Node, back_node->R_Node, d_lhs, d_rhs, device_dy);
        break;
      case add_op:
        d_lhs = device_dy;
        d_rhs = device_dy;
        break;
      case hadamard_op:
        hadamard_backward(lhs, rhs, d_lhs, d_rhs, device_dy, lhs_size);
        break;
      case dropout_op:
        dropout_backward(d_lhs, rhs, device_dy, lhs_size);
        d_rhs = device_dy;
        break;
      case gather_last_dim_op:
        gather_last_dim_backward(d_lhs, device_dy, back_node);
        d_rhs = device_dy;
        break;
      case broadcast_lastdim_add_op:
        d_lhs = device_dy;
        broadcast_lastdim_add_backward(d_rhs, device_dy, rhs_size, lhs_size);
        break;
      case mean_over_semilast_dim_op:
        mean_over_semilast_dim_backward(d_lhs, device_dy, back_node);
        break;
      
      // Custom Ops
      case lstm_op:
        lstm_backward(lhs, d_lhs, device_dy, back_node->scopeless_name);
        break;
      case embedding_op:
        embedding_backward(lhs, device_dy, back_node->scopeless_name);
        break;
      case mhsa_op:
        mhsa_backward(lhs, d_lhs, device_dy, back_node->scopeless_name);
        break;

      // Loss Ops
      case cross_entropy_op:
        CrossEntropyBackward(back_node->L_Node, back_node->R_Node, d_lhs, back_node->scalar);
        break;
      case cross_entropy_idx_op:
        CrossEntropyIdxBackward(back_node->L_Node, back_node->R_Node, d_lhs, back_node->scalar);
        break;
      case mse_op:
        MSEBackward(lhs, rhs, back_node->L_Node->dims_prod, d_lhs, back_node->scalar);
        break;
      case mse_is_w_op:
        MSEWithPrioritiesBackward(back_node, d_lhs);
        break;

      case custom_op:
        backward_functions[back_node->operation](lhs, lhs_size, out, d_lhs, device_dy, back_node->scopeless_name);
        break;

      default:
        std::string _error = "The operation "+std::to_string(op)+" does not yet have a backward implementation";
        LogErrorS(_error);
        break;
    }
  
  } else
  {
    //std::cout << "\n\nFROM A GRADLESS OP" << "\n\n\n";
    CleanTree_Backprop(back_node);
  }

  
  // if (in_int(op, loss_ops))
  // {
  //   to_pool(back_node->R_Node->dims_prod, back_node->R_Node->tensor_ptr, "in loss_ops");
  //   delete back_node->R_Node;
  //   back_node->R_Node = nullptr;
  // }
  

  TraversePreOrder(back_node->L_Node, d_lhs, from_custom, op);
  TraversePreOrder(back_node->R_Node, d_rhs, from_custom, op);

  
  // Garbage Collector
  if (back_node->Sparse_Idx_Tensor!=nullptr)
    save_from_pool(back_node->Sparse_Idx_Tensor);
  
  if(!in_int(op, loss_ops) && back_node->tensor_ptr!=nullptr) //loss op has leaves only
    to_pool(dims_prod, back_node->tensor_ptr, "op tensor");

  std::string _op = "dy of operation " + std::to_string(op) + " from parent op " + std::to_string(parent_op) + " and parameter " + param_name;  
  if(device_dy!=nullptr)
    to_pool(dims_prod, device_dy, _op);

  if (!back_node->weight)
    to_free_tensor(back_node);
}



extern "C" float backprop(Scope_Struct *scope_struct)
{

  int op; 
  std::string tensor_name;
  float *device_dy=nullptr;



  while(todo_backward_tensors.size()>0)
  {
    Tensor *back_node = todo_backward_tensors.back();
    todo_backward_tensors.pop_back();

    to_free_tensor(back_node);

    op = back_node->op;
    
    if (op==attribution)
    {
      tensor_name = back_node->name;
      //std::cout << "\n\n\n   backward attribution of " << tensor_name << "\n";
      device_dy = var_to_grad[tensor_name];
      var_to_grad.erase(tensor_name);
      
      back_node = back_node->R_Node;
    }
  
    TraversePreOrder(back_node, device_dy, false, op);
  }





  for(Tensor *tensor : backprop_Tensors_to_save) // e.g: sparse idx tensors
  { 
    backprop_Tensors_to_free.erase(std::remove(backprop_Tensors_to_free.begin(), backprop_Tensors_to_free.end(), tensor), backprop_Tensors_to_free.end()); 
    for(std::tuple<float, float *, std::string> pair : backprop_tensors_to_pool)
    {
      float *tensor_ptr = std::get<1>(pair);
      if (tensor->tensor_ptr == tensor_ptr)
      {
        //std::cout << "Remove " << tensor->name << "/" << tensor->scopeless_name << " from pool.\n";
        backprop_tensors_to_pool.erase(std::remove(backprop_tensors_to_pool.begin(), backprop_tensors_to_pool.end(), pair), backprop_tensors_to_pool.end());
        break;
      }
    } 
  }


  for(Tensor *tensor : backprop_Tensors_to_free)
    delete tensor;

  for(std::tuple<float, float *, std::string> pair : backprop_tensors_to_pool)
  {
    move_to_pool(0, std::get<0>(pair), std::get<1>(pair), std::get<2>(pair));
    //move_to_pool(0, pair.first, pair.second);
  }

  backprop_Tensors_to_save.clear();
  backprop_Tensors_to_free.clear();
  backprop_tensors_to_pool.clear();
  tensors_sent_to_pool.clear();
  var_to_grad.clear();
  return 0;
}