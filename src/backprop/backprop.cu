
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



void TraversePreOrder(Tensor *back_node, float *device_dy, bool from_gradless, bool from_custom, int parent_op)
{
  if(back_node==nullptr)
    return;

  int op=back_node->op;
  std::string tensor_name, param_name, bias_name;
  float *w;
  float *device_dx, *device_dw;
  device_dx=nullptr;
  device_dw=nullptr;
  float dims_prod = back_node->dims_prod;

  

  if(!in_int(op, gradless_ops) && !from_gradless)
  {

    //std::cout << "\nTraversing: " << back_node->name << "/" << back_node->scopeless_name << ", op: " << back_node->op << ", parent_op: " << parent_op << ", leaf: " << back_node->leaf << ", weight: " << back_node->weight << "\n";
    if(device_dy==nullptr && !in_int(op, loss_ops) && !from_custom)
    {
      std::string _err = "dy derivate is null at the backward mode with op "+std::to_string(op);
      LogErrorS(_err);
      return;
    }



    if (back_node->weight) // dw is updated by pointer
      return;
    

    tensor_name = back_node->scopeless_name;
    if (back_node->leaf)
    {
      if (!from_custom)
      {
        if(tensor_name!="")
        {
          if(var_to_grad.count(tensor_name)>0)
          {
            
            float *acc_y = var_to_grad[tensor_name];
            
            int grid_size, block_size, shared_mem_size;
            std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(dims_prod);
            grid_size = grid_block_mem_sizes[0];
            block_size = grid_block_mem_sizes[1];

            
            add_inplace<<<grid_size, block_size>>>(acc_y, device_dy, dims_prod);

            to_pool(dims_prod, acc_y, "dy of leaf");

          } else
            var_to_grad[tensor_name] = device_dy;
        }
        to_pool(dims_prod, device_dy, "dy of leaf");
      }
      //std::cout << "\n\nAccumulating grad of: " << tensor_name << "\n\n\n";
      
      to_pool(dims_prod, back_node->tensor_ptr, "leaf tensor");
      
      to_free_tensor(back_node);
      return;
    }

    from_custom = from_custom || (in_int(op, custom_ops));

    int B, C, OC;
    float x_size, w_size, b_size;

    float *inp, *b, *out, *last_inp;
    float *dinp, *dw, *db, *device_db;
    device_dw=nullptr;
    device_db=nullptr;
    w=nullptr;
    b=nullptr;
    

    
    //std::cout << "Acquire info"  << "\n";

    tensor_name = back_node->L_Node->scopeless_name;

    inp = back_node->L_Node->tensor_ptr;
    x_size = back_node->L_Node->dims_prod;

    out = back_node->tensor_ptr;

    //std::cout << "Check null" << "\n";
    if(back_node->R_Node!=nullptr)
    {
      //std::cout << "not null " << "\n";
      param_name  = back_node->R_Node->name;
      w = back_node->R_Node->tensor_ptr;
      w_size = back_node->R_Node->dims_prod;

      b = back_node->R_Node->b;
      b_size = back_node->R_Node->b_size;
    }


    //std::cout << "malloc device w" << "\n";

    // weight gradient
    if(!in_int(op, loss_ops)&&back_node->R_Node!=nullptr)
    {
      
      int grid_size, block_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(w_size);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];
      
      //std::cout << "Is weight: " << back_node->R_Node->weight << "\n";
      if(back_node->R_Node->weight)
      {
        float *new_grad_ptr;
        if (w!=nullptr&&op!=hadamard_op&&op!=add_op)
        {
          //std::cout << "weight of size " << w_size << "\n";
          if (NamedParamGrads[param_name]==nullptr)
          {
            
            new_grad_ptr = get_from_pool(0, w_size, "weight grad pointer");
            set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(new_grad_ptr, w_size);
            NamedParamGrads[param_name] = new_grad_ptr;
          }
          device_dw = NamedParamGrads[param_name];
        }

        if (b!=nullptr&&op!=hadamard_op&&op!=add_op)
        {
          bias_name = param_name+"_bias";

          if (NamedParamGrads[bias_name]==nullptr)
          {
            int grid_size_b, block_size_b; 
            std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(w_size);
            grid_size_b = grid_block_mem_sizes[0];
            block_size_b = grid_block_mem_sizes[1];
            
            new_grad_ptr = get_from_pool(0, w_size, "bias grad pointer");
            set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(new_grad_ptr, b_size);
            NamedParamGrads[bias_name] = new_grad_ptr;
          }
          device_db = NamedParamGrads[bias_name];
        }
      } else {
      
        if(!in_int(op, weightless_ops) && !from_custom && back_node->R_Node->op != detach_op)
        {
          /*
          if (w_size==4)
          {
            std::cout << "ulululu of op " << std::to_string(op) << "\n";
            std::cout << "" << param_name<< "\n";
            std::cout << "" << tensor_name<< "\n";
          }
          */
          std::string from = "dw of " + std::to_string(op);
          device_dw = get_from_pool(0, w_size, from);
          set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(device_dw, w_size);
        }
      }
    }
    


    // input gradient
    std::string from = "dx of "+ std::to_string(op);
    

    if(op!=add_op && op!=scalar_add_op && !from_custom && op!=lgrad_op && op!=broadcast_lastdim_add_op) {
      int grid_size, block_size; 
      std::vector<int> grid_block_mem_sizes = CalculateGridAndBlockSizes(x_size);
      grid_size = grid_block_mem_sizes[0];
      block_size = grid_block_mem_sizes[1];

      device_dx = get_from_pool(0, x_size, from);

      //TODO: remove this set to zero to improve performance (then, adjust gather op dx to be set to zero)
      set_to_zero_kernel<<<grid_size, block_size, 0, main_stream->stream>>>(device_dx, x_size);
    }
    

    //std::cout << "malloc done"  << "\n";


    B=0;
    C=0;
    OC=0;
    if(back_node->L_Node->dims.size()>0)
    {
      std::vector<float> BC = format_LinearLayer_Dims(back_node->L_Node->dims);
      B  = BC[0];
      C  = BC[1];
    }
    if (w!=nullptr)
      OC = back_node->R_Node->dims[0];
    


    //std::cout << "EXECUTING OP  " << op << "\n";
    switch (op)
    {
      // Simple Leaf Nodes Ops
      case scalar_add_op:
        device_dx = device_dy;
        break;
      case scalar_mult_op:
        scalarmult_backward(device_dx, device_dy, back_node->scalar, x_size); //todo: This one may be wrong
        break;
      case mult_op:
        matmul_backward(inp, w, B, C, OC, device_dx, device_dw, device_dy);
        break;
      case conv2d:
        conv2d_backward(inp, w, device_dx, device_dw, device_dy, param_name);
        break;
      case maxpool2d:
        maxpool2d_backward(inp, out, device_dx, device_dy, back_node->name);
        break;
      case batchnorm2d:
        batchnormd2d_backward(inp, device_dx, device_dw, device_db, device_dy, back_node->name);
        break;
      //case bn2drelu:
      //  bn2drelu_backward(inp, intermediate, out, device_dx, device_dw, device_db, device_dintermediate, device_dy, back_node->name);
      //  break;
      case relu_op:
        relu_backward(inp, x_size, device_dx, device_dy);
        break;
      case cudnn_relu_op:
        cudnn_relu_backward(inp, out, device_dx, device_dy, back_node->name);
        break;
      case gelu_op:
        gelu_backward(inp, x_size, device_dx, device_dy);
        break;
      case sigmoid_op:
        sigmoid_backward(out, x_size, device_dx, device_dy);
        break;
      case tanh_op:
        tanh_backward(out, x_size, device_dx, device_dy);
        break;
      case add_op:
        device_dx = device_dy;
        device_dw = device_dy;
        break;
      case hadamard_op:
        hadamard_backward(inp, w, device_dx, device_dw, device_dy, x_size);
        break;
      case dropout_op:
        dropout_backward(device_dx, w, device_dy, x_size);
        device_dw = device_dy;
        break;
      case gather_last_dim_op:
        gather_last_dim_backward(device_dx, device_dy, back_node);
        device_dw = device_dy;
        break;
      case broadcast_lastdim_add_op:
        device_dx = device_dy;
        broadcast_lastdim_add_backward(device_dw, device_dy, w_size, x_size);
        break;
      case mean_over_semilast_dim_op:
        mean_over_semilast_dim_backward(device_dx, device_dy, back_node);
        break;
      
      // Custom Ops
      case lstm_op:
        lstm_backward(inp, device_dx, device_dy, back_node->scopeless_name);
        break;
      case embedding_op:
        embedding_backward(inp, device_dy, back_node->scopeless_name);
        break;
      case mhsa_op:
        mhsa_backward(inp, device_dx, device_dy, back_node->scopeless_name);
        break;
      case linear_op:
        linear_backward(inp, device_dx, device_dy, back_node->scopeless_name);
        break;

      // Loss Ops
      case cross_entropy_op:
        CrossEntropyBackward(inp, w, B, C, device_dx, back_node->scalar);
        break;
      case cross_entropy_idx_op:
        CrossEntropyIdxBackward(inp, w, B, C, device_dx, back_node->scalar);
        break;
      case mse_op:
        MSEBackward(inp, w, back_node->L_Node->dims_prod, device_dx, back_node->scalar);
        break;
      case mse_is_w_op:
        MSEWithPrioritiesBackward(back_node, device_dx);
        break;

      case lgrad_op:
        device_dx = device_dy;
        break;

      default:
        std::string _error = "The operation "+std::to_string(op)+" does not yet have a backward implementation";
        LogErrorS(_error);
        break;
    }

    //if (ends_with(tensor_name, "ht"))
    //  PrintTensorF(device_dx, 4, 256);
  
  } else
  {
    //std::cout << "\n\nFROM A GRADLESS OP" << "\n\n\n";
    from_gradless = true;
  }

  
  if (in_int(op, loss_ops)||op==lgrad_op)
  {
    to_pool(back_node->R_Node->dims_prod, back_node->R_Node->tensor_ptr, "in loss_ops");
    delete back_node->R_Node;
    back_node->R_Node = nullptr;
  }
  


  // Garbage Collector on all lines below
  TraversePreOrder(back_node->L_Node, device_dx, from_gradless, from_custom, op);
  //from_gradless = (from_gradless || in_int(op, loss_ops));
  TraversePreOrder(back_node->R_Node, device_dw, from_gradless, from_custom, op);
  


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
    //std::cout << "\n\nbackprop:\n\n\n";
    Tensor *back_node = todo_backward_tensors.back();
    todo_backward_tensors.pop_back();

    to_free_tensor(back_node);

    op = back_node->op;
    
    if (op==attribution)
    {
      tensor_name = back_node->name;
      //std::cout << "\n\n\n   backward attribution of " << tensor_name << "\n";
      device_dy = var_to_grad[tensor_name];
      //if (device_dy==nullptr)
      //  std::cout << "propagating null device_dy"  << "\n";
      var_to_grad.erase(tensor_name);
      
      back_node = back_node->R_Node;
    }

    
    TraversePreOrder(back_node, device_dy, false, false, op);
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