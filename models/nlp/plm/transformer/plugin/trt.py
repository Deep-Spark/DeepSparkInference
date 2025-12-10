import tensorrt
import os 
import numpy as np
from typing import Dict, List
from functools import reduce
import pprint


import argparse

from .load_ixrt_plugin import load_ixrt_plugin

from .transformer_cfg import TransformerBaseConfig

TRT_LOGGER = tensorrt.Logger(tensorrt.Logger.ERROR)
load_ixrt_plugin(TRT_LOGGER)


import torch

def create_engine_context(engine_path, logger):
    with open(engine_path, "rb") as f:
        runtime = tensorrt.Runtime(logger)
        assert runtime
        engine = runtime.deserialize_cuda_engine(f.read())
        assert engine
        context = engine.create_execution_context()
        assert context

    return engine, context


def create_context(engine_file):
    host_mem = tensorrt.IHostMemory
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    engine, context = create_engine_context(engine_file, logger)
    
    return engine, context


def allocate_binding_buffer(types_dict, shapes_dict):
    '''
    Allocate binding buffers for trt based on provided types and shapes dict
    '''
    return {
        k: torch.zeros(reduce(lambda v, a: v*a, shape), dtype=types_dict[k]).cuda()
        for k, shape in shapes_dict.items()
    }

class T5TRTEncoder():


    def __init__(
        self,
        trt_engine_file: str,
        config,
        batch_size: int = 1,

    ):

        self.data_type = torch.float16

        self.max_sequence_length = config.max_sequence_length
        self.hidden_size = config.hidden_size
        self.main_input_name = "src_tokens"
        self.batch_size = batch_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.head_size = config.head_size
        
        # We only have one profile to select so we can just grab the profile at the start of the class
        # self.profile_idx = self.get_optimization_profile(batch_size=self.batch_size, sequence_length=1)
        
        print("Start Deserializing Encoder Engine,it will cost a little time...")
        self.trt_engine, self.trt_context = create_context(trt_engine_file)
        print("Deserializing Encoder Engine DONE !")
        
        self.input_shapes = {
            "src_tokens": (self.batch_size, self.max_sequence_length)
        }
        self.input_types = {
            "src_tokens": torch.int32
        }
        
        self.output_shapes = {}
        self.output_types = {}
        
        for layer_index in range(self.num_hidden_layers):  
            self.output_shapes[f"past_key_values.{layer_index}.encoder.key"] = (self.batch_size, self.num_attention_heads, self.max_sequence_length, self.head_size)
            self.output_shapes[f"past_key_values.{layer_index}.encoder.value"] = (self.batch_size, self.num_attention_heads, self.max_sequence_length, self.head_size)
            
            self.output_types[f"past_key_values.{layer_index}.encoder.key"] = torch.float16
            self.output_types[f"past_key_values.{layer_index}.encoder.value"] = torch.float16   
                 
        self.output_shapes["mask"] = (self.batch_size, self.max_sequence_length)
        self.output_types["mask"] = torch.int32
        
        self.bindings = self._allocate_memory(self.input_shapes, self.input_types, self.output_shapes, self.output_types)
        
    def _allocate_memory(self,
                         input_shapes: Dict[str, tuple],
                         input_types: Dict[str, torch.dtype],
                         output_shapes: Dict[str, tuple],
                         output_types: Dict[str, torch.dtype]):
        """Helper function for binding several inputs at once and pre-allocating the results."""
        # Allocate memories as 1D linear buffers for simpler handling of dynamic shapes.
        

        self.inputs = allocate_binding_buffer(input_types, input_shapes)
        self.outputs = allocate_binding_buffer(output_types, output_shapes)
        bindings = [0] * self.trt_engine.num_bindings
        for input_name, input_array in self.inputs.items():
            # Allocate memory for inputs
            input_idx = self.trt_engine.get_binding_index(input_name)
            self.trt_context.set_binding_shape(input_idx, input_shapes[input_name])
            bindings[input_idx] = input_array.data_ptr()

        assert self.trt_context.all_binding_shapes_specified

        for output_name, output_array in self.outputs.items():
            # Output shape should be allocated from context size
            output_idx = self.trt_engine.get_binding_index(output_name)
            bindings[output_idx] = output_array.data_ptr()
            

        return bindings

    def forward(self, input_ids, *args, **kwargs):

        self.bindings[0] = input_ids.data_ptr()
        self.trt_context.set_binding_shape(0, input_ids.shape)
        self.trt_context.execute_v2(self.bindings)
            
        return self.outputs
    
    def clear(self):
        del self.trt_context
        del self.trt_engine



class T5TRTDecoder():

    def __init__(
        self,
        trt_engine_file, 
        hf_config,
        batch_size: int = 1,
        num_beams: int = 1
    ):
        self.data_type =  torch.float16
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.use_cache = True
        self.max_input_length = hf_config.max_sequence_length
        self.max_output_length = hf_config.max_sequence_length
        
        self.device = torch.device('cuda') 
        self.main_input_name = "token_id"  #shape:[bsz,1]
        self.second_input_name = "steps"   #shape:[1]
        self.third_input_name = "mask"     #shape:[bsz,input_length] 
        
        self.main_out_name ="decoder_id"

        
        self.encoder_hidden_size = hf_config.hidden_size
        self.num_heads = hf_config.num_attention_heads
        self.embedding_size_per_head = hf_config.head_size
        self.num_decoder_layers = hf_config.num_hidden_layers
        
        print("Start Deserializing Decoder Engine,it will cost a little time...")
        self.trt_engine, self.trt_context = create_context(trt_engine_file)
        
        print("Deserializing Decoder Engine DONE !")
        self.bindings = [0] * self.trt_engine.num_bindings
        
        
        
        self.output = torch.ones((batch_size,1), dtype=torch.int32).cuda()
        out_index_1 = self.trt_engine.get_binding_index(self.main_out_name)
        self.bindings[out_index_1] = self.output.data_ptr()   
    
        if self.use_cache:

            self.self_attention_cache = {}
            self_attention_kv_shape = (self.batch_size * num_beams, self.num_heads,self.max_output_length - 1,self.embedding_size_per_head)

            # Set self attention kv cache shape and type
            for i in range(self.num_decoder_layers):
                for code in ["key", "value"]:
                    
                    self_attention_name = f"key_values.{i}.decoder.{code}"
                    input_buffer = torch.zeros(self_attention_kv_shape, dtype = self.data_type).cuda()
    
                    input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                    self.self_attention_cache[self_attention_name] = input_buffer
                    self.bindings[input_idx] = input_buffer.data_ptr()
                    
                    output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)
                    #TODO Allocate self attention buffer. The buffer is used both as inputs and outputs,IxRT now ERROR
                    #self.bindings[output_idx] = input_buffer.data_ptr()
            
                    self_attention_name_out = f"key_values.{i}.decoder.{code}.output"
                    output_buffer = torch.zeros(self_attention_kv_shape, dtype = self.data_type).cuda()
                    self.self_attention_cache[self_attention_name_out] = output_buffer
                    self.bindings[output_idx] = output_buffer.data_ptr()
                
       
            self.kv_cache_binding_offset = 3 # 0: token_id, 1: steps,2:mask,  kv cache input indices start from 3
            self.cross_kv_cache_binding_offset = self.kv_cache_binding_offset + 2 * self.num_decoder_layers

            
    def _switch_input_output_binding(self):
        '''
        For kv cache mode, switch input and output pointers to avoid data concurrency issue
        '''
        for i in range(self.num_decoder_layers):
            for code in ["key", "value"]:
                self_attention_name = f"key_values.{i}.decoder.{code}"
                input_idx = self.trt_engine.get_binding_index("past_" + self_attention_name)
                output_idx = self.trt_engine.get_binding_index("present_" + self_attention_name)
                # Switch generation mode kv cache bindings
                temp = self.bindings[output_idx]
                self.bindings[output_idx] = self.bindings[input_idx]
                self.bindings[input_idx] = temp    
        
        
    def forward(self, input_ids, encoder_out, step, sequence_len, *args, **kwargs):
        
        # Get the batch size.
        bs = input_ids.shape[0] # in beam search mode, bs is batch_size * num_beams
        ##############################################################################################
        #input bindings
        input_ids = input_ids.cuda()
        
        index_1 = self.trt_engine.get_binding_index(self.main_input_name)
        self.bindings[index_1] = input_ids.data_ptr()
        self.trt_context.set_binding_shape(index_1, input_ids.shape)
        
        #shape not channge
        index_2 = self.trt_engine.get_binding_index(self.second_input_name)
        step_tensor = torch.tensor([step + 1],dtype = torch.int32).cuda()
        self.bindings[index_2] = step_tensor.data_ptr()
        
        mask_shape = (bs , sequence_len) 
        index_3 = self.trt_engine.get_binding_index(self.third_input_name)
        self.bindings[index_3] = encoder_out["mask"].data_ptr()
        self.trt_context.set_binding_shape(index_3, mask_shape)
        

        if self.use_cache:
            self_atten_kv_shape = (bs, self.num_heads, step, self.embedding_size_per_head)
            for i in range(self.num_decoder_layers):
                
                self_atten_past_key = f"past_key_values.{i}.decoder.key"
                key_idx = self.trt_engine.get_binding_index(self_atten_past_key)
                self.trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i, self_atten_kv_shape)
                
                self_atten_past_value = f"past_key_values.{i}.decoder.key"
                value_idx = self.trt_engine.get_binding_index(self_atten_past_value) 
                self.trt_context.set_binding_shape(self.kv_cache_binding_offset+2*i + 1, self_atten_kv_shape)
            
            
            cross_atten_kv_shape = (bs, self.num_heads, sequence_len, self.embedding_size_per_head)     
            for i in range(self.num_decoder_layers):
                cross_atten_past_key = f"past_key_values.{i}.encoder.key"
                key_idx = self.trt_engine.get_binding_index(cross_atten_past_key)
                self.bindings[key_idx] = encoder_out[cross_atten_past_key].data_ptr()
                self.trt_context.set_binding_shape(key_idx, cross_atten_kv_shape)
                
                cross_atten_past_value = f"past_key_values.{i}.encoder.value"
                value_idx = self.trt_engine.get_binding_index(cross_atten_past_value)                
                self.bindings[value_idx] = encoder_out[cross_atten_past_value].data_ptr()
                self.trt_context.set_binding_shape(value_idx, cross_atten_kv_shape)
                        
        ##############################################################################################
                
        #output bindings                     
        assert self.trt_context.all_binding_shapes_specified                   
        self.trt_context.execute_v2(self.bindings)
        self._switch_input_output_binding()
        
        return self.output
    
    def clear(self):
        del self.trt_context
        del self.trt_engine
    

    
def inference(config,encoder,decoder,input_ids):
    
    prev_tokens = torch.full((input_ids.shape[0],1), int(config.sos_token_id),dtype = torch.int32).cuda()
    encoder_out = encoder.forward(input_ids)
    result_tokens = torch.full((input_ids.shape[0],config.max_sequence_length), int(config.sos_token_id),dtype = torch.int32).cuda()
    sequence_len = input_ids.shape[1]
    for step in range(config.max_sequence_length-1):
        current_tokens = decoder.forward(prev_tokens,encoder_out,step,sequence_len) 
        if step > 1:        
            update_tokens = torch.where(prev_tokens == int(config.eos_token_id), int(config.eos_token_id), current_tokens)
            result_tokens[:,step:step+1] = update_tokens
            prev_tokens = update_tokens 
            if torch.all(update_tokens == int(config.eos_token_id)): 
                break    
        else:
            result_tokens[:,step:step+1] = current_tokens
            prev_tokens = current_tokens 
                           
    return result_tokens 




def benchmark(config,encoder,decoder,input_ids,prev_tokens):
    encoder_out = encoder.forward(input_ids) 
    sequence_len = input_ids.shape[1]
    test_step =0
    for step in range(config.max_sequence_length-1): 
        test_step +=1
        current_tokens = decoder.forward(prev_tokens,encoder_out, step,sequence_len) 
        if step > 1:        
            update_tokens = torch.where(prev_tokens == int(config.eos_token_id), int(config.eos_token_id), current_tokens)
            prev_tokens = update_tokens 
            if torch.all(update_tokens == int(config.eos_token_id)): 
                break    
        else:
            prev_tokens = current_tokens 
        
 
    

    
def main():
    parser = argparse.ArgumentParser(description="TensorRT Transformer Sample")
    parser.add_argument("--decoder_engine", required=False, default="/inferencesamples/data/checkpoints/transformer/wmt14.en-fr.joined-dict.transformer/Decoder.engine", help="The transformer engine file, ex transformer.engine")
    parser.add_argument("--encoder_engine", required=False, default="/inferencesamples/data/checkpoints/transformer/wmt14.en-fr.joined-dict.transformer/Encoder.engine", help="The transformer engine file, ex transformer.engine")
    parser.add_argument("--config_file", required=False, default="./data/wmt14_en_de/transformer_config.json", help="The transformer config file")
        
    args = parser.parse_args()
    
    config_path =args.config_file
    config = TransformerBaseConfig(config_path)
       
    input_ids = torch.from_numpy(np.load("/inferencesamples/benchmarks/nlp/translation/transformer/plugin/data/tensorrt_input/2.npy")).cuda()
        
    batch_size = input_ids.shape[0]
    decoder = T5TRTDecoder(args.decoder_engine,config,batch_size=batch_size)    
    encoder = T5TRTEncoder(args.encoder_engine,config, batch_size=batch_size)
    
    
    result_tokens= inference(config,encoder,decoder,input_ids) 
    print(result_tokens)
    
        
if __name__ == "__main__":
    main()