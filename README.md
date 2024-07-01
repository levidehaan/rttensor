####
tensor rt implementation


```bash
CUDA_VISIBLE_DEVICES=0,1 trtllm-build --checkpoint_dir ./tensor_models/llama3_8b_instruct_enc --output_dir ./tensor_models/llama3_8b_instruct_tensor --gemm_plugin auto --use_custom_all_reduce disable --max_input_len 8192 --max_seq_len 16312 --gpt_attention_plugin auto --paged_state enable --remove_input_padding enable --paged_kv_cache enable --max_beam_width 1 --use_fused_mlp --logits_dtype float16 --context_fmha enable
```

```bash
mpirun -np 2 python app/app.py  --engine_dir tensor_models/llama3_8b_instruct_tensor --hf_model_dir tensor_models/llama3_8b_instruct
```


