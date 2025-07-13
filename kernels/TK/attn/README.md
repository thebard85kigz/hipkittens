```
root@gpu-10:/workdir/AMD-benchmarking-harness/kernels/TK/attn# make
/opt/rocm/bin/hipcc simple_kernel.cpp -DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 -w --save-temps -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/include -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/prototype -I/opt/conda/envs/py_3.12/include/python3.12 -I/opt/conda/envs/py_3.12/lib/python3.12/site-packages/pybind11/include -L/opt/conda/envs/py_3.12/lib/python3.12/config-3.12-x86_64-linux-gnu -L/opt/conda/envs/py_3.12/lib  -lpthread -ldl  -lutil -lm  -shared -fPIC -Rpass-analysis=kernel-resource-usage -I/workdir/AMD-benchmarking-harness/ThunderKittens-HIP/include -I/opt/rocm/include/hip  \
    -o tk_kernel.cpython-312-x86_64-linux-gnu.so 2>&1 | tee /workdir/data_logs/0713_013900_outputs/make_build.log
remark: simple_kernel.cpp:53:0: Function Name: _Z10attend_kerILi64EEv12attn_globalsIXT_EE [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     SGPRs: 40 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     VGPRs: 90 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     AGPRs: 16 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
remark: simple_kernel.cpp:53:0:     LDS Size [bytes/block]: 1024 [-Rpass-analysis=kernel-resource-usage]
root@gpu-10:/workdir/AMD-benchmarking-harness/kernels/TK/attn# python test_python.py
src: simple_kernel.cpp
Warning: tk_kernel.cpython-313-x86_64-linux-gnu.so not found at /workdir/AMD-benchmarking-harness/kernels/TK/attn/tk_kernel.cpython-313-x86_64-linux-gnu.so, skipping.
/workdir/AMD-benchmarking-harness/kernels/TK/attn/test_python.py:97: UserWarning: Using AOTriton backend for Flash Attention forward... (Triggered internally at /var/lib/jenkins/pytorch/aten/src/ATen/native/transformers/hip/flash_attn/flash_api.h:267.)
  out_ref = scaled_dot_product_attention(q, k, v, is_causal=causal)
out_ref.dtype=torch.bfloat16
PyTorch reference average execution time: 0.3226 ms
PyTorch reference performance: 212.99 TFLOPS for B=16 H=16 N=1024 D=64 causal=False
out.dtype=torch.bfloat16
Average execution time: 1.0563 ms
Performance: 65.06 TFLOPS for 1024x1024 matrix multiplication.

Max error between kernel and reference: 0.00390625
Max error: 0.00390625
Mean error: 0.00018589969840832055
Number of large errors (>0.1): 0
```