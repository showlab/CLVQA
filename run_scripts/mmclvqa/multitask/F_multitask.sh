CUDA_VISIBLE_DEVICES=1,2,3,4 mmf_run dataset=clvqa   \
model=unicl   \
config=EXP_CONFIG/cl_multitask_unicl.yaml  \
training.CL.use_cl=False \
training.checkpoint_interval=40000 \
training.batch_size=96