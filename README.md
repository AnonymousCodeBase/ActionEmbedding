# ActionEmbedding
This is the code repository for the paper [Learning Action - Transferable Policy with Action Embedding](http://arxiv.org/abs/1909.02291).



### Experiments

The code provide two sets of experiments

- Gridworld
- InvertedPendulum (need [mujoco license](https://www.roboti.us/license.html))



There needs some modifications for tasks without state embedding

- sac.py
  - Line 40-57
- agent.py
  - Line 64-68





### Run 

```
python agent.py
			-i identifier
			-t task_id
			
			-transfer 0/1
			-ckpt_path path
			-ckpt_step step
			-source_t source_task_id
```



### 