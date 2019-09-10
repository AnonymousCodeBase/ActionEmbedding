class Config:
    state_dims = [4, 4, 4]
    action_dims = [4, 16, 64]

    # for embedding
    state_embed_dim = 5
    action_embed_dim = 3

    state_embed_hiddens = [200, 100]
    action_embed_hiddens = [64, 32]
    ac_hiddens = [200, 100]
    # ac_hiddens = [300, 300]

    cell_num = 32
    seq_len = 25
    action_embed_lr = 0.001

    # for ac
    gamma = 0.99
    actor_lr = 0.00001  # 25
    critic_lr = 0.001
    tau = 0.999
    alpha = 0.2

    summary_folder = "summary/"
    model_save_folder = "saved_models/"

    # for agent
    max_step = 100
    epoches = 1500
    memory_size = 100000
    batch_size = 32
    action_batch_size = 32
