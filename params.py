import torch

config = {
    # ======================
    #        pathes
    # ======================

    'formulas_train_path' : '',
    'formulas_validation_path' : '',
    'formulas_test_path' : '',
    'images_train_path' : '',
    'images_validation_path' : '',
    'images_test_path' : '',

    'checkpoints_dir' : 'checkpoints/',

    'log_dir' : 'logs/log.txt',
    # ======================
    #   general parameters
    # ======================
    'batch_size' : 32,
    'epochs' : 40,
    'print_every_batch' : 100,
    'learning_rate' : 1e-3,
    'learning_rate_decay' : .5,
    'learning_rate_decay_step' : 3,
    'learning_rate_min': 1e-6,
    'teacher_forcing_ratio' : 1,
    'teacher_forcing_ratio_decay' : 0.95,
    'teacher_forcing_ratio_decay_step' : 2,
    'teacher_forcing_ratio_min': 0.7,
    'clip' : 5,
    'unk_token_threshold' : 10,
    'generation_method' : 'greedy', # or 'beam-search'
    'device' : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # ======================
    #   CNN parameters
    # ======================
    'cnn_params' : {
        'conv1_c' : 64,
        'conv1_k' : (3, 3),
        'conv1_s' : (1, 1),
        'conv1_p' : (1, 1),

        'pool1_k' : (2, 2),
        'pool1_s' : (2, 2),
        'pool1_p' : (0, 0),

        'conv2_c' : 128,
        'conv2_k' : (3, 3),
        'conv2_s' : (1, 1),
        'conv2_p' : (1, 1),

        'pool2_k' : (2, 2),
        'pool2_s' : (2, 2),
        'pool2_p' : (0, 0),

        'conv3_c' : 256,
        'conv3_k' : (3, 3),
        'conv3_s' : (1, 1),
        'conv3_p' : (1, 1),

        'conv4_c' : 256,
        'conv4_k' : (3, 3),
        'conv4_s' : (1, 1),
        'conv4_p' : (1, 1),

        'pool3_k' : (2, 1),
        'pool3_s' : (2, 1),
        'pool3_p' : (0, 0),

        'conv5_c' : 512,
        'conv5_k' : (3, 3),
        'conv5_s' : (1, 1),
        'conv5_p' : (1, 1),

        'pool4_k' : (1, 2),
        'pool4_s' : (1, 2),
        'pool4_p' : (0, 0),

        'conv6_c' : 512,
        'conv6_k' : (3, 3),
        'conv6_s' : (1, 1),
        'conv6_p' : (1, 1),
    },
    # ======================
    #   seq2seq parameters
    # ======================
    'embedding_size' : 80,
    'decoder_hidden_size' : 512,
    'encoder_hidden_size' : 256,
    'bidirectional' : True,
}
