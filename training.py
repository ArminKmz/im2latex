from src import utils, encoder, decoder, model, trainer, cnn, embedding
import params
import glob, subprocess

def get_trainer(config):

    vocab_config = {
        'pathes' : [config['formulas_train_path'], config['formulas_validation_path']],
        'unk_token_threshold' : config['unk_token_threshold'],
    }
    vocab = utils.Vocab(vocab_config)
    vocab_size = len(vocab.token2idx.keys())

    cnn_model = cnn.CNN(config['cnn_params']).to(config['device'])

    encoder_model = encoder.Encoder(config['cnn_params']['conv6_c'], config['encoder_hidden_size'],
                                    config['bidirectional'], config['device']).to(config['device'])

    train_loader_config = {
        'batch_size' : config['batch_size'],
        'images_path' : config['images_train_path'],
        'formulas_path' : config['formulas_train_path'],
        'sort_by_formulas_len' : True,
        'shuffle' : False,
    }
    train_loader = utils.data_loader(vocab, train_loader_config)

    embedding_model = embedding.Embedding(vocab_size, config['embedding_size'], vocab.pad_token).to(config['device'])

    decoder_model = decoder.AttnDecoder(config['embedding_size'], config['decoder_hidden_size'],
                                        config['encoder_hidden_size']*(2 if config['bidirectional'] else 1), vocab_size,
                                        config['device']).to(config['device'])

    _model = model.Model(cnn_model, encoder_model, embedding_model, decoder_model, config['device'])

    trainer_config = {
        'device' : config['device'],
        'checkpoints_dir' :  config['checkpoints_dir'],
        'log_dir' : config['log_dir'],
        'print_every_batch' : config['print_every_batch'],
        'clip': config['clip'],
        'learning_rate' : config['learning_rate'],
        'learning_rate_decay' : config['learning_rate_decay'],
        'learning_rate_decay_step' : config['learning_rate_decay_step'],
        'learning_rate_min': config['learning_rate_min'],
        'teacher_forcing_ratio' : config['teacher_forcing_ratio'],
        'teacher_forcing_ratio_decay' : config['teacher_forcing_ratio_decay'],
        'teacher_forcing_ratio_decay_step' : config['teacher_forcing_ratio_decay_step'],
        'teacher_forcing_ratio_min': config['teacher_forcing_ratio_min'],
    }

    _trainer = trainer.Trainer(_model, train_loader, trainer_config)

    return _trainer

def train(config):

    logger_config = {
        'owner' : 'Training.train',
        'log_dir' : config['log_dir'],
        'timezone' : 'Iran',
    }
    logger = utils.Logger(logger_config)

    _trainer = get_trainer(config)

    checkpoints = sorted(glob.glob(_trainer.checkpoints_dir + '/*.pt'))

    if len(checkpoints):
        _trainer.load(checkpoints[-1])

    last_epoch = _trainer.current_epoch

    predicted_path = 'tmp/predicted-train.txt'
    # if shuffled or sorted by length -> should update target_path accordingly
    target_path = config['formulas_train_path']
    dot = target_path.rfind('.')
    target_path = target_path[:dot] + '_' + target_path[dot:]
    if _trainer.train_loader.shuffle or _trainer.train_loader.sort_by_formulas_len:
        f = open(target_path, 'w')
        for formula in _trainer.train_loader.formulas:
            joinedformula = ' '.join(token for token in formula)
            f.write(joinedformula + '\n')
        f.close()

    for epoch in range(last_epoch, config['epochs']):
        predictions, epoch_loss, epoch_acc = _trainer.train_one_epoch()
        checkpoint = _trainer.save(epoch+1, epoch_loss, epoch_acc)

        bleu_message, edit_message = bleu_and_edit_distance(predictions, predicted_path, target_path)
        logger(bleu_message)
        logger(edit_message)
        evaluate('validation', config, checkpoint)

def evaluate(evalset, config, checkpoint=None):

    assert(evalset in ['validation', 'test'])

    logger_config = {
        'owner' : 'Training.evaluate',
        'log_dir' : config['log_dir'],
        'timezone' : 'Iran',
    }
    logger = utils.Logger(logger_config)

    _trainer = get_trainer(config)

    if checkpoint is None:
        checkpoints = sorted(glob.glob(_trainer.checkpoints_dir + '/*.pt'))
        assert(len(checkpoints))
        checkpoint = checkpoints[-1]
    _trainer.load(checkpoint)

    if evalset == 'validation':
        valid_loader_config = {
            'batch_size': config['batch_size'],
            'images_path': config['images_validation_path'],
            'formulas_path': config['formulas_validation_path'],
        }
        loader = utils.data_loader(_trainer.train_loader.vocab, valid_loader_config)
    elif evalset == 'test':
        test_loader_config = {
            'batch_size': config['batch_size'],
            'images_path': config['images_test_path'],
        }
        loader = utils.data_loader(_trainer.train_loader.vocab, test_loader_config)

    if loader.has_label:
        predictions, attn_weights, loss, acc = _trainer.evaluate(loader, config['generation_method'])
        logger('loss={}, acc={}'.format(loss, acc))
    else:
        predictions, attn_weights = _trainer.evaluate(loader, config['generation_method'])

    target_path = config['formulas_validation_path'] if evalset == 'validation' else config['formulas_test_path']
    predicted_path = 'tmp/predicted-{}.txt'.format(evalset)

    bleu_message, edit_message = bleu_and_edit_distance(predictions, predicted_path, target_path)

    logger(bleu_message)
    logger(edit_message)

def bleu_and_edit_distance(predictions, predicted_path, target_path):
    with open(predicted_path, 'w') as f:
        for pred in predictions:
            f.write(pred+'\n')
    output = subprocess.check_output('python src/Evaluation/bleu_score.py --target-formulas {} --predicted-formulas {} --ngram 5'.
                format(target_path, predicted_path), shell=True)
    output = str(output)
    bleu_message = output[output.find('BLEU'):-3]
    output = subprocess.check_output('python src/Evaluation/edit_distance.py --target-formulas {} --predicted-formulas {}'.
                format(target_path, predicted_path), shell=True)
    output = str(output)
    edit_message = output[output.find('Edit'):-3]
    return bleu_message, edit_message
