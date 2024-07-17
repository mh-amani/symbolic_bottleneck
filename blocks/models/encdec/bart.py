from blocks.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from blocks.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck
from transformers import BartForConditionalGeneration
from blocks.utils import instantiate_from_config

# example config for bart
# config_bart = {
#     '_target_': 'transformers.BartForConditionalGeneration',
#     'vocab_size': 50265,
#     'max_position_embeddings': 1024,
#     'encoder_layers': 12,
#     'encoder_ffn_dim': 4096,
#     'encoder_attention_heads': 16,
#     'decoder_layers': 12,
#     'decoder_ffn_dim': 4096,
#     'decoder_attention_heads': 16,
#     'encoder_layerdrop': 0.0,
#     'decoder_layerdrop': 0.0,
#     'activation_function': 'gelu',
#     'd_model': 1024,
#     'init_std': 0.02,
#     'eos_token_id': 2,
#     'pad_token_id': 1,
#     'bos_token_id': 0,
#     'attention_dropout': 0.1,
#     'activation_dropout': 0.1,
#     'dropout': 0.1,
#     'classifier_dropout': 0.1,
#     'scale_embedding': False,
#     'use_cache': True,
#     'num_labels': 3,
#     'forced_eos_token_id': 2,
#     }



########################################################################################################################
# unwrapped models

def Unwrappedbart(bart_config, discretizer_enc_config, discretizer_dec_config):
    model = instantiate_from_config(bart_config)
    
    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
    vocab_size = vector_model.config.vocab_size 
    embed_dim = vector_model.config.d_model
    dimensions = {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 
                  'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size}
    disc_config = {'dimensions': dimensions, 'encoder_embedding': encoder_embedding,
                     'decoder_embedding': decoder_embedding, 'linear_head': linear_head}
    discretizer_enc_config['config'].update(disc_config)
    discretizer_dec_config['config'].update(disc_config)
    discretizer_enc = instantiate_from_config(discretizer_enc_config)
    discretizer_dec = instantiate_from_config(discretizer_dec_config)

    return {
        'model': model, 'vector_model': vector_model,
        'discretizer_enc': discretizer_enc, 'discretizer_dec': discretizer_dec,}


def UnwrappedMbart(model_name="facebook/mbart-large-50-many-to-many-mmt"):
    """
    Unwraps the MBART model to get the encoder and decoder weights.
    Returns:
    """
    from transformers import MBartForConditionalGeneration
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
    vocab_size = vector_model.config.vocab_size
    embed_dim = vector_model.config.d_model

    discretizer_enc = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size},
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False,
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding,
                                'linear_head': linear_head,})
    discretizer_dec = SoftmaxDiscreteBottleneck({'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size},
                                'quantize_vector': True, 'temperature': 1.0,
                                'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 'linear_head_trainable': False,
                                'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding,
                                'linear_head': linear_head,})
    
    return {
        'model': model, 'vector_model': vector_model,
        'discretizer_enc': discretizer_enc, 'discretizer_dec': discretizer_dec,}



########################################################################################################################
# Autoreg-wrapped model


