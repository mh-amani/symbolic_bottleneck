

def UnwrappedPretrainedT5(base=True, pretrained_hf_name='t5-small'):
    """
    Unwraps the T5 model to get the encoder and decoder weights.
    Returns:
        model_enfr: The English to French model.
        model_fren: The French to English model.
        vector_model_enfr: The English to French model without embedding and head, pure transfomer.
        vector_model_fren: The French to English model without embedding and head, pure transfomer.
        discretizer_en: The English to French discretizer.
        discretizer_fr: The French to English discretizer.
        tokenizer_enfr: The English to French tokenizer.
        tokenizer_fren: The French to English tokenizer.
    """
    from transformers import T5ForConditionalGeneration, T5Model, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(pretrained_hf_name)
    if not base:
        model = T5ForConditionalGeneration.from_pretrained(pretrained_hf_name)
        vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
        vector_model = T5Model.from_pretrained(pretrained_hf_name)
    else:
        model = T5Model.from_pretrained(pretrained_hf_name)
        vector_model, encoder_embedding, decoder_embedding, linear_head = EncoderDecoderUnwrapper(model).values()
        vector_model = model

      
    embed_dim = vector_model.config.d_model
    vocab_size = vector_model.config.vocab_size

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
