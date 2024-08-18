from blocks.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
from transformers import BartConfig
import torch

# to find out what token should be prepended when generating output labels, we pass as text_target the input text
    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008

# example config for bart
config_bart = {
        '_target_': 'transformers.BartForConditionalGeneration',
        'config': BartConfig(d_model=128, encoder_layers=3, decoder_layers=3, 
                             vocab_size=23, max_position_embeddings=40,
                             encoder_attention_heads=2, decoder_attention_heads=2)
        }
    
discretizer_config = {
    '_target_': 'blocks.modules.discrete_bottlenecks.softmax.SoftmaxDiscreteBottleneck',
    'config':{ 
        'dimensions': None,
        'quantize_vector': True, 'temperature': 1.0,
        'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 
        'linear_head_trainable': False, 
        'encoder_embedding': None, 'decoder_embedding': None, 
        'linear_head': None,
        }
    }


########################################################################################################################
# unwrapped models

def UnwrappedBartTest():

    model, vector_model, discretizer, _ = EncoderDecoderUnwrapper(config_bart, discretizer_config, discretizer_config).values()
    model.eval()
    vector_model.eval()

    random_token_input_batch = torch.randint(0, 23, (4, 10))
    random_token_output_batch = torch.randint(0, 23, (4, 15))
    random_token_output_batch[:, 0] = 2 # prepend the <s> token
    random_token_output_batch[:, 14] = 1 # append the </s> token

    input_vector_embeddings = discretizer.encoder_embedding_from_id(random_token_input_batch)
    output_vector_embeddings = discretizer.decoder_embedding_from_id(random_token_output_batch)

    output_model = model(input_ids=random_token_input_batch, decoder_input_ids=random_token_output_batch, 
                         return_dict=True, output_hidden_states=True)
    output_vector_model = vector_model.forward(inputs_embeds=input_vector_embeddings, decoder_inputs_embeds=output_vector_embeddings,
                                            return_dict=True, output_hidden_states=True)
    discretized_output = discretizer(output_vector_model['last_hidden_state'])

    print("input token batch:", random_token_input_batch)
    print("output token batch:", random_token_output_batch)
    print("decoded output original model:", output_model.logits.argmax(dim=-1))
    print("decoded output decomposed model:", discretized_output['id'])
    print("logits are the same:", torch.allclose(discretized_output['logit'], output_model.logits, atol=1e-1))




########################################################################################################################
# Autoreg-wrapped model

if __name__ == '__main__':
    UnwrappedBartTest()