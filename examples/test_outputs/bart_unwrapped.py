from blocks.models.encdec.bart import UnwrappedMbart, Unwrappedbart
from transformers import MBart50TokenizerFast, BartConfig, MBartConfig
import torch

def UnwrappedMBartTest():
    config_bart = {
        '_target_': 'transformers.MBartForConditionalGeneration',
        'pretrained_model_name': "facebook/mbart-large-50-many-to-many-mmt"
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

    model, vector_model, discretizer, _ = Unwrappedbart(config_bart, discretizer_config, discretizer_config).values()
    model.eval()
    vector_model.eval()

    model_enfr, vector_model_enfr, en_discretizer, _, = UnwrappedMbart(model_name="facebook/mbart-large-50-many-to-many-mmt").values()
    model_fren, vector_model_fren, _, fr_discretizer, = UnwrappedMbart(model_name="facebook/mbart-large-50-many-to-many-mmt").values()

    tokenizer_enfr = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
    tokenizer_fren = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="fr_XX", tgt_lang="en_XX")

    en_batch = ["Everything not saved will be lost.", "one must imagine Sisyphus happy."]
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "il faut imaginer Sisyphe heureux."]
    print('en_batch:', en_batch)
    print('fr_batch:', fr_batch)

    en_tokenized = tokenizer_enfr(text=en_batch, return_tensors="pt", padding=True)
    fr_tokenized = tokenizer_fren(text=fr_batch, return_tensors="pt", padding=True)
    # to find out what token should be prepended when generating output labels, we pass as text_target the input text
    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)
    prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)
    labels_enfr = torch.cat((prefix_ids_fr.repeat(fr_tokenized['input_ids'].shape[0], 1), fr_tokenized['input_ids'][:, 1:]), axis=1)
    labels_fren = torch.cat((prefix_ids_en.repeat(en_tokenized['input_ids'].shape[0], 1), en_tokenized['input_ids'][:, 1:]), axis=1)

    print('en_tokenized:', en_tokenized)
    print('fr_tokenized:', fr_tokenized)
    print('labels_enfr:', labels_enfr)
    print('labels_fren:', labels_fren)
    
    # original model

    output_model_enfr = model_enfr(**en_tokenized, decoder_input_ids=labels_enfr, return_dict=True, output_hidden_states=True)
    output_model_fren = model_fren(**fr_tokenized, decoder_input_ids=labels_fren, return_dict=True, output_hidden_states=True)
    print('decoded output original model en fr:', tokenizer_enfr.batch_decode(output_model_enfr.logits.argmax(dim=-1), skip_special_tokens=False))
    print('decoded output original model fr en:', tokenizer_fren.batch_decode(output_model_fren.logits.argmax(dim=-1), skip_special_tokens=False))

    # unwrapped model
    # forward pass of one model
    input_en_vector_embeddings = en_discretizer.encoder_embedding_from_id(en_tokenized['input_ids'])
    output_fr_vector_embeddings = fr_discretizer.decoder_embedding_from_id(labels_enfr)
    input_fr_vector_embeddings = fr_discretizer.encoder_embedding_from_id(fr_tokenized['input_ids'])
    output_en_vector_embeddings = en_discretizer.decoder_embedding_from_id(labels_fren)
    
    output_vector_model_enfr = vector_model_enfr.forward(inputs_embeds=input_en_vector_embeddings, decoder_inputs_embeds=output_fr_vector_embeddings,
                                            attention_mask=en_tokenized['attention_mask'],
                                            return_dict=True, output_hidden_states=True)
    output_vector_model_fren = vector_model_fren.forward(inputs_embeds=input_fr_vector_embeddings, decoder_inputs_embeds=output_en_vector_embeddings,
                                            attention_mask=fr_tokenized['attention_mask'],
                                            return_dict=True, output_hidden_states=True)
    discretized_output_enfr = fr_discretizer(output_vector_model_enfr['last_hidden_state'])
    discretized_output_fren = en_discretizer(output_vector_model_fren['last_hidden_state'])

    # print the output of the discretizer discretized_output['id'], decoded with the tokenizer
    print('decoded output decomposed model en fr:', tokenizer_enfr.batch_decode(discretized_output_enfr['id'], skip_special_tokens=False))
    print('decoded output decomposed model fr en:', tokenizer_fren.batch_decode(discretized_output_fren['id'], skip_special_tokens=False))

    # check the logits being the same:
    print('logits are the same:', torch.allclose(discretized_output_enfr['logit'], output_model_enfr.logits, atol=1e-1))
    print('logits are the same:', torch.allclose(discretized_output_fren['logit'], output_model_fren.logits, atol=1e-1))


def UnwrappedBartTest():
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

    model, vector_model, discretizer, _ = Unwrappedbart(config_bart, discretizer_config, discretizer_config).values()
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



if __name__ == '__main__':
    UnwrappedMBartTest()
    # UnwrappedBartTest()