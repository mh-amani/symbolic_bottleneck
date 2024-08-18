from blocks.modules.model_unwrapper.transformer_enc_dec_unwrapper import EncoderDecoderUnwrapper
import torch
from transformers import T5Tokenizer

# tokenizer.pad_token_id = 0, is appended to inputs to the decode of T5

config_t5 = {
        '_target_': 'transformers.T5ForConditionalGeneration',
        'pretrained_model_name': "google-t5/t5-small"
    }
config_T5_base = {
        '_target_': 'transformers.T5Model',
        'pretrained_model_name': "google-t5/t5-small"
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

def UnwrappedOneT5Test():       
    unwrapped_model = EncoderDecoderUnwrapper(config_t5, discretizer_config, discretizer_config)
    model = unwrapped_model['model']
    vector_model = unwrapped_model['vector_model']
    input_discretizer = unwrapped_model['discretizer_enc']
    output_discretizer = unwrapped_model['discretizer_dec']
    # for t5 you need to scale the output weights
    input_discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)
    output_discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)

    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
    labels = tokenizer("Das Haus ist wunderbar.", return_tensors="pt").input_ids

    # the forward function automatically creates the correct decoder_input_ids
    logits = model(input_ids=input_ids, labels=labels).logits
    print(tokenizer.batch_decode(logits.argmax(dim=-1)))
    
    labels_with_pad = torch.cat((torch.ones((labels.shape[0], 1), dtype=torch.long).to(labels.device)* \
                                    tokenizer.pad_token_id, labels), dim=1)
    # using vectorized model
    input_vector_embeddings = input_discretizer.encoder_embedding_from_id(input_ids)
    input_attention_mask = torch.ones_like(input_ids)
    output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels_with_pad)
    output_attention_mask = torch.ones_like(labels_with_pad)
    # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels)
    # output_attention_mask = torch.ones_like(labels)
    output_vector_model = vector_model(inputs_embeds=input_vector_embeddings, attention_mask=input_attention_mask,
                                        decoder_inputs_embeds=output_vector_embeddings, decoder_attention_mask=output_attention_mask
                                        ,use_cache= None, return_dict=True, output_hidden_states=True)
    discretized_output = output_discretizer(output_vector_model['decoder_hidden_states'][-1])
    print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))



def UnwrappedTwoT5Test():    
    model, vector_model, discretizer, _ = EncoderDecoderUnwrapper(config_t5, discretizer_config, discretizer_config).values()
    model.eval()
    vector_model.eval()
    model_enfr = model
    model_fren = model
    vector_model_enfr = vector_model
    vector_model_fren = vector_model
    en_discretizer = discretizer
    fr_discretizer = discretizer
    en_discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)
    fr_discretizer.linear_head_scale = vector_model.model_dim ** (-0.5)
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    en_batch = ["Everything not saved will be lost.", "one must imagine Sisyphus happy."]
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "il faut imaginer Sisyphe heureux."]
    print('en_batch:', en_batch)
    print('fr_batch:', fr_batch)
    enfr_input_batch = ['translate English to French: ' + x for x in en_batch]
    fren_input_batch = ['translate French to English: ' + x for x in fr_batch]

    enfr_input_tokenized = tokenizer(text=enfr_input_batch, return_tensors="pt", padding=True)
    fren_input_tokenized = tokenizer(text=fren_input_batch, return_tensors="pt", padding=True)
    fr_tokenized = tokenizer(text=fr_batch, return_tensors="pt", padding=True)
    en_tokenized = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    
    prefix_ids = torch.tensor(tokenizer.pad_token_id).unsqueeze(0)
    enfr_decoder_tokenized = torch.cat((prefix_ids.repeat(fr_tokenized['input_ids'].shape[0], 1), fr_tokenized['input_ids'][:, 1:]), axis=1)
    fren_decoder_tokenized = torch.cat((prefix_ids.repeat(en_tokenized['input_ids'].shape[0], 1), en_tokenized['input_ids'][:, 1:]), axis=1)

    print('en_tokenized:', en_tokenized)
    print('fr_tokenized:', fr_tokenized)
    print('labels_enfr:', enfr_decoder_tokenized)
    print('labels_fren:', fren_decoder_tokenized)
    
    # original model

    output_model_enfr = model_enfr(**enfr_input_tokenized, decoder_input_ids=enfr_decoder_tokenized, return_dict=True, output_hidden_states=True)
    output_model_fren = model_fren(**fren_input_tokenized, decoder_input_ids=fren_decoder_tokenized, return_dict=True, output_hidden_states=True)
    print('decoded output original model en fr:', tokenizer.batch_decode(output_model_enfr.logits.argmax(dim=-1), skip_special_tokens=False))
    print('decoded output original model fr en:', tokenizer.batch_decode(output_model_fren.logits.argmax(dim=-1), skip_special_tokens=False))

    # unwrapped model
    # forward pass of one model
    input_en_vector_embeddings = en_discretizer.encoder_embedding_from_id(enfr_input_tokenized['input_ids'])
    output_fr_vector_embeddings = fr_discretizer.decoder_embedding_from_id(enfr_decoder_tokenized)
    input_fr_vector_embeddings = fr_discretizer.encoder_embedding_from_id(fren_input_tokenized['input_ids'])
    output_en_vector_embeddings = en_discretizer.decoder_embedding_from_id(fren_decoder_tokenized)
    
    output_vector_model_enfr = vector_model_enfr.forward(inputs_embeds=input_en_vector_embeddings, decoder_inputs_embeds=output_fr_vector_embeddings,
                                            attention_mask=enfr_input_tokenized['attention_mask'],
                                            return_dict=True, output_hidden_states=True)
    output_vector_model_fren = vector_model_fren.forward(inputs_embeds=input_fr_vector_embeddings, decoder_inputs_embeds=output_en_vector_embeddings,
                                            attention_mask=fren_input_tokenized['attention_mask'],
                                            return_dict=True, output_hidden_states=True)
    discretized_output_enfr = fr_discretizer(output_vector_model_enfr['decoder_hidden_states'][-1])
    discretized_output_fren = en_discretizer(output_vector_model_fren['decoder_hidden_states'][-1])

    # print the output of the discretizer discretized_output['id'], decoded with the tokenizer
    print('decoded output decomposed model en fr:', tokenizer.batch_decode(discretized_output_enfr['id'], skip_special_tokens=False))
    print('decoded output decomposed model fr en:', tokenizer.batch_decode(discretized_output_fren['id'], skip_special_tokens=False))

    # check the logits being the same:
    print('logits are the same:', torch.allclose(discretized_output_enfr['logit'], output_model_enfr.logits, atol=1e-1))
    print('logits are the same:', torch.allclose(discretized_output_fren['logit'], output_model_fren.logits, atol=1e-1))


if __name__ == '__main__':
    UnwrappedOneT5Test()
    UnwrappedTwoT5Test()
