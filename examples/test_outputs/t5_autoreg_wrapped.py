import torch
def test_auto_reg_wrapper_t5():
    torch.random.manual_seed(42)
    # an example for the encoder-decoder pretrained T5 model:
    # get the models and the discretizers
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    unwrapped_model = UnwrappedPretrainedT5(base=False)
    model = unwrapped_model['model']
    vector_model = unwrapped_model['vector_model']
    input_discretizer = unwrapped_model['discretizer_enc']
    output_discretizer = unwrapped_model['discretizer_dec']

    input_ids = tokenizer("translate English to French: Well, to continue to being alive is also an art.", return_tensors="pt").input_ids
    output_ids = tokenizer("Alors, continuer à vivre est aussi un art.", return_tensors="pt").input_ids
    # the forward function automatically creates the correct decoder_input_ids
    logits = model(input_ids=input_ids, labels=output_ids).logits
    print(tokenizer.batch_decode(logits.argmax(dim=-1)))
    
    config = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': False,
            'max_lengths': {'input': 30, 'output': 30,},
            'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
                                    'output_eos_token_id': tokenizer.eos_token_id, 
                                    'output_pad_token_id': tokenizer.pad_token_id,
                                    'output_unknown_token_id': tokenizer.unk_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            'output_prepending_ids': [tokenizer.pad_token_id,] # 0
            }
    
    autoreg_wrapped_t5 = AutoRegWrapper(vector_model, input_discretizer, output_discretizer, config)


    print('--'*20)
    print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    output = autoreg_wrapped_t5(input_ids=input_ids, input_attention_mask=None, input_embeds_enc=None,
                                                teacher_force_output=False)
    print('decoded output:', tokenizer.batch_decode(output['id'], skip_special_tokens=False))

    print('--'*20)
    print('teacher forced forward pass - teacher forcing the output')
    output = autoreg_wrapped_t5(input_ids=input_ids, output_ids=output_ids, teacher_force_output=True)
    print('decoded output:', tokenizer.batch_decode(output['id'], skip_special_tokens=False))

    print('--'*20)
    print('auto-regressive forward pass - starting from half of the output')
    t = 6
    output_ids = torch.cat((torch.ones((output_ids.shape[0], 1), dtype=torch.long).to(output_ids.device)*\
                                     tokenizer.pad_token_id, output_ids), dim=1)[:,:t]
    output = autoreg_wrapped_t5(input_ids=input_ids, output_ids=output_ids, teacher_force_output=False)
    print('decoded input:', tokenizer.batch_decode(output_ids[:, :t], skip_special_tokens=False))
    print('decoded output:', tokenizer.batch_decode(output['id'], skip_special_tokens=False))



    # an example with vector model directly to double check.
    # output_ids_with_pad = torch.cat((torch.ones((output_ids.shape[0], 1), dtype=torch.long).to(output_ids.device)*\
    #                                  tokenizer.pad_token_id, output_ids), dim=1)
    # # using vectorized model
    # input_vector_embeddings = input_discretizer.encoder_embedding_from_id(input_ids)
    # input_attention_mask = torch.ones_like(input_ids)
    # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(output_ids_with_pad)
    # output_attention_mask = torch.ones_like(output_ids_with_pad)
    # # output_vector_embeddings = output_discretizer.decoder_embedding_from_id(labels)
    # # output_attention_mask = torch.ones_like(labels)
    # output_vector_model = vector_model(inputs_embeds=input_vector_embeddings, attention_mask=input_attention_mask,
    #                                    decoder_inputs_embeds=output_vector_embeddings, decoder_attention_mask=output_attention_mask
    #                                    ,use_cache= None,)
    # discretized_output = output_discretizer(output_vector_model['last_hidden_state'])
    # print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))



def main():
    # test_auto_reg_wrapper_bart()
    test_auto_reg_wrapper_t5()
    
if __name__ == "__main__":
    main()
