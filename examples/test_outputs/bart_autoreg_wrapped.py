import torch

def test_auto_reg_wrapper_bart():
    torch.random.manual_seed(42)
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    unwrapped_model = UnwrappedMbart()
    vector_model = unwrapped_model['vector_model']
    en_discretizer = unwrapped_model['discretizer_enc']
    fr_discretizer = unwrapped_model['discretizer_dec']
    
    from transformers import MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)
    prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)

    config = {'device': 'cpu',
            'use_past_key_values': False, 'use_last_step_states': False,
            'max_lengths': {'input': 30, 'output': 30,},
            'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
                                    'output_eos_token_id': tokenizer.eos_token_id, 
                                    'output_pad_token_id': tokenizer.pad_token_id,
                                    'output_unknown_token_id': tokenizer.unk_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
            'output_prepending_ids': [2, 250008]
            }
    
    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, en_discretizer, fr_discretizer, config)
    fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, fr_discretizer, en_discretizer, config)

    # an example input and output sequences
    en_batch = ['Everything that is lost that is lost.', 'we must imagine Sisyphe happy.']
    input_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)
    input_ids_en = input_en['input_ids']
    output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, input_attention_mask=None, input_embeds_enc=None,
                                                teacher_force_output=False)
    # print the output of the model
    print('--'*20)
    print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
    print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # another example, starting from half of the output instead of the prepending embeddings
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "Il faut imaginer Sisyphe heureux."]
    output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids'][:, 1:5]
    output_ids_fr = torch.cat((prefix_ids_fr.repeat(2, 1), output_ids_fr), axis=1)
    output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, output_ids=output_ids_fr, 
                                                teacher_force_output=False)
    print('--'*20)
    print('auto-regressive forward pass - starting from half of the output')
    print('decoded input:', tokenizer.batch_decode(output_ids_fr, skip_special_tokens=False))
    print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # another example, teacher forcing the output
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "Il faut imaginer Sisyphe heureux."]
    output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids'][:, 1:]
    output_ids_fr = torch.cat((prefix_ids_fr.repeat(2, 1), output_ids_fr), axis=1)
    output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, output_ids=output_ids_fr, 
                                                teacher_force_output=True)
    print('--'*20)
    print('teacher forced forward pass - teacher forcing the output')
    print('decoded input to decoder:', tokenizer.batch_decode(output_ids_fr, skip_special_tokens=False))
    print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

    # another example, French to English translation without teacher forcing
    prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)
    config['output_prepending_ids'] = prefix_ids_en
    fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, fr_discretizer, en_discretizer, config)
    fr_batch = ["Tout ce qui n'est pas sauvé sera perdu.", "Il faut imaginer Sisyphe heureux."]
    input_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids']
    output_fr_en = fren_autoreg_wrapped_model(input_ids=input_ids_fr, teacher_force_output=False)
    print('--'*20)
    print('auto-regressive forward pass - French to English translation')
    print('decoded output:', tokenizer.batch_decode(output_fr_en['id'], skip_special_tokens=False))


def main():
    test_auto_reg_wrapper_bart()

if __name__ == "__main__":
    main()