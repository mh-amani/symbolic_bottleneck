from blocks.modules.auto_reg_wrapper import UnwrappedMbart, AutoRegWrapper
from transformers import MBart50TokenizerFast
import torch

# get the models and the discretizers 
unwrapped_model = UnwrappedMbart()
vector_model = unwrapped_model['vector_model']
en_discretizer = unwrapped_model['discretizer_enc']
fr_discretizer = unwrapped_model['discretizer_dec']

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")
# prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
# tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)
config = {'device': 'cpu',
        'use_past_key_values': False, 'use_last_step_states': True,
        'max_lengths': {'input': 30, 'output': 30,},
        'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
                                'output_eos_token_id': tokenizer.eos_token_id, 
                                'output_pad_token_id': tokenizer.pad_token_id,
                                'output_unknown_token_id': tokenizer.unk_token_id,},
        'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': True,},
        'output_prepending_ids': prefix_ids_fr
        }
enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, en_discretizer, fr_discretizer, config)


# an example input and output sequences
en_batch = ['One must imagine Sisyphus happy.',]
fr_batch = ['Il faut imaginer Sisyphe heureux.',]
input_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)
input_ids_en = input_en['input_ids']
output_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids']


output_en_fr = enfr_autoreg_wrapped_model(input_ids=input_ids_en, input_attention_mask=None, input_embeds_enc=None,
                                            teacher_force_output=False)


# print the output of the model
print('--'*20)
print('auto-regressive forward pass - starting from the prepending embeddings (bos!)')
print('decoded output:', tokenizer.batch_decode(output_en_fr['id'], skip_special_tokens=False))

# checking loss of the happy/heureux token
print("output_en_fr['id'][:, -3]: ", tokenizer.batch_decode(output_en_fr['id'][:, -3], skip_special_tokens=False))
# print("output_ids_fr[-3] ", tokenizer.batch_decode(output_ids_fr[:, -3], skip_special_tokens=False))
id_heureux = tokenizer.encode('heureux', return_tensors="pt")[0, 1]
# heureux: 159508 == output_ids_fr[:, -3]
print('probability of heureux:', output_en_fr['score'][:, -3, id_heureux], 'log probability:', torch.log(output_en_fr['score'][:, -3, id_heureux]))
loss = torch.nn.functional.nll_loss(output_en_fr['score'][:, -3, :].log(), output_ids_fr[:, -3])
print('loss on the heureux token:', loss)



output_en_fr['score'].retain_grad()
output_en_fr['quantized_vector_encoder'].retain_grad()
output_en_fr['quantized_vector_decoder'].retain_grad()
output_en_fr['output_attention_mask'].retain_grad()
output_en_fr['p_not_eos'].retain_grad()

torch.autograd.set_detect_anomaly(True)

loss.backward()
print('gradient of heureux probability token:',
        output_en_fr['score'].grad[:, -3, id_heureux])
print('gradient of heureux probability token w.r.t to other probabilties:',
        output_en_fr['score'].grad[:, -3, :])
print('gradient of heureux probability token w.r.t to the quantized vector encoder:',
        output_en_fr['quantized_vector_encoder'].grad)
print('gradient of heureux probability token w.r.t to the quantized vector decoder:',
        output_en_fr['quantized_vector_decoder'].grad)
print('gradient of heureux probability token w.r.t to the output_attention_mask:',
        output_en_fr['output_attention_mask'].grad)

# gradient w.r.t. model parameters and inputs
# print('gradient of heureux probability token w.r.t to the model parameters:',
#         output_en_fr['model'].parameters())

named_model_param_dict = dict(enfr_autoreg_wrapped_model.named_parameters())
print('gradient of heureux probability token w.r.t to model.decoder.layers.11.final_layer_norm.weight:',
        named_model_param_dict['model.decoder.layers.11.final_layer_norm.weight'].grad)

print('gradient of heureux probability token w.r.t to model.output_discretizer.encoder_embedding.weight:',
    named_model_param_dict['output_discretizer.encoder_embedding.weight'].grad)

print('gradient of heureux probability token w.r.t to model.output_discretizer.decoder_embedding.weight:',
    named_model_param_dict['output_discretizer.decoder_embedding.weight'].grad)

