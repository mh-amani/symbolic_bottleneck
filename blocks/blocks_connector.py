from typing import Any, Dict, Tuple
import torch
from torch.nn import ModuleDict, Module
from transformers import MBart50TokenizerFast  
from blocks.unwrapped_models.enc_dec_unwrapper import UnwrappedMbart
from blocks.modules.auto_reg_wrapper import AutoRegWrapper


class BlocksConnector(Module):
    """
    a wrapper connecting two sequence models with discrete bottleneck layers
    """
    def __init__(
        self,
        model_x_to_y: Module,
        model_y_to_z: Module,
        config,
    ) -> None:

        super().__init__()
        self.config = config
        self.model_x_to_y = model_x_to_y
        self.model_y_to_z = model_y_to_z

    def forward(self, x_ids: torch.Tensor=None, x_attention_mask: torch.Tensor=None, x_embeds_enc: torch.Tensor=None,
                      y_prepending_ids: torch.Tensor=None, y_prepending_embeds_enc: torch.Tensor=None, y_prepending_embeds_dec: torch.Tensor=None,
                      z_ids: torch.Tensor=None, z_attention_mask: torch.Tensor=None, z_embeds_dec: torch.Tensor=None,
                      teacher_force_z: bool=True, max_y_length: int=None, max_z_length: int=None) -> Dict[str, Any]:
        """Perform a forward pass through the models x -> y -> z

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if max_y_length is None:
            max_y_length = self.model_x_to_y.max_lengths['output']
        if max_z_length is None:
            max_z_length = self.model_y_to_z.max_lengths['output']        
        xy_outputs = self.model_x_to_y(input_ids=x_ids, input_attention_mask=x_attention_mask, input_embeds_enc=x_embeds_enc,
                output_ids=y_prepending_ids, output_embeds_enc=y_prepending_embeds_enc, output_embeds_dec=y_prepending_embeds_dec,
                teacher_force_output=False, max_output_length=max_y_length)

        y_inputs = self.transform_xy_outputs_to_y_inputs(xy_outputs)
        y_inputs['quantized_vector_encoder'] = y_inputs['quantized_vector_encoder'] * y_inputs['output_attention_mask'].unsqueeze(-1) + \
            (1 - y_inputs['output_attention_mask']).detach().unsqueeze(-1) * y_inputs['quantized_vector_encoder'] # I don't even know what I'm doing here

        yz_outputs = self.model_y_to_z(
            input_embeds_enc=y_inputs['quantized_vector_encoder'], input_attention_mask=y_inputs['output_attention_mask']>0, 
            output_ids=z_ids, output_embeds_dec=z_embeds_dec, output_attention_mask=z_attention_mask,
            teacher_force_output=teacher_force_z, max_output_length=max_z_length,)
            
        quantization_loss = xy_outputs['quantization_loss'] + yz_outputs['quantization_loss']
        
        return { 'id_y': xy_outputs['id'], 'id_z': yz_outputs['id'], 
                'score_y': xy_outputs['score'], 'score_z': yz_outputs['score'],
                'logit_y': xy_outputs['logit'], 'logit_z': yz_outputs['logit'],
                'quantized_vector_encoder': yz_outputs['quantized_vector_encoder'], 'quantized_vector_decoder': yz_outputs['quantized_vector_decoder'],
                'y_attention_mask': xy_outputs['output_attention_mask'], 'z_attention_mask': yz_outputs['output_attention_mask'],
                'quantization_loss': quantization_loss,}
                # 'xy_outputs': xy_outputs, 'yz_outputs': yz_outputs,} # remove later... aligator

    def transform_xy_outputs_to_y_inputs(self, xy_outputs: Dict[str, Any]) -> Dict[str, Any]:
        NotImplementedError
    
def main(): 
    from transformers import MBart50TokenizerFast
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="fr_XX")

    # prefix_ids_fr = tokenizer(text_target="", return_tensors="pt")['input_ids']
    # tokenizer.vocab['</s>']: 2, tokenizer.vocab['en_XX']: 250004, tokenizer.vocab['fr_XX']: 250008
    prefix_ids_fr = torch.tensor([2, 250008]).unsqueeze(0)
    prefix_ids_en = torch.tensor([2, 250004]).unsqueeze(0)
    config = {'device': 'cpu','use_past_key_values': False, 'use_last_step_states': True,
            'max_lengths': {'input': 30, 'output': 30,},
            'control_token_ids': { 'input_pad_token_id': tokenizer.pad_token_id,
                                    'output_eos_token_id': tokenizer.eos_token_id, 
                                    'output_pad_token_id': tokenizer.pad_token_id,
                                    'output_unknown_token_id': tokenizer.unk_token_id,},
            'soft_average': {'p_eos_backward': True, 'p_eos_forward': False ,'word_embeds_with_scores_forward': False,},
            }
    
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    unwrapped_model = UnwrappedMbart()
    vector_model = unwrapped_model['vector_model']
    vector_model.eval()
    en_discretizer = unwrapped_model['discretizer_enc']
    fr_discretizer = unwrapped_model['discretizer_dec']    
    enfr_autoreg_wrapped_model = AutoRegWrapper(vector_model, en_discretizer, fr_discretizer, config|{'output_prepending_ids': [2, 250008]})
    fren_autoreg_wrapped_model = AutoRegWrapper(vector_model, fr_discretizer, en_discretizer, config|{'output_prepending_ids': [2, 250004]})
    
    def transform_xy_outputs_to_y_inputs(xy_outputs: Dict[str, Any]) -> Dict[str, Any]:
        # since bart output has a eos <\s> token prepended in its output, we remove it for feeding to the next model
        return {'output_attention_mask': xy_outputs['output_attention_mask'][:, 1:],
                'quantized_vector_encoder': xy_outputs['quantized_vector_encoder'][:, 1:]}
    
    en_fr_en_connected_models = BlocksConnector(enfr_autoreg_wrapped_model, fren_autoreg_wrapped_model, config=None)
    en_fr_en_connected_models.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs
    fr_en_fr_connected_models = BlocksConnector(fren_autoreg_wrapped_model, enfr_autoreg_wrapped_model, config=None)
    fr_en_fr_connected_models.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs
    
    # Trying different inputs and check the outputs; making sure the model is working as expected
    
    # an example input and output sequences
    sequence_en_1 = "Everything not saved will be lost."
    sequence_en_2 = "one must imagine Sisyphus happy."
    sequence_fr_1= "Tout ce qui n'est pas sauvé sera perdu."
    sequence_fr_2 = "il faut imaginer Sisyphe heureux."
    en_batch = [sequence_en_1, sequence_en_2]
    fr_batch = [sequence_fr_1, sequence_fr_2]
    input_ids_en = tokenizer(text=en_batch, return_tensors="pt", padding=True)['input_ids']
    input_ids_fr = tokenizer(text_target=fr_batch, return_tensors="pt", padding=True)['input_ids']
    output_ids_en = torch.cat((prefix_ids_en.repeat(input_ids_en.shape[0], 1), input_ids_en[:, 1:]), dim=1)
    output_ids_fr = torch.cat((prefix_ids_fr.repeat(input_ids_fr.shape[0], 1), input_ids_fr[:, 1:]), dim=1)

    print('en-fr-en output, no teacherforcing on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=None, teacher_force_z=False)    
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, no teacherforcing on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=None, teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    t = 6
    print('en-fr-en output, few input tokens on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=output_ids_en[:, :t], teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('output z given:', tokenizer.batch_decode(output_ids_en[:, :t], skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, few input tokens on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=output_ids_fr[:, :t], teacher_force_z=False)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('output z given:', tokenizer.batch_decode(output_ids_fr[:, :t], skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('en-fr-en output, teacherforcing on z:')
    output_en_fr_en = en_fr_en_connected_models(x_ids=input_ids_en, z_ids=output_ids_en, teacher_force_z=True)
    print('x input:', tokenizer.batch_decode(input_ids_en, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_en_fr_en['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_en_fr_en['id_z'], skip_special_tokens=False))
    print('-'*50)

    print('fr-en-fr output, teacherforcing on z:')
    output_fr_en_fr = fr_en_fr_connected_models(x_ids=input_ids_fr, z_ids=output_ids_fr, teacher_force_z=True)
    print('x input:', tokenizer.batch_decode(input_ids_fr, skip_special_tokens=False))
    print('y output:', tokenizer.batch_decode(output_fr_en_fr['id_y'], skip_special_tokens=False))
    print('z output:', tokenizer.batch_decode(output_fr_en_fr['id_z'], skip_special_tokens=False))
    print('-'*50)

    # #comparing the output of the model with the output of the out of the box HF model, making sure the outputs are the same
    # en_batch_hf = ['Everything that is lost that is lost.', 'we must imagine Sisyphe happy.']
    # hfmodel_input_en_ids = tokenizer(en_batch_hf, return_tensors="pt", padding=True)['input_ids']
    # model = unwrapped_model['model']
    # output_fren_hf_model = model(input_ids=hfmodel_input_en_ids, decoder_input_ids=output_ids_fr) # attention_mask=model_input_fr_ids.ne(tokenizer.pad_token_id),
    # print('hf model en-fr teacherforcing on fr:')
    # print('y input:', tokenizer.batch_decode(hfmodel_input_en_ids, skip_special_tokens=False))
    # print('z output:', tokenizer.batch_decode(output_fren_hf_model['logits'].argmax(dim=-1), skip_special_tokens=False))
    # print('-'*50)
    
    # fr_batch_hf = ["Tout ce qui n'est pas sauvé sera perdu.", "on doit imaginer Sisyphus heureux."]
    # hfmodel_input_fr_ids = tokenizer(text_target=fr_batch_hf, return_tensors="pt", padding=True)['input_ids']
    # model = unwrapped_model['model']
    # output_enfr_hf_model = model(input_ids=hfmodel_input_fr_ids, decoder_input_ids=output_ids_en, output_attentions=True)
    # # attention_mask=model_input_en_ids.ne(tokenizer.pad_token_id),
    # print('hf model fr-en teacherforcing on en:')
    # print('y input:', tokenizer.batch_decode(hfmodel_input_fr_ids, skip_special_tokens=False))
    # print('z output:', tokenizer.batch_decode(output_enfr_hf_model['logits'].argmax(dim=-1), skip_special_tokens=False))
    # print('-'*50)



if __name__ == "__main__":
    main()
