from typing import Any, Dict, Tuple
import torch
import numpy as np
from torch.nn import ModuleDict, Module


class AutoRegWrapper(Module):
    """
    a wrapper connecting two sequence models with discrete bottleneck layers
    """
    def __init__(
        self,
        vector_model,
        input_discretizer,
        output_discretizer,
        config,
    ) -> None:

        super().__init__()
        self.config = config
        self.model = vector_model
        self.input_discretizer = input_discretizer
        self.output_discretizer = output_discretizer
        self.max_lengths = self.config['max_lengths']
        self.control_token_ids = self.config['control_token_ids']
        self.soft_average = self.config['soft_average']
        device=self.config['device']

        self.output_discretizer.to(device)
        self.input_discretizer.to(device)

        self.output_discretizer.encoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(self.output_discretizer.encoder_embedding.weight.device))
        self.output_discretizer.decoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(self.output_discretizer.encoder_embedding.weight.device))

        output_prepending_ids = self.config.get('output_prepending_ids', None)
        output_prepending_embeds_enc = self.config.get('output_prepending_embeds_enc', None)
        output_prepending_embeds_dec = self.config.get('output_prepending_embeds_dec', None)

        if output_prepending_ids is None and (output_prepending_embeds_enc is None or output_prepending_embeds_dec is None):
            raise ValueError("output_prepending_ids nor the embeddings are not provided")
        elif output_prepending_ids is None and (output_prepending_embeds_enc is not None and output_prepending_embeds_dec is not None):
            self.output_prepending_ids = self.control_token_ids['pad_token_id_y'] * np.ones(output_prepending_embeds_dec.shape[:2], dtype=np.long)
            self.output_prepending_embeds_enc = output_prepending_embeds_enc.numpy()
            self.output_prepending_embeds_dec = output_prepending_embeds_dec.numpy()
        else:
            self.output_prepending_ids = output_prepending_ids
            self.output_prepending_embeds_enc = self.output_discretizer.encoder_embedding_from_id(torch.from_numpy(np.array(self.output_prepending_ids)).to(device)).clone().detach().cpu().numpy()
            self.output_prepending_embeds_dec = self.output_discretizer.decoder_embedding_from_id(torch.from_numpy(np.array(self.output_prepending_ids)).to(device)).clone().detach().cpu().numpy()

    def forward(self, input_ids: torch.Tensor=None, input_attention_mask: torch.Tensor=None, input_embeds_enc: torch.Tensor=None,
                output_ids: torch.Tensor=None,
                output_embeds_enc: torch.Tensor=None, output_embeds_dec: torch.Tensor=None, output_attention_mask: torch.Tensor=None,
                teacher_force_output: bool=False, max_output_length=None) -> Dict[str, torch.Tensor]:
        """Perform a forward pass through the models x -> y_1 y_2 ... y_n

        """
        # assert that only one of the embeds or ids is provided, and attention mask should be provided if embeds are provided

        if max_output_length is None:
            max_output_length = self.max_lengths['output']
        
        assert (input_ids is not None) != (input_embeds_enc is not None), "Either input_ids or input_embeds should be provided"
        assert (input_embeds_enc is not None and input_attention_mask is not None) or (input_embeds_enc is None and input_attention_mask is None), "input_embeds and input_attention_mask should be provided together or not at all"
        assert (output_ids is None)  or (output_embeds_enc is None), "Either output_ids or output_embeds or neither should be provided, but not both"
        assert (output_embeds_enc is not None and output_embeds_dec is not None and output_attention_mask is not None) or (output_embeds_enc is None and output_embeds_dec is None and output_attention_mask is None), "output_embeds and output_attention_mask should be provided together or not at all"

        if input_ids is not None:
            input_embeds_enc = self.input_discretizer.encoder_embedding_from_id(input_ids)
            input_attention_mask = torch.logical_not(torch.eq(input_ids, self.control_token_ids['input_pad_token_id']))
        
        if output_ids is not None:
            # setting the output encoder and decoder embeddings and attention mask, starting generation from these embeddings
            output_embeds_enc = self.output_discretizer.encoder_embedding_from_id(output_ids)
            output_embeds_dec = self.output_discretizer.decoder_embedding_from_id(output_ids)
            output_attention_mask = torch.logical_not(torch.eq(output_ids, self.control_token_ids['output_pad_token_id']))
        elif output_embeds_enc is not None:
            # setting the output ids to unk tokens, as we don't have acces to the starting ids
            output_ids = self.control_token_ids['output_unknown_token_id'] * torch.ones(input_embeds_enc.shape[:2], dtype=torch.long).to(input_embeds_enc.device)
        else:
            # no output starting point is provided, so we start from the prepending embeddings
            output_ids = torch.from_numpy(np.array(self.output_prepending_ids)).repeat(input_embeds_enc.shape[0], 1).to(input_embeds_enc.device)
            output_embeds_enc = torch.tensor(self.output_prepending_embeds_enc).repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
            output_embeds_dec = torch.tensor(self.output_prepending_embeds_dec).repeat(input_embeds_enc.shape[0], 1, 1).to(input_embeds_enc.device)
            output_attention_mask = torch.ones(output_embeds_enc.shape[:2], dtype=torch.bool).to(output_embeds_enc.device)
        
        if not teacher_force_output:
            outputs = self._sequential_forward_from_embed(self.model, self.output_discretizer, 
                input_embeds=input_embeds_enc, input_attention_mask=input_attention_mask,
                output_embeds_enc=output_embeds_enc, output_embeds_dec=output_embeds_dec, 
                output_attention_mask=output_attention_mask,
                max_output_length=max_output_length)
        else:
            model_outputs = self.model.forward(inputs_embeds=input_embeds_enc, attention_mask=input_attention_mask,
                        decoder_inputs_embeds=output_embeds_dec, decoder_attention_mask=output_attention_mask,
                        output_hidden_states=True, output_attentions=True,)
            outputs = self.output_discretizer(model_outputs['last_hidden_state'], supervision=True,
                target_ids=output_ids, target_attention_mask=output_attention_mask, average_probs=self.soft_average['word_embeds_with_scores_forward'])
            # manually adding outputs that do not exist in the teacherforced forward pass
            outputs['score_list'] = []
            outputs['eos_flag'] = torch.any(torch.eq(outputs['id'], self.control_token_ids['output_eos_token_id']), dim=1).reshape(-1, 1)
            outputs['p_not_eos'] = 1 - outputs['score'][:, :, self.control_token_ids['output_eos_token_id']]
            outputs['output_attention_mask'] = output_attention_mask

        return {'id': outputs['id'], 'score': outputs['score'], 'score_list': outputs['score_list'], 'logit': outputs['logit'],
                'quantized_vector_encoder': outputs['quantized_vector_encoder'], 'quantized_vector_decoder': outputs['quantized_vector_decoder'],
                'output_attention_mask': outputs['output_attention_mask'], 'eos_flag': outputs['eos_flag'], 'p_not_eos': outputs['p_not_eos'],
                'quantization_loss': outputs['quantization_loss']}

    def _sequential_forward_from_embed(self, model, discretizer, input_embeds, input_attention_mask, 
        output_embeds_enc, output_embeds_dec, output_attention_mask,
        max_output_length):

        # initialize tensors
        quantization_loss = 0
        onehot_score_pad = torch.nn.functional.one_hot(torch.tensor(self.control_token_ids['output_pad_token_id']), num_classes=discretizer.vocab_size).to(output_embeds_enc.device).float()
        pad_embed_enc = self.output_discretizer.encoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(output_embeds_enc.device))
        pad_embed_dec = self.output_discretizer.decoder_embedding_from_id(torch.tensor(self.control_token_ids['output_pad_token_id']).to(output_embeds_enc.device))
        preprend_length = output_embeds_enc.shape[1] 
       
        ids = torch.ones(input_embeds.shape[0], max_output_length-preprend_length).to(input_embeds).int() * self.control_token_ids['output_pad_token_id']
        scores = torch.zeros(input_embeds.shape[0], max_output_length-preprend_length, discretizer.vocab_size).to(input_embeds) * onehot_score_pad
        logits = torch.zeros(input_embeds.shape[0], max_output_length-preprend_length, discretizer.vocab_size).to(input_embeds)
        # scores = torch.empty(input_embeds.shape[0], max_output_length-preprend_length, discretizer.vocab_size).to(input_embeds).fill_(0.0)
        logit_list = []
        scores_list = [] # this is only for visualizing the gradients. cause score matrix is a clone of scores, so gradaient propogation throught time doesn't showup in it. it is visible here.
        p_not_eoss = [torch.ones(input_embeds.shape[0], 1, requires_grad=True).to(input_embeds)]
        eos_flags = torch.zeros(input_embeds.shape[0], 1, dtype=torch.bool).to(input_embeds)
        output_embeds_encs = output_embeds_enc.requires_grad_(True)
        output_embeds_decs = output_embeds_dec.requires_grad_(True)
        output_attention_masks = output_attention_mask.float().requires_grad_(True)
        
        # scores.requires_grad = True
        # p_not_eoss.requires_grad = True
        # output_embeds_encs.requires_grad = True
        # output_embeds_decs.requires_grad = True
        # output_attention_masks.requires_grad = True

        step=0
        while step + preprend_length < max_output_length and not torch.all(eos_flags):

            if self.config['use_last_step_states'] and step > 0:
                last_step_states={'encoder_outputs':(current_output['encoder_last_hidden_state'], current_output['hidden_state'], 
                                                    current_output['encoder_attentions'])}
            else:
                last_step_states = {}
            # use the last hidden state of the encoder as the input to the decoder
            if self.config['use_past_key_values'] and step > 0:
                last_step_states['past_key_values'] = current_output['past_key_values'] # used to be torch.logical_not(eos_flag) for gpt2-gpt2,
            
            current_output = \
            self._one_step_sequential_forward_from_embed(model, discretizer, input_embeds, input_attention_mask,
                                                    output_embeds_decs[:, :step + preprend_length], output_attention_masks[:, :step + preprend_length], 
                                                    last_step_states, )

            ids[:, step] = current_output['id'].reshape(-1)
            scores_list.append(current_output['score'])
            scores[:, step] = current_output['score'][:, 0]
            logits[:, step] = current_output['logit'][:, 0]
            logit_list.append(current_output['logit'])
            p_not_eoss.append( (1 - current_output['p_eos']) * p_not_eoss[step-1])
            output_attention_masks = torch.cat((output_attention_masks, self.attention_mask(eos_flags, p_not_eoss[step])[:, 0].reshape(-1, 1)), dim=1)
            eos_flags = torch.logical_or(eos_flags, current_output['eos_flag'].reshape(-1, 1))
            quantization_loss += (current_output['quantization_loss'] * torch.logical_not(eos_flags).float())
            quantization_loss = quantization_loss * (current_output['quantization_loss'] * torch.logical_not(eos_flags))
            output_embeds_encs = torch.cat((output_embeds_encs, current_output['quantized_vector_encoder']), dim=1)
            output_embeds_decs = torch.cat((output_embeds_decs, current_output['quantized_vector_decoder']), dim=1)
            step = step + 1

        ids = ids[:, :step]
        scores = scores[:, :step]
        output_embeds_encs = output_embeds_encs[:, :step + preprend_length]
        output_embeds_decs = output_embeds_decs[:, :step + preprend_length]
        output_attention_masks = output_attention_masks[:, :step + preprend_length]

        # enforce pad tokens and embeds where attention mask is zero
        binary_attention_mask = output_attention_masks > 0
        ids = ids * binary_attention_mask[:, -ids.shape[1]:] + self.control_token_ids['output_pad_token_id'] * torch.logical_not(binary_attention_mask)[:, -ids.shape[1]:]
        scores = scores * output_attention_masks[:, -ids.shape[1]:].unsqueeze(-1) + \
            (1 - output_attention_masks[:, -ids.shape[1]:]).unsqueeze(-1) * onehot_score_pad
        # p_not_eoss = p_not_eoss * output_attention_masks[:, -ids.shape[1]:] + \
        #     (1 - output_attention_masks[:, -ids.shape[1]:]) * p_not_eoss[:, -1].reshape(-1, 1)
        output_embeds_encs = output_embeds_encs * output_attention_masks.unsqueeze(-1) + \
            pad_embed_enc * torch.logical_not(output_attention_masks).unsqueeze(-1)
        output_embeds_decs = output_embeds_decs * output_attention_masks.unsqueeze(-1) + \
            pad_embed_dec * torch.logical_not(output_attention_masks).unsqueeze(-1)

        
        return {
            'id': ids, 'score': scores, 'score_list': scores_list, 'logit': logits,
            'quantized_vector_encoder': output_embeds_encs, 'quantized_vector_decoder': output_embeds_decs,
            'quantization_loss': quantization_loss, 'output_attention_mask': output_attention_masks, 'eos_flag': eos_flags, 'p_not_eos': p_not_eoss
        }

    def _one_step_sequential_forward_from_embed(self, model, discretizer, input_embeds, input_attention_mask, 
                                    output_embeds, output_attention_mask, last_step_states={},):
        
        eos_token_id = self.control_token_ids['output_eos_token_id']
        # If you're using cached key values or encoder outputs or what not, you should pass them here. and you should check
        # the model source code to see if the gradients backpropagate through these values from next step or not.
        output = model(inputs_embeds=input_embeds, attention_mask=input_attention_mask,
                        decoder_inputs_embeds=output_embeds, decoder_attention_mask=output_attention_mask,
                        output_hidden_states=True, output_attentions=True, **last_step_states, use_cache=self.config['use_last_step_states'])
        
        # output_embed = output['decoder_hidden_states'][-1]
        output_embed = output['last_hidden_state'][:, -1:, :] # does the same thing as above size: (batch_size, 1, hidden_size)
        # output of the encoder to be used in generation, I don't get why the key values are needed though, only query values are useful
        # maybe using this is blocking the gradient flow, I should check this
        encoder_last_hidden_state = output.encoder_last_hidden_state
        past_key_values = output.past_key_values
        hidden_state = output.encoder_hidden_states
        encoder_attentions = output.encoder_attentions

        discretizer_output = discretizer(output_embed, supervision=False, average_probs=self.soft_average['word_embeds_with_scores_forward'])
        # discretizer_output.keys(): idx, score, logits, quantized_vector, quantization_loss
        
        current_eos_flag = (torch.eq(discretizer_output['id'][:, -1], eos_token_id))

        p_eos = discretizer_output['score'][:, :, eos_token_id]

        # idx, score, logits, quantized_vector, quantization_loss, current_eos_flag, p_eos, past_key_values, encoder_last_hidden_state, hidden_state, encoder_attentions
        return({'id': discretizer_output['id'], 'score': discretizer_output['score'], 'logit': discretizer_output['logit'], 
            'quantized_vector_encoder': discretizer_output['quantized_vector_encoder'], 'quantized_vector_decoder': discretizer_output['quantized_vector_decoder'], 
            'quantization_loss': discretizer_output['quantization_loss'], 'eos_flag': current_eos_flag, 'p_eos': p_eos, 
            'past_key_values': past_key_values, 'encoder_last_hidden_state': encoder_last_hidden_state, 
            'hidden_state': hidden_state, 'encoder_attentions': encoder_attentions})

    def attention_mask(self, eos_flags, p_not_eos):
        true_attention_mask = torch.logical_not(eos_flags)
        if self.config['soft_average']['p_eos_forward']:
            return p_not_eos
        elif self.config['soft_average']['p_eos_backward']:
            return true_attention_mask + p_not_eos - p_not_eos.detach()
        else:
            return true_attention_mask
        # what about eos_flags * p_not_eos?

