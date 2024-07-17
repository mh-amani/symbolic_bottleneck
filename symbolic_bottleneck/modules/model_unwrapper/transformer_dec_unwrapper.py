import torch
from typing import Optional
from blocks.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck

def DecoderUnwrapper(model):
    """
    Unwraps the encoder-decoder model to get the encoder and decoder weights.
    Args:
        dec_model: The encoder-decoder model.
    Returns:
        vector_model: The encoder-decoder model without embedding and head, pure transfomer.
        encoder_embedding_weight: The encoder weights.
        decoder_embedding_weight: The decoder weights.
        linearhead_weight: The linear head weights.
    """
    # Get the encoder and decoder weights
    vector_model = model.transformer
    encoder_embedding_weight = model.transformer.wte.weight
    decoder_embedding_weight = model.transformer.wte.weight
    linearhead_weight = model.lm_head.weight
    linearhead_bias = model.lm_head.bias
    return {'vector_model': vector_model, 'encoder_embedding_weight': encoder_embedding_weight, 
        'decoder_embedding_weight': decoder_embedding_weight, 'linearhead_weight': linearhead_weight, 'linearhead_bias': linearhead_bias} 

def UnwrappedGPT(config):
    """
    Unwraps the GPT model to get the encoder and decoder weights.
    Args:
        config: The configuration of the GPT model.
    Returns:
        vector_model: The GPT model without embedding and head, pure transfomer.
        encoder_embedding_weight: The encoder weights.
        decoder_embedding_weight: The decoder weights.
        linearhead_weight: The linear head weights.
    """
    # Get the encoder and decoder weights
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel(config)
    vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias = DecoderUnwrapper(model).values()
    return vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias

def UnwrappedPretrainedGPT():
    """
    Unwraps the GPT model to get the encoder and decoder weights.
    Returns:
        vector_model: The GPT model without embedding and head, pure transfomer.
        encoder_embedding_weight: The encoder weights.
        decoder_embedding_weight: The decoder weights.
        linearhead_weight: The linear head weights.
    """
    # Get the encoder and decoder weights
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias = DecoderUnwrapper(model).values()
    return vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias




def main() -> Optional[float]:
    # an example for the encoder-decoder MBART model:
    # get the models and the discretizers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    unwrapped_model = UnwrappedPretrainedGPT()
    vector_model, encoder_embedding_weight, decoder_embedding_weight, linearhead_weight, linearhead_bias = unwrapped_model
       
    sequence_en_1 = "Everything not saved will be lost."
    sequence_en_2 = "one must imagine Sisyphus happy."
    sequence_fr_1= "Tout ce qui n'est pas sauvé sera perdu." # "il m'a entarté"
    sequence_fr_2 = "il faut imaginer Sisyphe heureux."
    en_batch = [sequence_en_1, sequence_en_2]
    fr_batch = [sequence_fr_1, sequence_fr_2]
    en_input = tokenizer(en_batch, return_tensors="pt", padding=True,)
    fr_input = tokenizer(fr_batch, return_tensors="pt", padding=True,)
    
    gen_tokens_en = model.generate(
    **en_input,
    do_sample=True,
    temperature=0.9,
    max_length=100,
    )
    gen_text_en = tokenizer.batch_decode(gen_tokens_en)
    print(gen_text_en)
    gen_tokens_fr = model.generate(
    **fr_input,
    do_sample=True,
    temperature=0.9,
    max_length=100,
    )
    gen_text_fr = tokenizer.batch_decode(gen_tokens_fr)
    print(gen_text_fr)

    # outputs_en = model(**en_input, labels=en_input["input_ids"])
    # outputs_fr = model(**fr_input, labels=fr_input["input_ids"])
    
    
    # forward pass of one model
    input_vector_embeddings = en_discretizer.encoder_embedding_from_id(input_ids_enfr)
    output_vector_embeddings = fr_discretizer.decoder_embedding_from_id(output_ids_enfr) 
    output_vector_model = vector_model.forward(inputs_embeds=input_vector_embeddings, decoder_inputs_embeds=output_vector_embeddings,
                                            attention_mask=input_attention_mask_enfr, decoder_attention_mask=output_attention_mask_enfr,
                                            return_dict=True, output_hidden_states=True)
    discretized_output = fr_discretizer(output_vector_model['last_hidden_state'])
    # print the output of the discretizer discretized_output['id'], decoded with the tokenizer
    print('decoded output decomposed model:', tokenizer.batch_decode(discretized_output['id'], skip_special_tokens=False))
   
    # forward pass of the model without the discretizer (for comparison)
    # output_model = model_enfr(**input_enfr, labels=labels_enfr, return_dict=True, output_hidden_states=True)
    output_model = model(input_ids_enfr, attention_mask=input_attention_mask_enfr, 
                    decoder_input_ids=output_ids_enfr, decoder_attention_mask=output_attention_mask_enfr, return_dict=True)
    print('decoded output original model:', tokenizer.batch_decode(output_model.logits.argmax(dim=-1), skip_special_tokens=False))


if __name__ == "__main__":
    main()
