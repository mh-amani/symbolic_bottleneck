import torch
from transformers import BartForConditionalGeneration, BartConfig
from blocks.modules.auto_reg_wrapper import AutoRegWrapper
from blocks.unwrapped_models.enc_dec_unwrapper import EncoderDecoderUnwrapper
from blocks.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck


device='cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 4
batch_size = 1
seq_length = 4
emb_dim = 8
config_bart ={
    'vocab_size': vocab_size,
    'max_position_embeddings': seq_length,
    'encoder_layers': 1,
    'encoder_ffn_dim': emb_dim,
    'encoder_attention_heads': 1,
    'decoder_layers': 1,
    'decoder_ffn_dim': emb_dim,
    'decoder_attention_heads': 1,
    'd_model': emb_dim,
    'use_cache': True,
    'torch_dtype': 'float32'
}
discretizer_config = {
    device: device,
    'dimensions': {
        'encoder_embedding_dim': emb_dim,
        'decoder_embedding_dim': emb_dim,
        'vocab_size': vocab_size,
        'unembedding_dim': vocab_size
    },
    'encoder_embedding_trainable': True,
    'decoder_embedding_trainable': True,
    'linear_head_trainable': True,
    'quantize_vector': True,
    'temperature': 0.1 # Low temperature should make softmax near one-hot
}


def test_gradients_no_concat():
    # fix seed
    torch.manual_seed(45)

    model = BartForConditionalGeneration(BartConfig(**config_bart)).to(device)
    vector_model, encoder_embedding, decoder_embedding, linearhead = EncoderDecoderUnwrapper(model).values()
    discretizer = SoftmaxDiscreteBottleneck({**discretizer_config, 'encoder_embedding': encoder_embedding,
                                             'decoder_embedding': decoder_embedding, 'linear_head': linearhead,}).to(device)
    

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids[:, 0] = 0

    # Forward pass
    input_embeds_0 = discretizer.encoder_embedding_from_id(input_ids)
    output_embeds_0 = discretizer.encoder_embedding_from_id(output_ids[:, 0:1])
    # input_attention_mask_0 = torch.ones(output_embeds_0.shape[:2]).to(device)
    output_vector_model_1 = vector_model(inputs_embeds=input_embeds_0, # decoder_attention_mask=input_attention_mask_0,  #doesn't make a difference
                                         decoder_inputs_embeds=output_embeds_0,)['last_hidden_state']
    discrete_output_1 = discretizer(output_vector_model_1)
    discrete_output_1_scores = discrete_output_1['score']
    discrete_output_1_encoder_embeds = discrete_output_1['quantized_vector_encoder']
    discrete_output_1_decoder_embeds = discrete_output_1['quantized_vector_decoder']
    output_embeds_1 = torch.cat((output_embeds_0, discrete_output_1_decoder_embeds), dim=1)
    input_attention_mask_1 = torch.ones(output_embeds_1.shape[:2]).to(device)
    output_vector_model_2 = vector_model(inputs_embeds=input_embeds_0, # decoder_attention_mask=input_attention_mask_1,  #doesn't make a difference
                                         decoder_inputs_embeds=output_embeds_1, )['last_hidden_state']
    discrete_output_2 = discretizer(output_vector_model_2)
    discrete_output_2_scores = discrete_output_2['score']

    # torch.nn.functional.cross_entropy(discrete_output_2_scores[0, 1:], output_ids[:, 1].view(-1))
    loss = torch.nn.functional.nll_loss(torch.log(discrete_output_2_scores[0, 1:]), output_ids[:, 1].view(-1))
    
    # retain_grad
    input_embeds_0.retain_grad()
    discrete_output_1_scores.retain_grad()
    discrete_output_1_encoder_embeds.retain_grad()
    discrete_output_1_decoder_embeds.retain_grad()
    discrete_output_2_scores.retain_grad()

    loss.backward()
    print(f"Gradient of the loss w.r.t. the input: {input_embeds_0.grad}")
    print(f"Gradient of the loss w.r.t. the encoder embedding: {discrete_output_1_encoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the decoder embedding: {discrete_output_1_decoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the output1 probabilities: {discrete_output_1_scores.grad}")
    print(f"Gradient of the loss w.r.t. the output2 probabilities: {discrete_output_2_scores.grad}")
    print(f"Loss: {loss}")


def clone_test():
    vector_a = torch.rand(2, 2, 3, requires_grad=True)
    vector_b = torch.rand(2, 2, 3,  requires_grad=True)
    vector_c = vector_a * vector_b
    # vector_d = vector_c.clone()
    vector_d = torch.cat([vector_c[i].unsqueeze(0) for i in range(vector_c.shape[0])], dim=0)
    print(vector_c.shape, vector_d.shape)
    loss = torch.nn.functional.mse_loss(vector_d, torch.zeros(2, 2, 3))
    vector_d.retain_grad()
    vector_c.retain_grad()
    loss.backward()
    print("vector_a.grad", vector_a.grad)
    print("vector_b.grad", vector_b.grad)
    print("vector_c.grad", vector_c.grad)
    print("vector_d.grad", vector_d.grad)



def test_gradients_torchcat():
    # fix seed
    torch.manual_seed(45)

    model = BartForConditionalGeneration(BartConfig(**config_bart)).to(device)
    vector_model, encoder_embedding, decoder_embedding, linearhead = EncoderDecoderUnwrapper(model).values()
    discretizer = SoftmaxDiscreteBottleneck({**discretizer_config, 'encoder_embedding': encoder_embedding,
                                             'decoder_embedding': decoder_embedding, 'linear_head': linearhead,}).to(device)
    

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    output_ids[:, 0] = 0

    # Forward pass
    input_embeds = discretizer.encoder_embedding_from_id(input_ids)
    output_embeds = discretizer.encoder_embedding_from_id(output_ids[:, 0:1])
    # input_attention_mask = torch.ones(output_embeds.shape[:2]).to(device)
    
    output_vector_model = vector_model(inputs_embeds=input_embeds, # decoder_attention_mask=input_attention_mask_0,  #doesn't make a difference
                                         decoder_inputs_embeds=output_embeds,)['last_hidden_state']
    discrete_output = discretizer(output_vector_model)
    scores = discrete_output['score']
    discrete_output_encoder_embeds = discrete_output['quantized_vector_encoder']
    discrete_output_decoder_embeds = discrete_output['quantized_vector_decoder']

    output_embeds = torch.cat((output_embeds, discrete_output_decoder_embeds), dim=1)
    output_vector_model = vector_model(inputs_embeds=input_embeds, # decoder_attention_mask=input_attention_mask_1,  #doesn't make a difference
                                         decoder_inputs_embeds=output_embeds, )['last_hidden_state'][:, -1:]
    discrete_output = discretizer(output_vector_model)
    scores = torch.cat((scores, discrete_output['score']), dim=1)


    # torch.nn.functional.cross_entropy(discrete_output_2_scores[0, 1:], output_ids[:, 1].view(-1))
    loss = torch.nn.functional.nll_loss(torch.log(scores[0, 1:]), output_ids[:, 1].view(-1))
    
    # retain_grad
    input_embeds.retain_grad()
    scores.retain_grad()
    discrete_output_encoder_embeds.retain_grad()
    discrete_output_decoder_embeds.retain_grad()


    loss.backward()
    print(f"Gradient of the loss w.r.t. the input: {input_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the encoder embedding: {discrete_output_encoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the decoder embedding: {discrete_output_decoder_embeds.grad}")
    print(f"Gradient of the loss w.r.t. the output1 probabilities: {scores.grad}")

    print(f"Loss: {loss}")



if __name__ == '__main__':
    print('--- Test gradients no concat ---')
    test_gradients_no_concat()
    print('--- Test gradients torchcat ---')
    test_gradients_torchcat()
    clone_test()