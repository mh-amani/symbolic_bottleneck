from .abstract_discrete_layer import AbstractDiscreteLayer
import torch
import math

# example config for SoftmaxDiscreteBottleneck
# config = {
#     '_target_': 'blocks.modules.discrete_bottlenecks.softmax.SoftmaxDiscreteBottleneck',
#     'dimensions': {'decoder_embedding_dim': embed_dim, 'vocab_size': vocab_size, 
#                     'encoder_embedding_dim': embed_dim, 'unembedding_dim': vocab_size}, 
#     'quantize_vector': True, 'temperature': 1.0,
#     'encoder_embedding_trainable': False, 'decoder_embedding_trainable': False, 
#     'linear_head_trainable': False, 
#     'encoder_embedding': encoder_embedding, 'decoder_embedding': decoder_embedding, 
#     'linear_head': linear_head,
#     }


class SoftmaxDiscreteBottleneck(AbstractDiscreteLayer):
    def __init__(self, configs) -> None:
        super().__init__(configs)

        # a probability based discretizer requires the following assertions to hold
        assert self.linear_head.in_features == self.decoder_embedding_dim
        # assert self.linear_head.out_features == self.vocab_size
        assert self.encoder_embedding_dim == self.decoder_embedding_dim

        # self.logit_std = math.sqrt(self.output_dim * self.out_std**2)
        # self.logit_init = math.log(self.dictionary_dim)

    def discretize(self, x,**kwargs) -> dict:
        # the function that takes the output of the decoder and returns a discrete representation
        logits= x/self.temperature
        score = torch.softmax(logits, dim=-1)
        idx = torch.argmax(score, dim=-1)
        if self.quantize_vector:
            quantized_vector_encoder = self.encoder_embedding_from_id(idx) + torch.matmul(score, self.encoder_embedding.weight) - \
                    torch.matmul(score, self.encoder_embedding.weight).detach()
            quantized_vector_decoder = self.decoder_embedding_from_id(idx) + torch.matmul(score, self.decoder_embedding.weight) - \
                    torch.matmul(score, self.decoder_embedding.weight).detach()
        elif not self.quantize_vector:
            quantized_vector_encoder = torch.matmul(score, self.encoder_embedding.weight)
            quantized_vector_decoder = torch.matmul(score, self.decoder_embedding.weight)
        quantization_loss = torch.tensor(0.0).to(x)
        return {"id": idx, "score": score, "logit": logits, "quantized_vector_encoder": quantized_vector_encoder, 
                "quantized_vector_decoder": quantized_vector_decoder, "quantization_loss": quantization_loss}
