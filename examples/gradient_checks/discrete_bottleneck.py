from blocks.modules.discrete_bottlenecks.softmax import SoftmaxDiscreteBottleneck
import torch
from torch.autograd import gradcheck


def test_gradients_torch_autograd():
    vocab_size = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    emb_dim = 5
    configs = {
        'dimensions': {
            'encoder_embedding_dim': emb_dim,
            'decoder_embedding_dim': emb_dim,
            'vocab_size': vocab_size,
            'unembedding_dim': vocab_size
        },
        'encoder_embedding_trainable': True,
        'decoder_embedding_trainable': True,
        'linear_head_trainable': True,
        'quantize_vector': True
    }
    model = SoftmaxDiscreteBottleneck(configs).double().to(device)
    def score_softmax(x):
        return model(x)['score']
    
    # Input for the module (should be of size suitable for gradcheck)
    test_input = torch.randn(batch_size, emb_dim, dtype=torch.double, requires_grad=True).to(device)
    model_input = model.linear_head(test_input)
    # Output of the module
    output = model(test_input)
    
    # Check gradients
    test_passed = gradcheck(score_softmax, (test_input,), eps=1e-6, atol=1e-4)
    print(f"Gradient check passed: {test_passed}")


def test_gradients_functionality():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 4
    batch_size = 1
    emb_dim = 5
    configs = {
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
    model = SoftmaxDiscreteBottleneck(configs).double().to(device)
    
    test_input = torch.randn(batch_size, emb_dim, dtype=torch.double, requires_grad=True).to(device)
    output_1 = model(test_input)
    output_2 = model(0.3*output_1['quantized_vector_decoder']+0.3*output_1['quantized_vector_encoder']+0.4*torch.randn_like(output_1['quantized_vector_decoder']))
    loss = torch.nn.functional.nll_loss(output_2['score'].log(), output_2['id'])
    print(f"Loss: {loss}")

    output_1['score'].retain_grad()
    output_1['quantized_vector_encoder'].retain_grad()
    output_1['quantized_vector_decoder'].retain_grad()
    output_2['score'].retain_grad()
    
    loss.backward()
    print(f"Gradient of the loss w.r.t. the input: {test_input.grad}")
    print(f"Gradient of the loss w.r.t. the encoder embedding: {output_1['quantized_vector_encoder'].grad}")
    print(f"Gradient of the loss w.r.t. the decoder embedding: {output_1['quantized_vector_decoder'].grad}")
    print(f"Gradient of the loss w.r.t. the output1 probabilities: {output_1['score'].grad}")
    print(f"Gradient of the loss w.r.t. the output2 probabilities: {output_2['score'].grad}")
    # output_2['score'].grad
    # tensor([[ 0.0000, -2.0221,  0.0000,  0.0000]], dtype=torch.float64)
    # output_1['score'].grad
    # tensor([[ 1.3486, -0.3745,  3.2421, -1.4203]], dtype=torch.float64)

if __name__ == '__main__':
    test_gradients_torch_autograd()
    test_gradients_functionality()