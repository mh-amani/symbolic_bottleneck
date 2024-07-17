import torch
import torch.nn as nn
import torch.optim as optim
from random import randrange
import numpy as np
import matplotlib.pyplot as plt

debug_print = False

######### ------------------ UTILS ------------------ #########

def create_quantized_image(dictionary, img_size=64):
    # Given a dictionary of shape (dic_dim, vocab_size),
    # generate an image of shape [1, dic_dim, img_size, img_size]
    # where each pixel has one of the dictionary elements
    # and image_ids of shape [1, 1, img_size, img_size]
    D, N = dictionary.shape
    dictionary = np.array(dictionary)
    image = np.ones((1, D, img_size, img_size))
    image_ids = np.zeros((1, 1, img_size, img_size), dtype=np.uint8)
    image[:, :, :, :] = dictionary[:, 0][np.newaxis, :, np.newaxis, np.newaxis]

    for i in range(50):
        size = randrange(3, img_size // 2)
        size = randrange(3, img_size // 2)
        id = randrange(1, N)
        center_x = randrange(img_size//2 - 4*img_size//5, img_size//2 + 4*img_size//5)
        center_y = randrange(img_size//2 - 4*img_size//5, img_size//2 + 4*img_size//5)
        image[:, :, center_x-size//2:center_x+size//2, center_y-size//2:center_y+size//2] = dictionary[:, id][np.newaxis, :, np.newaxis, np.newaxis]
        image_ids[:, :, center_x-size//2:center_x+size//2, center_y-size//2:center_y+size//2] = np.array([id])[np.newaxis, :, np.newaxis, np.newaxis]

    return image, image_ids

def visualize_mask_pred(image, mask, prediction, point=None, alpha=0.5, ):
    if not(isinstance(image, np.ndarray)):
        image = image.cpu().numpy()
        mask = mask.cpu().numpy()

    if image.shape[:1] != mask.shape[:1]:
        raise ValueError(f"Incompatible image/mask shapes: {image.shape} vs {mask.shape}") 
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title('Original Image')

    image_masked = np.zeros_like(image)
    image_masked[mask > 0] = np.array([[255, 0, 0]])
    
    axes[1].imshow(image_masked)
    axes[1].axis('off')
    axes[1].set_title('Image with Overlay')
    axes[1].set_title('GT mask')
    
    if point is not None:
        axes[0].plot(point[1], point[0], 'ro') # Swapped here!
        axes[0].set_title('Original Image with prompt')

    img_pred = np.zeros((prediction.shape[0], prediction.shape[1], 3))
    img_pred[:, :, 0] = prediction
    axes[2].imshow(img_pred)
    axes[2].axis('off')
    axes[2].set_title('Prediction Image')
    plt.show()

        
######### ------------------ MODEL ------------------ #########s
    
class TransformerEncoderModel(nn.Module):
    def __init__(self, vocab_dim):
        super(TransformerEncoderModel, self).__init__()

        self.enable_pos = True
        self.binary_pos = True
        if self.enable_pos:
            if not(self.binary_pos):
                max_position_embeddings = 32*32+1
            else:
                # try only two pos embeddings, one for prompt and one shared for all pixels
                # this way results should be the same per blob at least (kinda?)
                max_position_embeddings = 2 

            self.positional_embedding = nn.Embedding(max_position_embeddings, vocab_dim)
                
        encoder_layer = nn.TransformerEncoderLayer(d_model=vocab_dim, nhead=1, dim_feedforward=8, dropout=0.0)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.lin = nn.Linear(vocab_dim, 1)

    def forward(self, image, prompt_coord):
        prompt = image[prompt_coord[0], prompt_coord[1]].unsqueeze(0) # (1, C)
        if debug_print:
            print("VALUE", prompt)

        H, W, C = image.shape
        assert prompt.shape[1] == C
        assert prompt.shape[0] == 1

        image_flat = image.view(H*W, C)
        
        if self.enable_pos: # Generate positional embeddings
            if not(self.binary_pos):
                positions = torch.arange(0, H * W + 1).to("cuda")
                pos_embedding = self.positional_embedding(positions)  # (H*W+1, C)
                print("pos_embedding.shape", pos_embedding.shape)
                print("torch.cat((prompt, image_flat)).shape", torch.cat((prompt, image_flat)).shape) # (H*W+1, C)
                src = torch.cat((prompt, image_flat)) + pos_embedding # (H*W+1, C)
            else:
                
                positions = torch.tensor([0] + H*W * [1]).to("cuda")
                pos_embedding = self.positional_embedding(positions)  # (H*W+1s, C)
                src = torch.cat((prompt, image_flat)) + pos_embedding # (H*W+1, C)

        else:
            src = torch.cat((prompt, image_flat))
        
        out = self.transformer(src) # out of shape 1+HW, C
        out = out[1:] # (HW, C) discard first output corresponding to prompt
        out = self.lin(out) # (H*W, C) -> (H*W, 1) 
        out = torch.sigmoid(out) # (H*W, 1) 
        return out.view(H, W)


######### ------------------ TRAIN ------------------ #########
    
if __name__ == "__main__":
    criterion = nn.BCELoss() # /!\ Will throw some wrong CUBLAS error if predicted values are in the wrong range

    image_size = 8
    vocab_size = 4
    vocab_dim = 3
    dictionary = torch.rand((vocab_dim, vocab_size)).float().to("cuda")

    #model = TransformerModelCombined(vocab_dim, vocab_size).to("cuda")
    model = TransformerEncoderModel(vocab_dim).to("cuda")

    print(f"Number of parameters in model: {sum(p.numel() for p in model.parameters())}")

    parameters_to_optimize = list(model.parameters()) #+ [dictionary]
    optimizer = optim.Adam(parameters_to_optimize, lr=0.5)
    num_steps = 10000

    loss_hist = []
    for step in range(num_steps):
        ### --- Make data --- ###
        prompt_np = np.array([randrange(image_size), randrange(image_size)])
        prompt = torch.tensor(prompt_np).to("cuda")
        image, image_ids = create_quantized_image(dictionary.cpu().numpy(), image_size)
        image = torch.tensor(image).float().to("cuda") # (1, vocab_dim, image_size, image_size)
        image = image[0].permute(1, 2, 0) # (image_size, image_size, vocab_dim)
        
        ### --- Make GT mask --- ###
        image_ids = image_ids[0, 0]
        selected_id = image_ids[prompt[0], prompt[1]] 
        target_mask_np = image_ids == selected_id
        target_mask = torch.tensor(target_mask_np).float().to("cuda")

        ### --- PREDICT --- ###
        #print("IMAGE", image[:5,:5])
        output = model(image, prompt)
        
        ### --- OPTIM --- ###
        if debug_print:
            print("VALUE from id:", dictionary[:, selected_id])
            print("OUTPUT:", output[:5, :5])
            print("EXPECTED:",target_mask[:5, :5])
        loss = criterion(output, target_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

        print(f'Step {step+1}/{num_steps}, Loss: {loss.item()}')
        if step > 10 and (step) % 1000 == 0:
            img_np = image.detach().cpu().numpy()[:,:,:3]
            visualize_mask_pred(img_np, target_mask.detach().cpu().numpy(), prediction=output.detach().cpu().numpy(), point=prompt_np, alpha=1.0)
            
    window_size = 100
    smoothed_loss = np.convolve(loss_hist, np.ones(window_size)/window_size, mode='valid')
    plt.plot(loss_hist, label='Original')
    plt.plot(np.arange(window_size - 1, len(loss_hist)), smoothed_loss, label=f'Smoothed (Window Size={window_size})')
    plt.legend()
    plt.show()