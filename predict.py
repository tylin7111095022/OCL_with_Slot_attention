from torch.utils.data import DataLoader
from model import SlotAttentionAutoEncoder
from dataset import ChromosomeDataset
import matplotlib.pyplot as plt
import torch
import numpy as np

def main():
    model = SlotAttentionAutoEncoder(resolution=(128,128),num_slots=2,num_iterations=3, device=torch.device("cpu"))
    dataset = ChromosomeDataset(img_dir = r"data\zong\10000\val2017", mask_dir= r"data\zong\10000\val_mask",imgsize=128)
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    loader_iter = loader.__iter__()
    plotter = Ploter(model=model)
    plotter.load_weight(r"log\log2\slot_autoencoder_68_0.pth")

    while True:
        try:
            imgs, labels = next(loader_iter)
            plotter.visualize_slot(imgs, idx=0)
        except StopIteration:
           print("plot over")
           break
    return
        
    
      
def renormalize(x):
  """Renormalize from [-1, 1] to [0, 1]."""
  return x / 2. + 0.5

def get_prediction(model, imgs, idx=0):
    with torch.no_grad():
        recon_combined, recons, masks, slots = model(imgs)
        image = renormalize(imgs)[idx]
        recon_combined = renormalize(recon_combined)[idx]
        recons = renormalize(recons)[idx]
        masks = masks[idx]
    recon_combined = recon_combined.permute(1,2,0).numpy()
    image = image.permute(1,2,0).numpy()
    recons = recons.permute(0,2,3,1).numpy()
    masks = masks.permute(0,2,3,1).numpy()
    slots =  slots.numpy()
    return image, recon_combined, recons, masks, slots

class Ploter(object):
    def __init__(self,model):
        model.eval()
        self.plot_time = 0
        self.model = model

    def load_weight(self, weight_path:str):
        self.model.load_state_dict(torch.load(weight_path))

    def visualize_slot(self, batch, idx=0):
        self.plot_time += 1
        image, recon_combined, recons, masks, slots = get_prediction(self.model, batch,idx)
        # print(np.unique(recon_combined))
        num_slots = len(masks)
        fig, ax = plt.subplots(1, num_slots + 2, figsize=(12, 2))
        ax[0].imshow(image)
        ax[0].set_title('Image')
        ax[1].imshow(recon_combined)
        ax[1].set_title('Recon.')
        for i in range(num_slots):
            ax[i + 2].imshow(recons[i] * masks[i] + (1 - masks[i]))
            ax[i + 2].set_title('Slot %s' % str(i + 1))
        for i in range(len(ax)):
            ax[i].grid(False)
            ax[i].axis('off')
        fig.savefig(f"{self.plot_time}.png")

if __name__ == "__main__":
   main()