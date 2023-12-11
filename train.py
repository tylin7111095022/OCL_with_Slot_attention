import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader,random_split #random_split幫助切割dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

#custom module
# from metric import iou, compute_mIoU
from model import SlotAttentionAutoEncoder
from dataset import ChromosomeDataset

dir_img = r'F:\2023\chromosomes\chromosome_data\old_chang\images' #訓練集的圖片所在路徑
dir_truth = None #訓練集的真實label所在路徑

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    # parser.add_argument('--image_channel','-i',type=int, default=3,dest='in_channel',help="channels of input images")
    parser.add_argument('--num_slot',type=int,default=2,help='number of slot in a picture')
    parser.add_argument('--num_iter',type=int,default=3,help="Number of attention iterations.")
    parser.add_argument('--imgsize',type=int,default=128,help='img size of dataset')
    parser.add_argument('--total_epoch',type=int,default=500,metavar='E',help='times of training model')
    parser.add_argument('--warm_up_epoch',type=int,default=150,help='warm up epoch')
    parser.add_argument('--batch',type=int,default=64, help='Batch size')
    parser.add_argument("--save_every_iter",type=int, default=500)
    parser.add_argument('--rate_of_learning','-r',type = float, dest='lr', default=4e-4,help='learning rate of model')
    parser.add_argument('--log_dir', type=str,default='log/log4',help='filename of log')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')

    return parser.parse_args()

def main():
    args = get_args()
    #設置 log
    # ref: https://shengyu7697.github.io/python-logging/
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir,exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(args.log_dir,"log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    ###################################################
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    model = SlotAttentionAutoEncoder(resolution=(args.imgsize,args.imgsize),num_slots=args.num_slot,num_iterations=args.num_iter,device=device)
    trainingDataset = ChromosomeDataset(img_dir = dir_img, mask_dir= dir_truth,imgsize=args.imgsize)
    logging.info(model)
    
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    Parameters of training:
    data path: {dir_img}
    label path: {dir_truth}
    {args}
    =======================================
    ''')
    record = train(args,model=model,dataset=trainingDataset,device=device)

    fig, ax = plt.subplots(2,1,figsize=(16,8))
    for i,k in enumerate(record.keys()):
        ax[i].plot(record[k],label=k)
    ax[i].legend()
    plt.show()       

    return

def train(args, model, dataset, device):
    rec_loss_fn = torch.nn.MSELoss()
    dataloader = DataLoader(dataset=dataset,batch_size=args.batch, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr,eps=1e-8)
    iter = 0
    epoch_losses = []
    lrs = []
    writer = SummaryWriter()

    model = model.to(device)
    for epoch in range(1,args.total_epoch):
        loss_every_epoch = 0.0
        if epoch <= args.warm_up_epoch:
            lr = (args.lr * epoch) / args.total_epoch
        else:
            lr = args.lr

        lr = lr * ((args.total_epoch- epoch) / args.total_epoch)
        writer.add_scalar("lr",lr, epoch)
        
        for param in optimizer.param_groups:
            param["lr"] = lr
            
        for img, _ in tqdm(dataloader):
            img = img.to(device)
            model.train()
            iter += 1
            preds = model(img)
            recon_combined, recons, masks, slots = preds
            loss = rec_loss_fn(img, recon_combined)
            writer.add_scalar("loss/reconcruct_loss",(loss.item()), iter)
            loss_every_epoch += loss.item()
            del recons, masks, slots  # Unused.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (iter % args.save_every_iter) == 0:
                torch.save(model.state_dict(),os.path.join(args.log_dir,f"slot_autoencoder_{epoch}_{iter % len(dataloader)}.pth"))

        epoch_losses.append(loss_every_epoch)
        lrs.append(lr)

    record = {"epoch_loss":epoch_losses,"lr":lrs}
    writer.flush()
    writer.close
    
    return record

if __name__ == '__main__':
    main()