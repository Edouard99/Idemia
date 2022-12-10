import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import tqdm
import numpy as np

class dataset(Dataset):
    def __init__(self, data_0, data_1):
        self.samples = []
        self.transform=transforms.Compose([  transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomAffine(20,(0.12,0.12),(0.8,1.2),interpolation=transforms.InterpolationMode.NEAREST,fill=0),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        for img in data_0:
            img=transforms.ToTensor()(img)
            self.samples.append((img,0))
        for img in data_1:
            img=transforms.ToTensor()(img)
            self.samples.append((img,1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, id):
        (img,label)=self.samples[id]
        img=self.transform(img)
        return (img,label)


def lossfunction(x,y):
    num_0=torch.where(y==0)[0].shape[0]
    num_1=x.shape[0]-num_0
    w0=0.
    w1=0.
    if num_0!=0:
        w0=torch.divide(x.shape[0],(2*num_0),).float()
    if num_1!=0:
        w1=torch.divide(x.shape[0],(2*num_1),).float()
    w=torch.ones_like(x).float()
    w[torch.where(y==0)[0]]=w0
    w[torch.where(y==1)[0]]=w1
    return torch.nn.BCELoss(w)(x.float(),y.float())

def training(num_epochs,facenet,optimizer,scheduler,dataloader_t,dataloader_v,device,alpha1,alpha2,threshold,path):

    Loss_train_train=[]
    Loss_train_eval=[]
    Loss_val_eval=[]
    for epoch in tqdm.tqdm(range(num_epochs)):
        L_t_t=[]
        L_t_e=[]
        L_v_e=[]
        tp_t,fn_t,tn_t,fp_t=0,0,0,0
        tp_v,fn_v,tn_v,fp_v=0,0,0,0

        facenet.train()
        for i, dataj in (enumerate(dataloader_t, 0)):
            facenet.zero_grad()
            x=dataj[0].float().to(device)
            gt=dataj[1].float().to(device)
            y,aux1,aux2=facenet(x)
            loss=lossfunction(y.view(-1),gt)
            loss_aux1=lossfunction(aux1.view(-1),gt)
            loss_aux2=lossfunction(aux2.view(-1),gt)
            total_loss=(loss+alpha1*loss_aux1+alpha2*loss_aux2)/(1+alpha1+alpha2)
            total_loss.backward()
            optimizer.step()
            L_t_t.append([loss.item(),loss_aux1.item(),loss_aux2.item(),total_loss.item()])
            
        facenet.eval()
        for i, dataj in enumerate(dataloader_t, 0):
            x=dataj[0].float().to(device)
            gt=dataj[1].float().to(device)
            y=facenet(x).view(-1)
            loss=lossfunction(y,gt)
            L_t_e.append(loss.item())
            pred=y>threshold
            tp_t+=torch.sum(((pred)==gt)[torch.where((gt)==1)]).item()#TP
            fn_t+=torch.sum(((pred)!=gt)[torch.where((gt)==1)]).item()#FN
            tn_t+=torch.sum(((pred)==gt)[torch.where((gt)==0)]).item()#TN
            fp_t+=torch.sum(((pred)!=gt)[torch.where((gt)==0)]).item()#FP
        for i, dataj in enumerate(dataloader_v, 0):
            x=dataj[0].float().to(device)
            gt=dataj[1].float().to(device)
            y=facenet(x).view(-1)
            loss=lossfunction(y,gt)
            L_v_e.append(loss.item())
            pred=y>threshold
            tp_v+=torch.sum(((pred)==gt)[torch.where((gt)==1)]).item()#TP
            fn_v+=torch.sum(((pred)!=gt)[torch.where((gt)==1)]).item()#FN
            tn_v+=torch.sum(((pred)==gt)[torch.where((gt)==0)]).item()#TN
            fp_v+=torch.sum(((pred)!=gt)[torch.where((gt)==0)]).item()#FP
        far_v=fp_v/(tn_v+fp_v)
        frr_v=fn_v/(tp_v+fn_v)
        far_t=fp_t/(tn_t+fp_t)
        frr_t=fn_t/(tp_t+fn_t)
        hter_t=0.5*(far_t+frr_t)
        hter_v=0.5*(far_v+frr_v)
        scheduler.step()
        err_t_t=np.mean(L_t_t,0)
        err_t_e=np.mean(L_t_e,0)
        err_v_e=np.mean(L_v_e,0)
        Loss_train_train.append(err_t_t)
        Loss_train_eval.append(err_t_e)
        Loss_val_eval.append(err_v_e)
        print("Training : Loss {} ||| Hter {} \t Validation : Loss {} ||| Hter {}".format(err_t_e,hter_t,err_v_e,hter_v))
        torch.save({
            'epoch': epoch,
            'model_state_dict': facenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train_train': Loss_train_train,
            'loss_train_eval': Loss_train_eval,
            'loss_val_eval':Loss_val_eval,
            'metrics_train':{'tp':tp_t,'fn':fn_t,'fp':fp_t,'tn':tn_t,'far':far_t,'frr':frr_t,'hter':hter_t},
            'metrics_val':{'tp':tp_v,'fn':fn_v,'fp':fp_v,'tn':tn_v,'far':far_v,'frr':frr_v,'hter':hter_v}
            }, path+"checkpoint_{}.pth".format(epoch))