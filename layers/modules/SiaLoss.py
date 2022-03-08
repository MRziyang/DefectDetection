import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class DContrastiveLoss(nn.Module):
	def __init__(self,margin1=.4,margin2=2.8,eps=1e-6):
		super(DContrastiveLoss,self).__init__()
		self.margin1=margin1
		self.margin2=margin2
		self.eps=eps
	
	def forward(self,x1,x2,label):
		diff=torch.abs(x1-x2)
		dist_sq=torch.pow(diff+self.eps,2).sum(dim=1)
		dist=torch.sqrt(dist_sq)

		mdist_pos=torch.clamp(dist-self.margin1,min=0.0)
		mdist_neg=torch.clamp(self.margin2-dist,min=0.0)
		loss_pos=(1-label)*(mdist_pos.pow(2))
		loss_neg=label*(mdist_neg.pow(2))
		loss=torch.mean(loss_pos+loss_neg)
		return loss 

class ContractiveLoss(nn.Module):
	def __init__(self,margin=2.0,dist_flag='l2'):
		super(ContractiveLoss,self).__init__()
		self.margin=margin
		self.dist_flag=dist_flag
	
	def various_distance(self,out_vec_t0,out_vec_t1):
		if self.dist_flag=='l2':
			distance=F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
		if self.dist_flag=='l1':
			distance=F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
		if self.dist_flag=='cos':
			similarity=F.cosine_similarity(out_vec_t0,out_vec_t1)
			distance=1-2*similarity/np.pi
		return distance 
	
	def forward(self,out_vec_t0,out_vec_t1,label):
		distance=self.various_distance(out_vec_t0,out_vec_t1)
		h=label*distance
		h=torch.sum(h)
		constractive_loss=torch.sum((1-label)*torch.pow(distance,2)+label*torch.pow(torch.clamp(self.margin-distance,min=0.0),2))
		return constractive_loss


class FocalDContrastiveLoss(nn.Module):
	def __init__(self,margin1=.4,margin2=2.8,eps=1e-6,alpha=0.25,gamma=2):
		super(FocalDContrastiveLoss,self).__init__()
		self.margin1=margin1
		self.margin2=margin2
		self.eps=eps
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self,x1,x2,label):
		diff=torch.abs(x1-x2)
		dist_sq=torch.pow(diff+self.eps,2).sum(dim=1)
		dist=torch.sqrt(dist_sq)

		mdist_pos=torch.clamp(dist-self.margin1,min=0.0)
		mdist_neg=torch.clamp(self.margin2-dist,min=0.0)
		loss_pos=(1-label)*((self.alpha*(mdist_pos.pow(2)))**self.gamma)
		#当某样本类别比较明确些，它对整体loss的贡献就比较少；而若某样本类别不易区分，则对整体loss的贡献就相对偏大。这样得到的loss最终将集中精力去诱导模型去努力分辨那些难分的目标类别，于是就有效提升了整体的目标检测准度。
		loss_neg=label*(mdist_neg.pow(2))
		loss=torch.mean(loss_pos+loss_neg)
		return loss 