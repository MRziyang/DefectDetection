import torch.nn as nn 
class SELayer(nn.Module):
	'''由于对各个channel的卷积结果做了sum，所以channel特征关系与卷积核学习到的空间关系混合在一起,而SE模块就是为了抽离这种混杂，使得模型直接学习到channel特征关系。'''
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)
		# self.fc = nn.Sequential(
		#     nn.Conv2d(channel, channel // reduction,kernel_size=1,stride=1),
		#     nn.ReLU(inplace=True),
		#     nn.Conv2d(channel // reduction, channel,kernel_size=1,stride=1),
		#     nn.Sigmoid()
		# )

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y.expand_as(x)