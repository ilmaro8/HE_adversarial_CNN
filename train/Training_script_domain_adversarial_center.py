import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.utils.data
from sklearn import metrics 
import os
import shutil
import sys, getopt
import warnings 
np.random.seed(0)
import argparse

warnings.filterwarnings("ignore")

argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("working on gpu")
else:
    device = torch.device("cpu")
    print("working on cpu")
print(torch.backends.cudnn.version())

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=32)
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='domain_center')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=True)
parser.add_argument('-i', '--input_folder', help='path of the folder where train.csv and valid.csv are stored',type=str, default='./partition/')
parser.add_argument('-o', '--output_folder', help='path where to store the model weights',type=str, default='./models/')


args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
TASK = args.TASK
EMBEDDING_bool = args.features
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder


def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory) 

print("CREATING/CHECKING DIRECTORIES")

create_dir(OUTPUT_FOLDER)

models_path = OUTPUT_FOLDER
checkpoint_path = models_path+'checkpoints_MIL/'
create_dir(checkpoint_path)

#path model file
model_weights_filename = models_path+'model.pt'

print("CSV LOADING ")
csv_folder = INPUT_FOLDER

csv_filename_training = csv_folder+'train.csv'
csv_filename_validation = csv_folder+'valid.csv'

#read data
train_dataset = pd.read_csv(csv_filename_training, sep=',', header=None).values#[:10]
valid_dataset = pd.read_csv(csv_filename_validation, sep=',', header=None).values#[:10]


n_centers = 2

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None, num_samples=None):
				
		# if indices is not provided, 
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset)))             if indices is None else indices
			
		# if num_samples is not provided, 
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)             if num_samples is None else num_samples
			
		# distribution of classes in the dataset 
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
				
		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
				   for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):
		return dataset[idx,1]
				
	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

#MODEL DEFINITION
pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', CNN_TO_USE, pretrained=imageNet_weights)

if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features

#reverse autograd
from torch.autograd import Function
class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

class domain_predictor(torch.nn.Module):
	def __init__(self, n_centers):
		super(domain_predictor, self).__init__()
		# domain predictor
		self.fc_feat_in = fc_input_features
		self.n_centers = n_centers

		if (EMBEDDING_bool==True):
			
			if ('resnet18' in CNN_TO_USE):
				self.E = 128

			elif ('resnet34' in CNN_TO_USE):
				self.E = 128

			elif ('resnet50' in CNN_TO_USE):
				self.E = 256
			elif ('densenet121' in CNN_TO_USE):
				self.E = 128

			self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
			self.domain_classifier = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)

	def forward(self, x):

		dropout = torch.nn.Dropout(p=0.2)
		#m_binary = torch.nn.Sigmoid()

		domain_emb = self.domain_embedding(x)
		domain_emb = dropout(domain_emb)
		domain_prob = self.domain_classifier(domain_emb)

		#domain_prob = m_binary(domain_prob)

		return domain_prob

class CNN_model(torch.nn.Module):
	def __init__(self):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(CNN_model, self).__init__()
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])

		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)
		
		self.fc_feat_in = fc_input_features
		self.N_CLASSES = 4

		if (EMBEDDING_bool==True):

			if ('resnet18' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			elif ('resnet34' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES
				#self.K = 1
			elif ('resnet50' in CNN_TO_USE):
				self.E = 256
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES
			elif ('densenet121' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			#self.embedding = siamese_model.embedding
			self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
			self.embedding_fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)

		else:
			self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)
			
			if ('resnet18' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet34' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.L = self.E
				self.D = 256
				self.K = self.N_CLASSES		
		
		self.domain_predictor = domain_predictor(n_centers)

	def forward(self, x, mode, alpha):
			"""
			In the forward function we accept a Tensor of input data and we must return
			a Tensor of output data. We can use Modules defined in the constructor as
			well as arbitrary operators on Tensors.
			"""
			#if used attention pooling
			A = None
			#m = torch.nn.Softmax(dim=1)
			m_binary = torch.nn.Sigmoid()
			m_multiclass = torch.nn.Softmax()
			dropout = torch.nn.Dropout(p=0.2)
			
			if x is not None:
				#print(x.shape)
				conv_layers_out=self.conv_layers(x)
				#print(x.shape)
				if ('densenet' in CNN_TO_USE):
					n = torch.nn.AdaptiveAvgPool2d((1,1))
					conv_layers_out = n(conv_layers_out)
				
				conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

			#print(conv_layers_out.shape)

			if ('mobilenet' in CNN_TO_USE):
				dropout = torch.nn.Dropout(p=0.2)
				conv_layers_out = dropout(conv_layers_out)
			#print(conv_layers_out.shape)

			if (EMBEDDING_bool==True):
				embedding_layer = self.embedding(conv_layers_out)
				features_to_return = embedding_layer

				embedding_layer = dropout(embedding_layer)
				logits = self.embedding_fc(embedding_layer)

			else:
				logits = self.fc(conv_layers_out)
				features_to_return = conv_layers_out

			output_fcn = m_multiclass(logits)

			if (mode=='train'):
				reverse_feature = ReverseLayerF.apply(conv_layers_out, alpha)

				output_domain = self.domain_predictor(reverse_feature)
				output_fcn = m_multiclass(logits)

				return logits, output_fcn, output_domain

			return logits, output_fcn


model = CNN_model()

#DATA AUGMENTATION
from torchvision import transforms
prob = 0.5

if (TASK=='no_augment'):

	pipeline_transform = A.Compose([
		A.VerticalFlip(p=prob),
		A.HorizontalFlip(p=prob),
		A.RandomRotate90(p=prob),
		#A.ElasticTransform(alpha=0.1,p=prob),
		#A.HueSaturationValue(hue_shift_limit=(-9),sat_shift_limit=25,val_shift_limit=10,p=prob),
		])

else:

	pipeline_transform = A.Compose([
		A.VerticalFlip(p=prob),
		A.HorizontalFlip(p=prob),
		A.RandomRotate90(p=prob),
		#A.ElasticTransform(alpha=0.1,p=prob),
		A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),p=prob),
		])


#DATA NORMALIZATION
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class Dataset_patches(data.Dataset):

	def __init__(self, list_IDs, labels, mode):

		self.labels = labels
		self.list_IDs = list_IDs
		self.mode = mode
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):

		# Select sample
		ID = self.list_IDs[index]
		# Load data and get label
		X = Image.open(ID)
		X = np.asarray(X)
		y = self.labels[index]
		#data augmentation

		domain = 0

		if (self.mode == 'train'):
			X = pipeline_transform(image=X)['image']

				#to modify with the classes
			if ('TMA' in ID):
				domain = 0
			elif ('slide' in ID):
				domain = 2
			else:
				domain = 1

			domain = np.asarray(domain)

		new_image = np.asarray(X)
		#data transformation
		input_tensor = preprocess(new_image)
				
		return input_tensor, np.asarray(y), domain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Parameters

num_workers = 2
params_train = {'batch_size': BATCH_SIZE,
		  #'shuffle': True,
		  'sampler': ImbalancedDatasetSampler(train_dataset),
		  'num_workers': num_workers}

params_valid = {'batch_size': BATCH_SIZE,
		  'shuffle': True,
		  #'sampler': ImbalancedDatasetSampler(valid_dataset),
		  'num_workers': num_workers}

params_test = {'batch_size': BATCH_SIZE,
		  'shuffle': True,
		  #'sampler': ImbalancedDatasetSampler(test_dataset),
		  'num_workers': num_workers}

max_epochs = int(EPOCHS_str)



# In[28]:


#CREATE GENERATORS
#train
training_set = Dataset_patches(train_dataset[:,0], train_dataset[:,1],'train')
training_generator = data.DataLoader(training_set, **params_train)

validation_set = Dataset_patches(valid_dataset[:,0], valid_dataset[:,1],'valid')
validation_generator = data.DataLoader(validation_set, **params_valid)

#semi-weakly supervision

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

class_sample_count = np.unique(train_dataset[:,1], return_counts=True)[1]
weight = class_sample_count / len(train_dataset[:,1])
#for avoiding propagation of fake benign class
samples_weight = torch.from_numpy(weight).type(torch.FloatTensor)

import torch.optim as optim
criterion = torch.nn.CrossEntropyLoss()
#criterion_domain = torch.nn.BCELoss()
criterion_domain = torch.nn.CrossEntropyLoss()

num_epochs = EPOCHS
epoch = 0
early_stop_cont = 0
EARLY_STOP_NUM = 8
optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
model.to(device)

lambda_val = 0.5 

def evaluate_validation_set(generator):
	#accumulator for validation set
	y_pred = []
	y_true = []

	valid_loss = 0.0

	with torch.no_grad():
		j = 0
		for inputs,labels,_ in generator:
			inputs, labels = inputs.to(device), labels.to(device)

			# forward + backward + optimize
			logits, outputs = model(inputs, None, None)

			loss = criterion(logits, labels)
			#outputs = F.softmax(outputs)

			valid_loss = valid_loss + ((1 / (j+1)) * (loss.item() - valid_loss)) 
			
			outputs_np = outputs.cpu().data.numpy()
			labels_np = labels.cpu().data.numpy()
			outputs_np = np.argmax(outputs_np, axis=1)

			y_true = np.append(y_true, outputs_np)
			y_pred = np.append(y_pred, labels_np)

			j = j+1			

		acc_valid = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
		kappa_valid = metrics.cohen_kappa_score(y1=y_true,y2=y_pred, weights='quadratic')
		print("loss: " + str(valid_loss) + ", accuracy: " + str(acc) + ", kappa score: " + str(kappa))
		
	return valid_loss
# In[35]:

best_loss_valid = 100000.0

losses_train = []
losses_valid = []


#p = float(i + epoch * iterations) / num_epochs / iterations

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
	
	y_true = []
	y_pred = []

	#loss functions outputs and network
	train_loss = 0.0
	train_loss_patches = 0.0
	train_loss_domain = 0.0
		
	is_best = False
	
	i = 0
	
	model.train()

	tot_iterations = int(len(train_dataset)/BATCH_SIZE)
	
	for inputs,labels, domains in training_generator:

		inputs, labels, domains = inputs.to(device), labels.to(device), domains.type(torch.LongTensor).to(device)
		
		p = float(i + epoch * tot_iterations) / num_epochs / tot_iterations

		alpha = 2. / (1. + np.exp(-10 * p)) - 1

		# zero the parameter gradients
		optimizer.zero_grad()
		
		# forward + backward + optimize
		logits, outputs, pred_domain = model(inputs, 'train', alpha)

		#print(logits, pred_domain)

		#pred_domain = pred_domain.view(-1)

		loss_patches = criterion(logits, labels)

		loss_domains = lambda_val * criterion_domain(pred_domain, domains)

		loss = loss_patches + loss_domains

		loss.backward()
		optimizer.step()
		
		train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))   
		train_loss_patches = train_loss_patches + ((1 / (i+1)) * (loss_patches.item() - train_loss_patches)) 
		train_loss_domain = train_loss_domain + ((1 / (i+1)) * (loss_domains.item() - train_loss_domain)) 

		#outputs = F.softmax(outputs)
		#accumulate values
		outputs_np = outputs.cpu().data.numpy()
		labels_np = labels.cpu().data.numpy()
		outputs_np = np.argmax(outputs_np, axis=1)

		y_true = np.append(y_true, outputs_np)
		y_pred = np.append(y_pred, labels_np)

		i = i+1

		if (i%100==0):

			print("loss: " + str(train_loss) + " loss patches: " + str(train_loss_patches) + " loss domains: " + str(train_loss_domain))

			print("["+str(i)+"/"+str(tot_iterations)+"]")
			acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
			kappa = metrics.cohen_kappa_score(y1=y_true,y2=y_pred, weights='quadratic')

			print("accuracy: " + str(acc))
			print("kappa score: " + str(kappa))
		
	model.eval()

	print("epoch "+str(epoch)+ " train loss: " + str(train_loss) + " acc_train: " + str(acc))
	
	print("evaluating validation")
	valid_loss = evaluate_validation_set(validation_generator)
	
	if (best_loss_valid>valid_loss):
		print ("=> Saving a new best model")
		print("previous loss TMA: " + str(best_loss_valid) + ", new loss function TMA: " + str(valid_loss))
		best_loss_valid = valid_loss
		torch.save(model, model_path)
		early_stop_cont = 0
	else:
		early_stop_cont = early_stop_cont+1
		
	epoch = epoch + 1
	
print('Finished Training')
