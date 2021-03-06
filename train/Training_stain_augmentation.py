import sys
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
import pickle
import argparse
from scipy.spatial import KDTree, cKDTree

warnings.filterwarnings("ignore")

argv = sys.argv[1:]

#torch.multiprocessing.set_start_method('spawn')

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
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='stain_normalization')
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


imageNet_weights = True

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

class CNN_model(torch.nn.Module):
	def __init__(self):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(CNN_model, self).__init__()
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		"""
		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)
		"""
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
				self.L = fc_input_features
				self.D = 256
				self.K = self.N_CLASSES	

			elif ('densenet121' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 64
				self.K = self.N_CLASSES
	

	def forward(self, x, conv_layers_out):
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
			
			return logits, output_fcn 


model = CNN_model()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

#DATA AUGMENTATION
from torchvision import transforms
prob = 0.5

pipeline_transform = A.Compose([
		A.VerticalFlip(p=prob),
		A.HorizontalFlip(p=prob),
		A.RandomRotate90(p=prob),
		#A.ElasticTransform(alpha=0.1,p=prob),
		#A.HueSaturationValue(hue_shift_limit=(-9),sat_shift_limit=25,val_shift_limit=10,p=prob),
		])

#DATA NORMALIZATION
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def H_E_Staining(img, Io=240, alpha=1, beta=0.15):

	# define height and width of image
	h, w, c = img.shape

	# reshape image
	img = img.reshape((-1,3))

	# calculate optical density
	OD = -np.log((img.astype(np.float)+1)/Io)

	# remove transparent pixels
	ODhat = OD[~np.any(OD<beta, axis=1)]

	# compute eigenvectors
	eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

	#eigvecs *= -1

	#project on the plane spanned by the eigenvectors corresponding to the two 
	# largest eigenvalues    
	That = ODhat.dot(eigvecs[:,1:3])

	phi = np.arctan2(That[:,1],That[:,0])

	minPhi = np.percentile(phi, alpha)
	maxPhi = np.percentile(phi, 100-alpha)

	vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
	vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

	# a heuristic to make the vector corresponding to hematoxylin first and the 
	# one corresponding to eosin second
	if vMin[0] > vMax[0]:
		HE = np.array((vMin[:,0], vMax[:,0])).T
	else:
		HE = np.array((vMax[:,0], vMin[:,0])).T

	return HE

def unique_elements(array):
	
	unique, counts = np.unique(array, return_counts=True)
	
	b = True
	
	for c in counts:
		
		if (c>1):
			
			b = False
	
	return b

def normalizeStaining(img, HERef, Io=240, alpha=1, beta=0.15):
	''' 
	Normalize staining appearence of H&E stained images
	
	Example use:
		see test.py
		
	Input:
		I: RGB input image
		Io: (optional) transmitted light intensity
		
	Output:
		Inorm: normalized image
		H: hematoxylin image
		E: eosin image
	
	Reference: 
		A method for normalizing histology slides for quantitative analysis. M.
		Macenko et al., ISBI 2009
	'''

	maxCRef = np.array([1.9705, 1.0308])
	
	# define height and width of image
	h, w, c = img.shape
	
	# reshape image
	img = img.reshape((-1,3))

	# calculate optical density
	OD = -np.log((img.astype(np.float)+1)/Io)
	
	# remove transparent pixels
	ODhat = OD[~np.any(OD<beta, axis=1)]
		
	# compute eigenvectors
	eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
	
	#eigvecs *= -1
	
	#project on the plane spanned by the eigenvectors corresponding to the two 
	# largest eigenvalues    
	That = ODhat.dot(eigvecs[:,1:3])
	
	phi = np.arctan2(That[:,1],That[:,0])
	
	minPhi = np.percentile(phi, alpha)
	maxPhi = np.percentile(phi, 100-alpha)
	
	vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
	vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
	
	# a heuristic to make the vector corresponding to hematoxylin first and the 
	# one corresponding to eosin second
	if vMin[0] > vMax[0]:
		HE = np.array((vMin[:,0], vMax[:,0])).T
	else:
		HE = np.array((vMax[:,0], vMin[:,0])).T
	
	# rows correspond to channels (RGB), columns to OD values
	Y = np.reshape(OD, (-1, 3)).T
	
	# determine concentrations of the individual stains
	C = np.linalg.lstsq(HE,Y, rcond=None)[0]
	
	# normalize stain concentrations
	maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
	tmp = np.divide(maxC,maxCRef)
	C2 = np.divide(C,tmp[:, np.newaxis])
	
	# recreate the image using reference mixing matrix
	Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
	Inorm[Inorm>255] = 254
	Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
	"""
	# unmix hematoxylin and eosin
	H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
	H[H>255] = 254
	H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
	
	E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
	E[E>255] = 254
	E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
	"""
	return Inorm

def stain_augmentation(patch_np, sigma1, sigma2):

    b = False
    i = 0

    HE_ref = H_E_Staining(patch_np)
    #print(HE_ref)
    #print()

    HE_ref = H_E_Staining(patch_np)

    alpha_sig = np.random.uniform(1 - sigma1, 1 + sigma1)
    beta_sig = np.random.uniform(-sigma2, sigma2)

    HE_ref *= alpha_sig 
    HE_ref += beta_sig

    A_np = normalizeStaining(patch_np, HE_ref)
    #print(i)
    #print(HE_ref)
    return A_np

if (TASK=='stain_normalization'):

	target_img_fname = train_dataset[0,0]
	
	img_target = Image.open(target_img_fname)
	img_target_np = np.asarray(img_target)

	HE_target = H_E_Staining(img_target_np)

sigma1 = 0.2
sigma2 = 0.2

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

		if (self.mode == 'train'):
			X = pipeline_transform(image=X)['image']

			rand_val = np.random.rand(1)[0]

			if (TASK=='stain_augmentation' and rand_val>prob):
				
				X = stain_augmentation(X, sigma1, sigma2)


		if (TASK=='stain_normalization'):
				
			X = normalizeStaining(X, HE_target)

		new_image = np.asarray(X)
		#data transformation
		input_tensor = preprocess(new_image)
				
		return input_tensor, np.asarray(y)


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

num_epochs = EPOCHS
epoch = 0
early_stop_cont = 0
EARLY_STOP_NUM = 5
optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
model.to(device)

def evaluate_validation_set(generator):
	#accumulator for validation set
	y_pred = []
	y_true = []

	valid_loss = 0.0

	with torch.no_grad():
		j = 0
		for inputs,labels in generator:
			inputs, labels = inputs.to(device), labels.to(device)

			# forward + backward + optimize
			logits, outputs = model(inputs, None)

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
		print("loss: " + str(valid_loss) + ", accuracy: " + str(acc_valid) + ", kappa score: " + str(kappa_valid))
		
	return valid_loss
# In[35]:

best_loss_valid = 100000.0

losses_train = []
losses_valid = []


while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
	
	y_true = []
	y_pred = []

	#loss functions outputs and network
	train_loss = 0.0
		
	is_best = False
	
	i = 0
	
	model.train()

	tot_iterations = int(len(train_dataset)/BATCH_SIZE)
	
	for inputs,labels in training_generator:
		inputs, labels = inputs.to(device), labels.to(device)
		
		# zero the parameter gradients
		optimizer.zero_grad()
		
		# forward + backward + optimize
		logits, outputs = model(inputs, None)
		#print(logits.shape,labels.shape)
		loss = criterion(logits, labels)

		loss.backward()
		optimizer.step()
		
		train_loss = train_loss + ((1 / (i+1)) * (loss.item() - train_loss))   
		#outputs = F.softmax(outputs)
		#accumulate values
		outputs_np = outputs.cpu().data.numpy()
		labels_np = labels.cpu().data.numpy()
		outputs_np = np.argmax(outputs_np, axis=1)

		y_true = np.append(y_true, outputs_np)
		y_pred = np.append(y_pred, labels_np)

		
		if (i%100==0):
			print("["+str(i)+"/"+str(tot_iterations)+"]")
			acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
			kappa = metrics.cohen_kappa_score(y1=y_true,y2=y_pred, weights='quadratic')

			print("accuracy: " + str(acc))
			print("kappa score: " + str(kappa))

		i = i+1
		
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
