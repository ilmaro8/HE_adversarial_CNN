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
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='domain_center')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=True)
parser.add_argument('-d', '--dataset', help='dataset where test data',type=str, default='TMAZ')
parser.add_argument('-m', '--model', help='path of the model to load',type=str, default='./model/')
parser.add_argument('-i', '--input', help='path of input csv',type=str, default='./model/')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
TASK = args.TASK
EMBEDDING_bool = args.features
DATASET = args.dataset
INPUT_DATA = args.input
MODEL_PATH = args.model

def create_dir(directory):
	if not os.path.isdir(directory):
		try:
			os.mkdir(directory)
		except OSError:
			print ("Creation of the directory %s failed" % directory)
		else:
			print ("Successfully created the directory %s " % directory) 

#DIRECTORIES CREATION
print("CREATING/CHECKING DIRECTORIES")
checkpoint_path = MODEL_PATH+'checkpoints_MIL/'
create_dir(checkpoint_path)

#path model file
model_weights_filename = MODEL_PATH

#CSV LOADING

print("CSV LOADING ")
csv_filename_testing = INPUT_DATA
#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

imageNet_weights = True

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
		m_binary = torch.nn.Sigmoid()

		domain_emb = self.domain_embedding(x)
		domain_emb = dropout(domain_emb)
		domain_prob = self.domain_classifier(domain_emb)

		domain_prob = m_binary(domain_prob)

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


#DATA AUGMENTATION
from torchvision import transforms
prob = 0.5

if (TASK=='domain_center'):

	pipeline_transform = A.Compose([
		A.VerticalFlip(p=prob),
		A.HorizontalFlip(p=prob),
		A.RandomRotate90(p=prob),
		#A.ElasticTransform(alpha=0.1,p=prob),
		#A.HueSaturationValue(hue_shift_limit=(-9),sat_shift_limit=25,val_shift_limit=10,p=prob),
		])

elif (TASK=='domain_center_aug'):

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

		if (DATASET=='diagset' or DATASET=='PANDA'):
			X = X.convert("RGB")
		
		X = np.asarray(X)
		y = self.labels[index]
		#data augmentation

		if (self.mode == 'train'):
			X = pipeline_transform(image=X)['image']

		new_image = np.asarray(X)
		#data transformation
		input_tensor = preprocess(new_image)
				
		return input_tensor, np.asarray(y), ID

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Parameters

num_workers = 2

params_test = {'batch_size': BATCH_SIZE,
		  #'shuffle': True,
		  #'sampler': ImbalancedDatasetSampler(test_dataset),
		  'num_workers': num_workers}

testing_set = Dataset_patches(test_dataset[:,0], test_dataset[:,1],'test')
testing_generator = data.DataLoader(testing_set, **params_test)


model = torch.load(model_path)
model.to(device)
model.eval()

filenames_wsis = []
pred_b = []
pred_gp3 = []
pred_gp4 = []
pred_gp5 = []

y_true = []
y_pred = []

with torch.no_grad():	
	for inputs,labels, filename in testing_generator:
		inputs, labels = inputs.to(device), labels.to(device)
		
		# zero the parameter gradients
		
		# forward + backward + optimize
		_, outputs = model(inputs, None)

		outputs_np = outputs.cpu().data.numpy()
		labels_np = labels.cpu().data.numpy()

		filenames_wsis = np.append(filenames_wsis,filename)
		pred_b = np.append(pred_b,outputs_np[:,0])
		pred_gp3 = np.append(pred_gp3,outputs_np[:,1])
		pred_gp4 = np.append(pred_gp4,outputs_np[:,2])
		pred_gp5 = np.append(pred_gp5,outputs_np[:,3])

		outputs_np = np.argmax(outputs_np, axis=1)

		y_true = np.append(y_true, outputs_np)
		y_pred = np.append(y_pred, labels_np)

filename_training_predictions = checkpoint_path+'patches_predictions_'+DATASET+'.csv'

File = {'filenames':filenames_wsis, 'pred_gpb':pred_b, 'pred_gp3':pred_gp3,'pred_gp4':pred_gp4,'pred_gp5':pred_gp5}

df = pd.DataFrame(File,columns=['filenames','pred_gpb', 'pred_gp3','pred_gp4','pred_gp5'])
np.savetxt(filename_training_predictions, df.values, fmt='%s',delimiter=',')	

kappa_score_TMA_general_filename = checkpoint_path+'kappa_score_'+DATASET+'.csv'
confusion_matrix_TMA_general_filename = checkpoint_path+'conf_matrix_'+DATASET+'.csv'
acc_TMA_general_filename = checkpoint_path+'acc_'+DATASET+'.csv'
acc_balanced_TMA_general_filename = checkpoint_path+'acc_balanced_'+DATASET+'.csv'
f1_score_TMA_general_filename = checkpoint_path+'f1_score_'+DATASET+'.csv'
roc_auc_TMA_general_filename = checkpoint_path+'roc_auc_'+DATASET+'.csv'

#k-score
k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
print("k_score " + str(k_score))

#confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
print("confusion_matrix ")
print(str(confusion_matrix))

#confusion matrix normalized
np.set_printoptions(precision=2)
cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(str(cm_normalized))

f1_score = metrics.f1_score(y_true, y_pred, average='macro')
print("f1_score " + str(f1_score))

acc_balanced = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
print("acc_balanced " + str(acc_balanced))

acc = metrics.accuracy_score(y_true, y_pred)
print("acc " + str(acc))

try:
	roc_auc_score = metrics.roc_auc_score(y_true, outs, average='weighted')
	print("roc_auc " + str(roc_auc_score))
except:
	pass

kappas = [k_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_TMA_general_filename, df.values, fmt='%s',delimiter=',')

kappas = [confusion_matrix]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(confusion_matrix_TMA_general_filename, df.values, fmt='%s',delimiter=',')

kappas = [f1_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(f1_score_TMA_general_filename, df.values, fmt='%s',delimiter=',')

kappas = [acc_balanced]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(acc_balanced_TMA_general_filename, df.values, fmt='%s',delimiter=',')

kappas = [acc]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(acc_TMA_general_filename, df.values, fmt='%s',delimiter=',')

try:
	kappas = [roc_auc_score]

	File = {'val':kappas}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(roc_auc_TMA_general_filename, df.values, fmt='%s',delimiter=',')
except:
	pass