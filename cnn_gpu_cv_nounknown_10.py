import datetime
import argparse,os
import numpy as np
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import chainer
import cupy as cp
from chainer import Chain, optimizers, training, cuda
from chainer.training import extensions,triggers
import chainer.functions as F
import chainer.links as L

parser = argparse.ArgumentParser(description='Predcition of multiple cancer with gene information')
parser.add_argument('--batchsize', '-mb', type=int, default=32, help='Number of bins in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=120, help='Number of iters over the dataset for train')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')#結果存放資料夾
parser.add_argument('--modelPath', '-mp', default='model', help='model output path')#trainingmodel圖片存放
parser.add_argument('--frequency', '-f', type=int, default=-1, help='Frequency of taking a snapshot')
parser.add_argument('--resume', '-r', default='', help='Resume the training from snapshot')
parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (if none -1)')
parser.add_argument('--unit', '-u', type=int, default=100, help='Number of units')
parser.add_argument('--data', '-in', default="../data/TCGA_BRCA/", help='Data path')
parser.add_argument('--type', '-t', type=int, default=0, required=True, help='Type of the feature')
parser.add_argument('--ipunit', '-iu', type=int, default=3264, help='The number of input unit')
parser.add_argument('--monitor', '-mnt', type=int, default=0, help='Early stop monitor')

args = parser.parse_args()


gpu_flag = args.gpu

catalog_set = ['Alphabet','EZID_KOorthologs','EZID_KOparalogs', 'Shuffle', 'EZID_ProSim', 'EZID_IES', 'EZID_IES7up', 'EZID_Shuffle16321', 'EZID_Shuffle', 'EZID_Shuffle6045_1',
'EZID_Shuffle6045_2', 'EZID_Shuffle6045_3', 'EZID_Shuffle6045_4', 'EZID_Shuffle6045_5', 'EZID_Shuffle6045_6', 'EZID_Shuffle6045_7', 'EZID_Shuffle6045_8', 'EZID_Shuffle6045_9', 'EZID_Shuffle6045_10', 'EZID_FC2P0.05',
'EZID_IES7up_DelPAM50', 'EZID_IES7up_PlusPAM50', 'EZID_IES9up', 'EZID_IES11up', 'EZID_PAM50', 'EZID_IES3down', 'EZID_IES5down','EZID_IES7down','EZID_IES11up3down','EZID_IES9up3down',
'WholeGenome_Alphabet','EZID_Shuffle6045']#31種
monitor_set = ['validation/main/loss', 'main/loss', 'validation/main/accuracy','main/accuracy']
catalog = catalog_set[args.type]
mnt = monitor_set[args.monitor]
log = 'log2'
dir_ = './data/TCGA_BRCA/'
FC=''
toy = ''
unknown = '_CliSubType_NoUn' #_CliSubType_NoUn or _NoUn
multicancer = ''

if gpu_flag>=0:
        chainer.cuda.get_device_from_id(gpu_flag).use()
xp = cuda.cupy if gpu_flag >= 0 else np

#CNN model
class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__(
            conv1 = L.Convolution2D(1, 16, (2,10), nobias = False),
            conv2 = L.Convolution2D(16, 32, (1,5), nobias = False, stride = 2),
            conv3 = L.Convolution2D(32, 64, (1,3), nobias = False, stride = 2),
           # bn1 = L.BatchNormalization(32),
           # bn2 = L.BatchNormalization(64),
            l1 = L.Linear(args.ipunit, args.unit),
            l2 = L.Linear(args.unit, 50),
            l3 = L.Linear(50, 5, initialW=np.zeros((5, 50), dtype=np.float32))
        )

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), (1,4))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), (1,4))
        h = F.max_pooling_2d(F.relu(self.conv3(h)), (1,4))
        h = F.dropout(F.relu(self.l1(h)), 0.5)
        h = F.dropout(F.relu(self.l2(h)), 0.5)
        y = self.l3(h)
        return y


EPOCH_NUM = args.epoch
BATCH_SIZE = args.batchsize

#load data
x_file = dir_+'feature_BRCA_'+log+'RNAseq'+FC+'_CNV_Profile_P_X_'+catalog+'_LinkFormat_Order'+unknown+'.txt'
y_file = 'feature_BRCA_'+log+'RNAseq'+FC+'_CNV_Profile_P_X_'+catalog+'_LinkFormat_Order'+unknown
#創建result存放資料夾
if not os.path.exists("./" + y_file):
    os.makedirs("./" + y_file)
    os.chmod("./" + y_file,0o777)
data = np.loadtxt(x_file)

t_file = './data/TCGA_BRCA/label_TCGA_BRCA_Subtypes'+unknown+'.txt'
target = np.loadtxt(t_file)

N = data.shape[0]
span= int(data.shape[1]/2)
print(data.shape[0], data.shape[1])


#change the data format to be able to read by chainer.datasets
dataset = []
for x, t in zip(data.astype(xp.float32), target.astype(xp.int32)):
    dataset.append((x.reshape(1, 2, span), t))
N = len(dataset)

print(N, len(dataset[0]), dataset[0][0])

    #apply k-fold cv on trai_val set, in this case: 5
cv_set = chainer.datasets.get_cross_validation_datasets_random(dataset, 10, seed = 0)
total_acc = 0
for i in range(len(cv_set)):
    print("Train:",i+1)
    train, val = cv_set[i]
    c_train = [0,0,0,0,0]
    for j in range(len(train)):
        c_train[train[j][1]]+=1
    c_val = [0,0,0,0,0]
    for j in range(len(val)):
        c_val[val[j][1]]+=1
    print(c_train,c_val)

    #model = L.Classifier(CNN(), lossfun=F.softmax_cross_entropy(train[i][0].astype(xp.ndarray),train[i][1].astype(xp.ndarray),class_weight=None))
    model = L.Classifier(CNN())
    if gpu_flag >= 0:
        model.to_gpu(gpu_flag)

    optimizer = optimizers.Adam(alpha=0.001, beta1=0.8, beta2 = 0.98)
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
    val_iter = chainer.iterators.SerialIterator(val, BATCH_SIZE, repeat=False, shuffle=False)

    stop_trigger = triggers.EarlyStoppingTrigger(
        check_trigger = (120, 'epoch'),
        #monitor = mnt,
        patients = 3,
        verbose = True,
        max_trigger = (120, 'epoch'))

    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_flag)
    #trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
    trainer = training.Trainer(updater, stop_trigger, out=y_file)
    log_interval = 120, 'epoch'
    trainer.extend(extensions.Evaluator(val_iter, model, device=gpu_flag))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=log_interval,log_name="{}".format(i+1)+"_log.csv"))
    trainer.extend(extensions.snapshot(), trigger=(EPOCH_NUM, 'epoch'))
    trainer.extend(extensions.PlotReport( ["main/loss", "validation/main/loss"], file_name=toy+'loss_'+catalog+log+FC+'_'+str(i)+unknown+'.png'))
    trainer.extend(extensions.PlotReport( ["main/accuracy", "validation/main/accuracy"], file_name=toy+'accuracy_'+catalog+log+FC+'_'+str(i)+unknown+'.png'))
    trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"]))
    trainer.run()

        #test the model
 #   print('test')
 #   test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)
 #   evaluator = extensions.Evaluator(test_iter, model, device = gpu_flag)
 #   result = evaluator()
 #   print('test accuracy:', float(result['main/accuracy']))
 #   total_acc += float(result['main/accuracy'])

#ave_acc = float(total_acc/len(cv_set))
#print('average accuracy:', ave_acc)
