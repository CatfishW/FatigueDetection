from dataset import FatigueRegressionDataset,FatigueClassificationDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from model import FatigueRegressionModel,FatigueClassificationModel
import torch.nn as nn
import pandas as pd
import torch
import tqdm
from sklearn.metrics import r2_score
#BERT
from transformers import BertTokenizer, BertModel
# def standardize(data):
#     return (data - data.mean())/data.std()
# def unstandardize(data,mean,std):
#     return data*std + mean
exp_name = 'exp_gru'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
type = 'classification'
model = FatigueRegressionModel(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    num_layers=2,
    target_length=5,
    dropout=0.1
    ).to(device) if type == 'regression' else FatigueClassificationModel(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    num_layers=2,
    feature_length=12,
    num_cls=2,
    dropout=0.001,
    trick=False
    ).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
#train_dataset =FatigueRegressionDataset('./data/merged_data.csv')
#train_dataset = FatigueClassificationDataset('./data/MEFAR_MID.csv')
train_dataset = FatigueClassificationDataset('./data/merged_data.csv',train=True)
train_dataloader = tqdm.tqdm(DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True))
#test_dataset = FatigueRegressionDataset('./data/merged_data.csv',train=False)
#test_dataset = FatigueClassificationDataset('./data/MEFAR_MID.csv',train=False)
test_dataset = FatigueClassificationDataset('./data/merged_data.csv',train=False)
test_dataloader = tqdm.tqdm(DataLoader(test_dataset, batch_size=64, shuffle=False,drop_last=True))
criterion_regression = nn.MSELoss()
criterion_classification = nn.CrossEntropyLoss()
#train
def train_regression():
    for epoch in range(50):
        for i, data in enumerate(train_dataloader):
            inputs,targets = data
            #inputs,targets= standardize(inputs),standardize(targets)
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_regression(outputs,targets)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch}, MSELoss {loss.item()}')
                #r2 = r2_score(targets.squeeze(0).detach().cpu().numpy(),outputs.squeeze(0).detach().cpu().numpy())
                #print(f'Epoch {epoch}, R2 Score: {r2}')
        scheduler.step()
        #save model
        torch.save(model.state_dict(), f'{exp_name}.pth')
def train_classification(pretrained=None,trick=False,val=False):
    if trick:
        #BERT
        with torch.no_grad():
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased').eval()
            text_data = ['acc_x', 'acc_y', 'acc_z', 'GSR', 'HR', 'SKT', 'PPG','space_distance', 'hdistance_to_eye_center', 'distance_to_eye_center', 'intersection_pca', 'Pose_pca']
            text_inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
            outputs = bert_model(**text_inputs)
            text_inputs = outputs.pooler_output
            text_inputs = text_inputs.to(device)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    for epoch in range(50):
        acc_ = []
        for i, data in enumerate(train_dataloader):
            inputs,targets = data
            #inputs,targets= standardize(inputs),standardize(targets)
            inputs,targets = inputs.to(device),targets.to(device)
            optimizer.zero_grad()
            if len(inputs.shape) == 2:
                inputs = inputs.unnsqueeze(1)
            outputs = model(inputs) if not trick else model(inputs,text_inputs)
            loss = criterion_classification(outputs.flatten(0,1),targets.flatten())
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Epoch {epoch}, CrossEntropyLoss {loss.item()}')
                #acc
                outputs = torch.softmax(outputs,dim=-1)
                acc = (outputs.argmax(-1)[:,0] == targets[:,0]).sum().item()/len(targets)
                acc_.append(acc)
                print(f'Epoch {epoch}, Accuracy {acc}')
        #mean acc
        print(f'Train Mean Accuracy: {sum(acc_)/len(acc_)}')
        
        scheduler.step()
        #save model
        torch.save(model.state_dict(), f'{exp_name}.pth')
        if val:
            model.eval()
            with torch.no_grad():
                acc_ = []
                for i, data in enumerate(test_dataloader):
                    inputs,targets = data
                    #inputs,targets= standardize(inputs),standardize(targets)
                    inputs,targets = inputs.to(device),targets.to(device)
                    if len(inputs.shape) == 2:
                        inputs = inputs.unnsqueeze(1)
                    outputs = model(inputs) if not trick else model(inputs,text_inputs)
                    #to probability
                    outputs = torch.softmax(outputs,dim=-1)
                    acc = (outputs.argmax(-1)[:,0] == targets[:,0]).sum().item()/len(targets)
                    acc_.append(acc)
                    print(f'Accuracy: {acc}')
                val_mean_acc = sum(acc_)/len(acc_)
                print(f'Val Mean Accuracy: {val_mean_acc}')
                torch.save(model.state_dict(), f'{exp_name}_val_{val_mean_acc}.pth')
def test_CLS():
    model.load_state_dict(torch.load(f'{exp_name}.pth'))
    model.eval()
    with torch.no_grad():
        acc_ = []
        for i, data in enumerate(test_dataloader):
            inputs,targets = data
            #inputs = standardize(inputs)
            #targets = standardize(targets)
            inputs = inputs.to(device)
            targets = targets.to(device)
            if len(inputs.shape) == 2:
                inputs = inputs.unnsqueeze(1)
            outputs = model(inputs)
            #to probability
            outputs = torch.softmax(outputs,dim=-1)
            acc = (outputs.argmax(-1)[:,0] == targets[:,0]).sum().item()/len(targets)
            acc_.append(acc)
            acc_.append(acc)
            print(f'Accuracy: {acc}')
        print(f'Mean Accuracy: {sum(acc_)/len(acc_)}')
        #save txt
        with open(f'{exp_name}.txt','w') as f:
            f.write(f'Mean Accuracy: {sum(acc_)/len(acc_)}')
#test
def test_R2():
    model.load_state_dict(torch.load(f'{exp_name}.pth'))
    model.eval()
    with torch.no_grad():
        r2_= []
        for i, data in enumerate(test_dataloader):
            inputs,targets = data
            #inputs = standardize(inputs)
            #targets = standardize(targets)
            inputs = inputs.to(device)
            outputs = model(inputs)
            #outputs = unstandardize(outputs,inputs_mean,inputs_std)
            targets = targets.squeeze().cpu().numpy()
            outputs = outputs.squeeze().cpu().numpy()
            r2 = r2_score(targets,outputs)
            r2_.append(r2)
            print(f'R2 Score: {r2}')
        print(f'Mean R2 Score: {sum(r2_)/len(r2_)}')
        #save txt
        with open(f'{exp_name}.txt','w') as f:
            f.write(f'Mean R2 Score: {sum(r2_)/len(r2_)}')
if __name__ == '__main__':
    #train_regression()
    #train_classification(pretrained='exp1.pth')
    train_classification(pretrained=None,trick=False,val=True)
    #test_R2()
    #test_CLS()
