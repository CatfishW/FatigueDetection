import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FatigueRegressionDataset(Dataset):
    def __init__(self,dataset_path,train=True):
        super().__init__()
        self.data_csv = pd.read_csv(dataset_path)
        self.sequence_length = 30
        self.train = train
        #dataset_split
        if train:
            self.data_csv = self.data_csv.iloc[:int(0.8*len(self.data_csv))]
        else:
            self.data_csv = self.data_csv.iloc[int(0.8*len(self.data_csv)):]
        
    def __len__(self):
        if self.train:
            return len(self.data_csv)//self.sequence_length
        else:
            return 1
    def __getitem__(self, index):
        if self.train:
            start_index = index*self.sequence_length
            end_index = (index+1)*self.sequence_length
            data = self.data_csv.iloc[start_index:end_index]
            features = data[['acc_x', 'acc_y', 'acc_z', 'GSR', 'HR', 'SKT', 'PPG']]
            targets = data[['space_distance', 'hdistance_to_eye_center', 'distance_to_eye_center', 'intersection_pca', 'Pose_pca']]
            return torch.tensor(features.values).float(), torch.tensor(targets.values).float()
        else:
            features = self.data_csv[['acc_x', 'acc_y', 'acc_z', 'GSR', 'HR', 'SKT', 'PPG']]
            targets = self.data_csv[['space_distance', 'hdistance_to_eye_center', 'distance_to_eye_center', 'intersection_pca', 'Pose_pca']]
            return torch.tensor(features.values).float(), torch.tensor(targets.values).float()
class FatigueClassificationDataset(Dataset):
    def __init__(self,dataset_path,train=True):
        super().__init__()
        self.scaler = StandardScaler()
        self.data_csv = pd.read_csv(dataset_path)
        self.train = train
        self.sequence_length = 10
        #dataset_split
        if train:
            self.data_csv = self.data_csv.iloc[:int(0.8*len(self.data_csv))]
            #random shuffle
            self.data_csv = self.data_csv.sample(frac=1).reset_index(drop=True)
        else:
            self.data_csv = self.data_csv.iloc[int(0.8*len(self.data_csv)):]
        
    def __len__(self):
        return len(self.data_csv)-10#//self.sequence_length
    def __getitem__(self, index):
        # start_index = index*self.sequence_length
        # end_index = (index+1)*self.sequence_length
        start_index = index
        end_index = index+self.sequence_length
        data = self.data_csv.iloc[start_index:end_index]
        #features = data[['BVP','EDA', 'TEMP', 'AccX', 'AccY', 'AccZ', 'HR', ' Delta', ' Theta', ' Alpha1', ' Alpha2', ' Beta1', ' Beta2', ' Gamma1', ' Gamma2', ' Attention', ' Meditation']]
        features = data[['acc_x', 'acc_y', 'acc_z', 'GSR', 'HR', 'SKT', 'PPG','space_distance', 'hdistance_to_eye_center', 'distance_to_eye_center', 'intersection_pca', 'Pose_pca']]
        
        features = self.scaler.fit_transform(features)
        #to a list
        try:
            features = features.tolist()
        except:
            features = features.values
        #targets = data['class']
        targets = data['Bfatigue']
        return torch.tensor(features).float(), torch.tensor(targets.values).long()
