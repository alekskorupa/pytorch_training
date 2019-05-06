
# coding: utf-8

# In[20]:


#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# In[26]:


file_id = '15xyAVSjvmJE8FJrAA2WVNp9d3tqPXUqh'
dest_path = '/home/projects/flickr90k/dataset.tgz'
destination = dest_path
download_file_from_google_drive(file_id, destination)


# ## Prepare data

# In[31]:


import json
import pandas as pd
    
def info_to_df(paths: list, x_col='filename', y_col='class') -> pd.DataFrame:
    info_ds = pd.DataFrame(paths, columns=[x_col])
    info_ds[y_col] = info_ds[x_col].apply(lambda x: x.split('/')[0])

    return info_ds.sample(frac=1)


# In[43]:


dataset_path = '/home/projects/flickr90k/'

with open(os.path.join(dataset_path,'train_test_split.json'), 'r') as f:
    split_info = json.load(f)

train_df = info_to_df(split_info['train'])
test_df = info_to_df(split_info['test'])

print("Training data size: {} \nTest data size: {}".format(len(train_df),
                                                           len(test_df)))


# In[44]:


train_df.head()


# In[47]:


from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train_df['class'].values),
                                                  train_df['class'].values)
class_weights = dict(enumerate(class_weights))
class_weights


# In[56]:


get_ipython().system(' source activate py36')


# In[51]:


import torch
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader


# In[49]:


class Flickr90kDataset(Dataset):
    """Recrutment Task dataset."""

    def __init__(self, dataframe, image_dir, transform=None):
        """
        Args:
            dataframe (): ...
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir,
                                self.dataframe.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.dataframe.iloc[idx, 1]
        sample = {'image': image, 'class': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# In[ ]:


train_dataset = RecrutmentTaskDataset(train_df,
                                      image_dir,
                                      transforms.Compose([
                                               to_pil_image(),
                                               torchvision.transforms.Resize(224),
                                               transforms.ToTensor()))
test_dataset = RecrutmentTaskDataset(test_df,
                                     image_dir,
                                     transforms.Compose([
                                         to_pil_image(),
                                         resize(256),
                                         to_tensor()]))

