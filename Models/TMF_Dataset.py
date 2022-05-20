import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TMF_Dataset(Dataset):
    def __init__(self, interact_data, recepy_data, user_id = 'user_id',
     recipe_id = 'recipe_id', 
     user_taggs_int = 'user_taggs_int', recipe_taggs_int ='recipe_taggs_int',
      rating='rating'):

        self.interact_data = interact_data
        self.recepy_data = recepy_data
        self.user_id = user_id
        self.recipe_id = recipe_id
        self.user_taggs_int = user_taggs_int
        self.recipe_taggs_int = recipe_taggs_int
        self.rating = rating
        self.tags = 'tags'

        self.interact_df , self.recepy_df, self.unique_tags_to_int, self.unique_tags_int_to_tags = self.preprocessing(interact_data, recepy_data)


    def tags_to_int_list(self, taggs_string,  unique_tags_to_int):
        tags = taggs_string.replace('[','').replace(']','').replace("'",'').split(',')
        tags = [item.strip() for item in tags if len(item)>0]
        tags = [ unique_tags_to_int[tag] for tag in tags]
        return tags

    def preprocessing(self, interact_data, recepy_data):

        unique_tags = recepy_data[self.tags].values
        unique_tags = [st.replace('[','').replace(']','').replace("'",'').split(',') for st in unique_tags]
        unique_tags = [item.strip() for sublist in unique_tags for item in sublist if len(item)>0]
        unique_tags = list(set(unique_tags))

        unique_tags_to_int = {'pad':0}
        unique_tags_int_to_tags = {0:'pad'}
        for i,tag in enumerate(unique_tags):
            unique_tags_to_int[tag] = i+1
            unique_tags_int_to_tags[i+1] = tag

        recepy_data['tags_int'] = recepy_data[self.tags].apply(lambda x: self.tags_to_int_list(x, unique_tags_to_int))

        nunique_users = interact_data[self.user_id].nunique()
        
        users_taggs_dic = {}
        for i in range(nunique_users):
            user_recepies = interact_data[interact_data[self.user_id]==i][self.recipe_id].values
            user_taggs = recepy_data[recepy_data[self.recipe_id].isin(user_recepies)]['tags_int'].values
            user_taggs = [item for sublist in user_taggs for item in sublist]
            users_taggs = list(set(user_taggs))
            users_taggs_dic[i] = users_taggs
        
        interact_data['user_taggs_int'] = interact_data[self.user_id].apply(lambda x: users_taggs_dic[x])
        interact_data['recipe_taggs_int'] = interact_data[self.recipe_id].apply(lambda x: recepy_data[recepy_data[self.recipe_id]==x]['tags_int'].values[0])
        return interact_data, recepy_data, unique_tags_to_int, unique_tags_int_to_tags

    def __len__(self):
        return len(self.interact_df)

    def __getitem__(self, idx):
        user_id = np.array(self.interact_df[self.user_id].values[idx])
        recipe_id = np.array(self.interact_df[self.recipe_id].values[idx])
        user_taggs = np.array(self.interact_df[self.user_taggs_int].values[idx])
        recipe_taggs = np.array(self.interact_df[self.recipe_taggs_int].values[idx])
        rating = np.array(self.interact_df[self.rating].values[idx])

        return torch.tensor(user_id), torch.tensor(recipe_id), torch.tensor(user_taggs), torch.tensor(recipe_taggs), torch.tensor(rating)

class TMFCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        
        user_id, recipe_id, user_taggs, recipe_taggs, rating = zip(*batch)
        
        user_taggs = pad_sequence(user_taggs, batch_first=True, padding_value=self.pad_idx)
        recipe_taggs = pad_sequence(recipe_taggs, batch_first=True, padding_value=self.pad_idx)
        
        user_id = torch.stack(user_id)
        recipe_id = torch.stack(recipe_id)
        rating = torch.stack(rating)
        
        return user_id, recipe_id, user_taggs, recipe_taggs, rating
