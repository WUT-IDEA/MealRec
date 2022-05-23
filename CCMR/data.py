import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class UserRecipeMealDataset(Dataset):

    def __init__(self, user_tensor, pos_tensor, neg_tensor):
        self.user_tensor = user_tensor
        self.pos_tensor = pos_tensor
        self.neg_tensor = neg_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.pos_tensor[index], self.neg_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class RecipeMealDatasetGenerator(object):
    """Construct dataset"""

    def __init__(self, data_path):
        self.data_path = data_path
        self.user_recipe = load_obj(data_path + '/user_recipe_matrix')
        self.user_meal = load_obj(data_path + '/user_meal_matrix')
        self.meal_recipe = load_obj(data_path + '/meal_recipe_matrix')
        self.test_recipe, self.train_recipe = load_obj(data_path + '/test_user_recipe'), load_obj(
            data_path + '/train_user_recipe')
        self.test_meal, self.train_meal = load_obj(data_path + '/test_user_meal'), load_obj(
            data_path + '/train_user_meal')
        self.negative = load_obj(data_path + '/negative')

        self.user_shape, self.recipe_shape = self.user_recipe.shape
        self.meal_shape, self.recipe_shape = self.meal_recipe.shape

    def instance_a_train_meal_loader(self, batch_size):
        # print("instance_meal_dataloader...")
        train_meal_u, train_meal_i = self.train_meal.nonzero()
        neg = np.random.choice(range(self.meal_shape), len(train_meal_u), replace=True)
        for i in range(len(neg)):
            meal = self.user_meal[train_meal_u[i]].nonzero()[1]
            while neg[i] in meal:
                neg[i] = np.random.choice(range(self.meal_shape))
        dataset = UserRecipeMealDataset(user_tensor=torch.LongTensor(train_meal_u),
                                        pos_tensor=torch.LongTensor(train_meal_i),
                                        neg_tensor=torch.LongTensor(neg))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    @property
    def UserRecipeMealShape(self):
        return self.user_shape, self.recipe_shape, self.meal_shape

    @property
    def meal_evaluate_data(self):
        """create meal evaluate data"""
        test_users, test_pos_meals = self.test_meal.nonzero()

        test_neg_meals = []
        for i in range(len(test_users)):
            neg = self.negative[test_users[i]]
            test_neg_meals.append(neg)
        return [torch.LongTensor(test_users).unsqueeze(1), torch.LongTensor(test_pos_meals).unsqueeze(1),
                torch.LongTensor(test_neg_meals).unsqueeze(2)]
