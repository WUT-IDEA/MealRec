import numpy as np
import torch
import torch.nn as nn
from engine import Engine
from utils import use_cuda, load_obj


class CCMRModel(nn.Module):
    def __init__(self, config):
        super(CCMRModel, self).__init__()
        self.config = config
        self.num_users = config['num_users']
        self.num_recipes = config['num_recipes']
        self.num_meals = config['num_meals']
        self.embed_shape = config['embed_shape']
        self.user_meal = load_obj(config['data_path'] + '/user_meal_matrix')
        self.user_recipe = load_obj(config['data_path'] + '/user_recipe_matrix')
        self.meal_recipe = load_obj(config['data_path'] + '/meal_recipe_matrix')
        self.meal_max_num = config['meal_max_num']
        self.meal_recipe_max_num = config['meal_recipe_max_num']
        self.user2meal = self.getUserMeals(config['data_path'] + '/train_user_meal')
        self.meal2recipe = self.getMealRecipes(config['data_path'] + '/meal_idx_2_recipe_idx')

        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_shape)
        self.embedding_recipe = nn.Embedding(num_embeddings=self.num_recipes + 1, embedding_dim=self.embed_shape,
                                             padding_idx=self.num_recipes)
        self.embedding_meal = nn.Embedding(num_embeddings=self.num_meals + 1, embedding_dim=self.embed_shape,
                                           padding_idx=self.num_meals)

        self.position = nn.Parameter(torch.FloatTensor(self.meal_recipe_max_num, self.embed_shape))

        self.sdense = nn.Linear(self.embed_shape * 3, self.embed_shape * 2)
        self.dense = nn.Linear(self.embed_shape * 2, self.embed_shape * 2)

        self.ipred = nn.Linear(self.embed_shape * 2, 1)
        self.bpred = nn.Linear(self.embed_shape * 2, 1)

        self.dropout = config['dropout']
        self.att_dp = config['att_dropout']

        nn.init.normal_(self.embedding_recipe.weight, mean=0, std=0.001)
        nn.init.normal_(self.embedding_meal.weight, mean=0, std=0.001)
        nn.init.normal_(self.embedding_user.weight, mean=0, std=0.001)
        nn.init.normal_(self.position, mean=0, std=0.001)

        self.func = nn.LeakyReLU()

    def forward(self, user_indices, indices):
        user_emb = self.embedding_user(user_indices)

        meal_recipes_idx = self.meal2recipe[indices]
        # ================== Personalized Recipe Embedding for Meal ====================
        predict_recipe_feature = self.embedding_recipe(meal_recipes_idx)
        predict_recipe_feature += self.position.unsqueeze(0).repeat(predict_recipe_feature.shape[0], 1, 1)

        # ================== Recipe-level Aggregation for Meal ====================
        attention_output = self.ScaledDotProductAttention(
            query=user_emb.unsqueeze(-2),
            key=predict_recipe_feature,
            value=predict_recipe_feature
        )
        # ================== Meal Representation ====================
        y_feature = attention_output.squeeze(-2)

        # ================== Personalized Recipe Embedding for User ====================
        user_meals_idx = self.user2meal[user_indices]

        user_meals_idx.masked_fill_(indices.unsqueeze(-1).repeat(1, self.meal_max_num) == user_meals_idx,
                                    self.num_meals)
        user_meal_recipes_idx = self.meal2recipe[user_meals_idx]

        recipe_feature = self.embedding_recipe(user_meal_recipes_idx)
        recipe_feature += self.position.unsqueeze(0).repeat(recipe_feature.shape[0] * recipe_feature.shape[1], 1,
                                                            1).view(-1,
                                                                    self.meal_max_num,
                                                                    self.meal_recipe_max_num,
                                                                    self.embed_shape)

        mask = user_meals_idx == self.num_meals

        user_category_feature = []
        for i in range(recipe_feature.shape[-2]):
            category_recipes = recipe_feature[:, :, i, :]
            # ================== Recipe-level Aggregation for User ====================
            user_category = self.ScaledDotProductAttention(
                query=predict_recipe_feature[:, i, :].unsqueeze(-2),
                key=category_recipes,
                value=category_recipes,
                key_padding_mask=mask
            )
            user_category_feature.append(user_category)
        user_category_feature = torch.cat(user_category_feature, dim=-2)
        # ================== Category-level Aggregation ====================
        user_feature = self.ScaledDotProductAttention(
            query=user_emb.unsqueeze(-2),
            key=user_category_feature,
            value=user_category_feature
        )
        user_feature = user_feature.squeeze(-2)

        x = torch.cat([user_feature, y_feature, user_feature * y_feature], dim=-1)
        x = nn.Dropout(p=self.dropout)(self.func(self.sdense(x)))
        x = nn.Dropout(p=self.dropout)(self.func(self.dense(x)))

        x = self.func(self.bpred(x))
        return x

    def getUserMeals(self, path):
        user2meal = []
        train_meal = load_obj(path)
        for i in range(train_meal.shape[0]):
            meals_list = train_meal[i].nonzero()[1]
            if len(meals_list) > self.meal_max_num:
                sample = np.random.choice(range(len(meals_list)), self.meal_max_num, replace=False)
                meals_list = meals_list[sample]
            else:
                padding = [self.num_meals] * (self.meal_max_num - len(meals_list))
                meals_list = np.append(meals_list, padding)
            user2meal.append(meals_list)
        user2meal = torch.LongTensor(user2meal)
        return user2meal

    def getMealRecipes(self, np_path):
        meal2recipe = load_obj(np_path)
        meal2recipe = np.row_stack((meal2recipe, np.array([self.num_recipes] * 3)))
        meal2recipe = torch.LongTensor(meal2recipe)
        return meal2recipe

    def ScaledDotProductAttention(self, query, key, value, key_padding_mask=None):
        scores = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.embed_shape)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-2).expand_as(scores)
            scores.masked_fill_(key_padding_mask, -1e9)
        attn = nn.Dropout(p=self.att_dp)(nn.Softmax(dim=-1)(scores))
        context = torch.matmul(attn, value)
        return context

    def load2GPU(self):
        self.meal2recipe = self.meal2recipe.cuda()
        self.user2meal = self.user2meal.cuda()


class CCMRModelEngine(Engine):

    def __init__(self, config):
        self.model = CCMRModel(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
            self.model.load2GPU()
        super(CCMRModelEngine, self).__init__(config)
        print(self.model)
