from tqdm import tqdm
from CCMR import CCMRModelEngine
from data import RecipeMealDatasetGenerator
from utils import seed_it

CCMR_config = {'model_name': 'CCMR',
               'alias': '{}_dim{}_bz{}_lr{}',
               'data_path': './data/MealRec',
               'save_checkpoint': False,
               'epoch_num': 100,
               'meal_batch_size': 1024,
               'dropout': 0,
               'att_dropout': 0,
               'optimizer': 'adam',
               'adam_lr': 0.003,
               'l2_regularization': 0,
               'meal_max_num': 64,
               'meal_recipe_max_num': 3,
               'embed_shape': 5,
               'use_cuda': True,
               'device_id': 0,
               'model_dir': './checkpoints',
               }

config = CCMR_config

data_generator = RecipeMealDatasetGenerator(config['data_path'])

meal_evaluate_data = data_generator.meal_evaluate_data

config['num_users'], config['num_recipes'], config['num_meals'] = data_generator.UserRecipeMealShape

engine = CCMRModelEngine(config)

for k in config.keys():
    print(k, config.get(k))

for epoch_id in tqdm(range(config['epoch_num'])):
    epoch_id = epoch_id + 1
    train_meal = data_generator.instance_a_train_meal_loader(config['meal_batch_size'])
    engine.train_an_epoch(train_meal, epoch_id)
    if epoch_id % 5 == 0 and epoch_id > 0:
        p_k, MAP = engine.evaluate(meal_evaluate_data, epoch_id=epoch_id, K=5)
exit()
