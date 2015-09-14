
# coding: utf-8

# In[27]:

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pulp
from matplotlib.backends.backend_pdf import PdfPages
sb.set()

pd.set_option('precision', 2)

macro_nutrients = [
              'kcal per 100 g ready to eat',
              'Protein_(g)',
              'Lipid_Tot_(g)',
              'FA_Sat_(g)',
              'Carbohydrt_(g)']

micro_nutrients = [
              'Fiber_TD_(g)',
              'Vit_A_RAE',
              'Vit_B6_(mg)',
              'Vit_B12_(mcg)',
              'Vit_C_(mg)',
              'Vit_E_(mg)',
              'Folate_DFE_(mcg)',
              'Vit_K_(mcg)',
              'Riboflavin_(mg)',
              'Calcium_(mg)',
              'Niacin_(mg)',
              'Cholestrl_(mg)',
              'FA_Mono_(g)',
              'FA_Poly_(g)',
              'Sugar_Tot_(g)',
              'soluble fiber',
              'total Flavonoid',
              'Iron_(mg)',
              'Alpha_Carot_(mcg)',
              'Beta_Carot_(mcg)',
              'Lycopene_(mcg)',
              'Lut+Zea_(mcg)',
              'Beta_Crypt_(mcg)',
              'Sodium_(mg)',
              'Selenium_(mcg)',
              'Potassium_(mg)',
              'Phytosterols(mg)Per 100 g']

nutrient_list = macro_nutrients + micro_nutrients

food_abbrev = ['almond',
                'apples',
                'apricot',
                'asparagus',
                'avocado',
                'banana',
                'barley',
                'kidney bean',
                'snap bean',
                'blueberry',
                'broccoli',
                'cabbage',
                'canola oil',
                'cantaloupe',
                'carrots',
                'cauliflower',
                'celery',
                'cherry',
                'collard',
                'corn flr',
                'corn starch',
                'corn grits',
                'swt corn',
                'cucumber',
                'garlic',
                'grapefruit',
                'grapes',
                'hazelnuts',
                'honeydew',
                'kiwi',
                'lemon',
                'lettuce',
                'macadamia',       
                'oats',            
                'olive oil',       
                'onions',          
                'orange',          
                'peach',           
                'peanut',         
                'pear',            
                'peas',            
                'grn pepper',     
                'pineapple',       
                'pistachio',       
                'potato',          
                'pumpkin',         
                'raspberry',     
                'rice',            
                'soy oil',         
                'spinach',         
                'squash',          
                'strawberry',    
                'sweet potato',    
                'tomato',        
                'walnut',         
                'watermelon',      
                'wheat',           
                'chickpea',       
                'lentil',         
                'soybean',        
                'tofu',            
                'buckwheat',   
                'sorghum',
                'rye',             
                'spelt',           
                'saflwer oil',
                'syrup cane',
                'hfcs'
                ]

ANIMALS =      ['beef',
                'dairy',
                'egg',
                'chicken',
                'pork',
                'salmon',
                'tuna']

food_abbrev += ANIMALS

SUBSETS_TO_REPLACE = map(lambda x: [x], ANIMALS) + [ANIMALS]
#SUBSETS_TO_REPLACE = [['egg']]

UDSA_COLNAME = 'USDA groups'
MAD_COLNAME = 'MAD (kcal/cap/d)'
ENERGY_COLNAME = 'Energ_Kcal'
data_frame = pd.DataFrame.from_csv('plVals.csv')
index2abbrev = dict(zip(data_frame.index, food_abbrev))
data_frame = data_frame.rename(index=index2abbrev)

#%%
# find the corresponding index of the columns in the dataframe to the nutrient list.
# it is enough if the prefix of the column name matches exactly the nutrien name
col_names = list(data_frame.columns)
col_ind_order = np.zeros((1, len(nutrient_list)))
nutrient_list_updated = [''] * len(nutrient_list)
for i, nutr in enumerate(nutrient_list):
    for j, col in enumerate(col_names):
        l_min = min(len(nutr), len(col))
        if nutr[:l_min] == col[:l_min]:
            col_ind_order[0, i] = j
            nutrient_list_updated[i] = col
            continue
nutrient_list = nutrient_list_updated
data = data_frame[nutrient_list].copy()
usda_group = data_frame[UDSA_COLNAME]

# correct to eaten mass basis, so that D entries will be in "per g ready2eat"
data.loc[:, nutrient_list[0]] = data[[nutrient_list[0]]] * 0.01 # convert first column from kcal/100g to kcal/g
for nutr in nutrient_list[1:]:
    # multiply each column the vector of kcal per 100 g ready2eat / kcal per 100 g
    data.loc[:, nutr] *= data.loc[:, nutrient_list[0]] / data_frame.loc[:, ENERGY_COLNAME]
data = data.rename(columns={nutrient_list[0]:'kcal'})
nutrient_list[0] = 'kcal'
data[pd.isnull(data)] = 0

#%%
pdf_pages = PdfPages('pareto.pdf')

for food_subset in SUBSETS_TO_REPLACE:
    print 'Attempting to replace: ' + str(food_subset)
    
    # calculate how many grams of a certain animal-based food product is consumed in MAD
    MAD = pd.Series(data_frame.loc[:, MAD_COLNAME] / data.loc[:, 'kcal'], food_abbrev)
    MAD[pd.isnull(MAD)] = 0
    MAD_to_replace = MAD[food_subset]
    for f in food_subset:
        print '%.1f grams of %s are in MAD per day' % (MAD_to_replace[f], f)
    
    # calculate the fraction of this animal portion from the total MAD diet
    MAD_fraction = sum(data_frame.loc[food_subset, MAD_COLNAME]) / data_frame[[MAD_COLNAME]].sum(numeric_only=True)
    
    # adjust the vector of required nutrients to more realistic values
    b_MAD_to_replace = data.loc[food_subset, :].transpose().dot(MAD_to_replace)
    b_const = b_MAD_to_replace.copy()
    b_const['Sugar_Tot_(g)'] = 90                    # set a more realistic sugar  upper bound
    b_const['Sodium_(mg)'] = 600                     # set a more realistic sodium upper bound (note that the rightful by-mass portion of the total 2300 mg/d of MAD_a is 790 mg/d 
    b_const['Calcium_(mg)'] = 1100.0 * MAD_fraction  # Calcium, see https://www.nlm.nih.gov/medlineplus/magazine/issues/winter11/articles/winter11pg12.html
    b_const['Niacin_(mg)'] = 15.0 * MAD_fraction     # niacin,  see https://www.nlm.nih.gov/medlineplus/ency/article/002409.htm
    b_const['Selenium_(mcg)'] = 55 * MAD_fraction    # set a more realistic selenium lower bound, see https://ods.od.nih.gov/factsheets/Selenium-HealthProfessional/
    
    b_const['Lipid_Tot_(g)'] = 0                     # we don't care about fat specifically, just calories
    b_const['Fiber_TD_(g)'] = 0                      # doesn't exist in animals anyway
    b_const['Vit_B12_(mcg)'] = 0                     # impossible to get from plants
    
    
    # create the LP using PuLP
    MINIMIZATION_NUTRIENTS = ['kcal', 'FA_Sat_(g)', 'Cholestrl_(mg)', 'Sugar_Tot_(g)', 'Sodium_(mg)']
    
    C  = -np.ones((1, data.shape[1]))
    
    lp = pulp.LpProblem('replace_animal_diet', pulp.LpMinimize)
    x = pulp.LpVariable.dicts('mass', food_abbrev)
    x_vec = map(x.get, food_abbrev)
    
    Ax = data.transpose().dot(x_vec)
    for nutr in set(nutrient_list).difference(MINIMIZATION_NUTRIENTS):
        lp += pulp.lpSum(Ax[nutr]) >= b_const[nutr], nutr
    
    # generally set the upper bound on mass of a single food to 50
    # unless it is in the "hated foods" categorie
    upper_bounds = dict([(f, 50) for f in food_abbrev])
    upper_bounds['garlic'] = 5
    upper_bounds['asparagus'] = 20
    upper_bounds['broccoli'] = 20
    upper_bounds['collard'] = 20
    upper_bounds['macadamia'] = 20
    upper_bounds['spinach'] = 20
   # upper_bounds['tofu'] = 20
   # upper_bounds['soybean'] = 20

    for a in ANIMALS: # set all animal-based foods to 0 (i.e. don't allow any)
        upper_bounds[a] = 0
    
    for f, ub in upper_bounds.iteritems():
        lp += x[f] <= ub, '%s_ub' % f
        lp += x[f] >= 0, '%s_lb' % f
    
    # Draw the Pareto plot to see tradeoff between the two objective (mass and kcal)
    
    N = 1000
    result_data = np.zeros((N, 3))
    
    for i, log_alpha in enumerate(np.linspace(-2, 2, N)):
        alpha = 10.0**log_alpha
        lp.setObjective(pulp.lpSum(x) + alpha * pulp.lpSum(Ax[['kcal']]))
        pulp_solver = pulp.CPLEX(msg=0)
        lp.solve(pulp_solver)
        if lp.status != pulp.LpStatusOptimal:
            raise pulp.solvers.PulpSolverError("cannot replace these animal foods")
        SANE = pd.Series(map(pulp.value, map(x.get, food_abbrev)), food_abbrev)
        result_data[i, :] = log_alpha, SANE.sum(), data[['kcal']].transpose().dot(SANE)
    
    result_data = pd.DataFrame(result_data, columns=('log(alpha)', 'mass', 'kcal'))
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(result_data[['mass']], result_data[['kcal']], '.-')
    ax.legend(['plant alternatives'])
    ax.set_xlabel('total diet mass (gr)')
    ax.set_ylabel('total diet energy (kcal)')
    ax.set_title('Pareto plot for mass vs. energy objectives')
    ax.plot(MAD_to_replace.sum(), b_MAD_to_replace['kcal'], 'r.')
    ax.text(MAD_to_replace.sum(), b_MAD_to_replace['kcal'], '+'.join(food_subset), horizontalalignment='right', verticalalignment='top')
    ax.set_xlim([0, None])
    ax.set_ylim([0, None])
    pdf_pages.savefig(fig)

pdf_pages.close()