import fredapi    
fred = fredapi.Fred(api_key= 'ae9eac413fee6bc7cbe0747b78d0b32c' )

import sys, os
sys.path.append('../')
from linear import Regression

data = fred.get_series('INDPRO').to_frame().rename(columns={0:'INDPRO'})\
    .merge(fred.get_series('GDP').to_frame().rename(columns={0:'GDP'}), left_index=True, right_index=True)\
    .merge(fred.get_series('RSXFS').to_frame().rename(columns={0:'RSXFS'}), left_index=True, right_index=True)
print(data.head())

reg = Regression(data = data, dep_var='GDP',  verbose = False)
lm = reg.examine_stats()
print(lm.summary())
print(lm.params)
# reg.plot_linear_univariate()
reg.reg_plots()
# reg.confidence_intervals()
print(reg.get_params())
regressor = reg.train_model(SEED=1)
df_preds  = reg.predict()
mae, mape, mse, rmse = reg.evaluate_predictions()
reg.plot_residuals()
