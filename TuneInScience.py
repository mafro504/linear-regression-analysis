#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from scipy import stats


#READ IN DATA
TuneIn = pd.read_csv('TuneIn_HistoricalData5.csv')


#CHECK DATA
#first few lines look correct
print (TuneIn.head())

#data types look correct (all integer or float)
print (TuneIn.info())

#initial stats of data
print (TuneIn.describe())

#view column headers
print (TuneIn.columns)

#visualizations
sns.pairplot(TuneIn)

#visualize outcome variable
sns.distplot(TuneIn['Target_Lift'])


#CORRELATION
TuneInCorrs = TuneIn.corr()
TuneInCorrs.to_excel('TuneInCorrelations2.xlsx')


#TRAIN LINEAR REGRESSION MODEL
#set x and y based on columns
X = TuneIn[['Lift_All_Windows', 'Lift_Target_Window',
       'CampaignDuration', 'NewReturning', 'Network_CableTV',
       'Network_Sports', 'Network_PremiumTV', 'PremiereLeadTime',
       'PremiereDay', 'PremierePrimeTime', 'EpisodesMeasured',
       'Samba_Universe_Reach', 'Exposed_Households',
       'Target_Control_VTR', 'Target_Exposed_VTR',
      'Target_TuneIns', '10kTuneIns',
        'TargetViewWindow_SD', 'TargetViewWindow_L3',
       'TargetViewWindow_L7',
       'Premiere_Frequency', 'Premiere_Impressions', 'FirstView',
       'BrandReminder', 'Poll', 'PreRoll', 'Campaign_Cost']]
y = TuneIn['Target_Lift']

#split data based on training and test. 30% = test, 70% = training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#creating and training model
lm = LinearRegression()
model = lm.fit(X_train,y_train)


#EVALUATE MODEL
r_sq = model.score(X_train,y_train)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

#predictor coefficients
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
print (coeff_df)
coeff_df.to_excel('Coefficients_v2.xlsx')

#standard deviation for use in calculating standardized coefficients
print (np.std(TuneIn))

#p-values to identify statistical significance 
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


















