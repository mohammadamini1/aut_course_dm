#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ابتدا داده ها را خوانده و اطلاعات کلی را بررسی میکنیم

# In[13]:


# Import dataset
df = pd.read_csv("./diabetes.csv")
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ','_')
df = df.drop(["unnamed:_0"], axis=1)
df.info()
df.isnull().sum()


# <div dir=rtl>
# 
# 
# <b>‫حذف داده های پوچ:
# </b>
# 
# چون تعداد داده های پوچ (20 خطا در 6 سطر) نسبت به تعداد سطر ها (70692) بسیار کمتر است، پس حذف این سطر ها راحت ترین کار است و به داده ها نیز آسیبی وارد نمیکند و مدت زمان پردازشی کمتری نسبت به روش های جایگزینی دیگر دارد
# </div>
# 
# 

# In[14]:


## drop rows with nan
df_nan = df[df.isna().any(axis=1)]
df = df.drop(df_nan.index)
## reset index
df = df.reset_index()
df = df.drop(['index'], axis=1)
df


# <div dir=rtl>
#     <b>
# نرمالیزه کردن:
#     </b>
# 
#  داده های هر ستون را بررسی کرده و در صورت نیاز اصلاح میکنیم. مثلا بررسی میکنیم ستون هایی که باینری هستند فقط شامل 0 و 1 باشند یا داده ها case sensitive نباشند
# 
# سپس داده ها را نرمال میکنیم
# 
# </div>

# In[15]:


## replace white spaces & lower case data
df = df.replace(' ', '_', regex=True)
for col in ['general_health', 'education', 'income']:
    df[col] = df[col].str.lower()

## normalization
df['bmi'] = df['bmi'] // 10
df['mental_health'] = df['mental_health'] // 3
df['physical_health'] = df['physical_health'] // 3
df['age'] = df['age'] // 2

## get all unique values in columns
pd.unique(df['general_health'])
for k in df.keys():
    print("->{}({}): {}".format(k, len(df[k]), pd.unique(df[k])))


# <div dir=rtl>
# یافتن ویژگی های دسته بندی شده با
#     <b>One Hot Encoding</b>
# 
# 
# 
# </div>

# In[16]:


## one hot encoding categorical columns
for col in ['general_health', 'sex', 'education', 'income']:
    one_hot = pd.get_dummies(df[col], prefix=col)
    ## add new columns
    df = df.join(one_hot)
    ## drop col
    df = df.drop(col, axis=1)

df.info()
df


# <div dir=rtl> و ستون diabetes_binary را جدا میکنیم. سپس به دو دسته train و test جهت آموزش تقسیم میکنیم.
# 
# 
# 
# </div>

# In[17]:


from sklearn.model_selection import train_test_split
## pop result column to y
y = df.pop('diabetes_binary')
X = df
## split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2
                                                    , random_state=123
                                                   )


# <div dir=rtl>
# 
# <b>ساخت مدل طبقه بند</b>
#     
# 
# 
# </div>

# In[18]:


from xgboost import XGBClassifier

model = XGBClassifier(
    learning_rate =0.1,
    max_depth=4,
    n_estimators=200,
    subsample=0.5,
    colsample_bytree=1,
    random_state=123,
    eval_metric='auc',
    verbosity=1,
)
model.fit(
    X_train, y_train, 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    early_stopping_rounds=10,
)
model


# In[19]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

## cal train accuracy
y_pred = model.predict(X_train)
predictions = [value for value in y_pred]
accuracy = accuracy_score(y_train, predictions)
print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

## cal test accuracy
y_pred = model.predict(X_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

## confusion matrix
data = {'y_test': y_test, 'y_pred': y_pred,}
confusion_matrix_df = pd.DataFrame(data, columns=['y_test','y_pred'])
confusion_matrix = pd.crosstab(confusion_matrix_df['y_test'], confusion_matrix_df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
print()
print(confusion_matrix)

## cal precision & recall
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
precision = precision[1];recall = recall[1]
print('\nrecall_Score:', recall)
print('precision_Score:', precision)


# <div dir=rtl>
# 
# حال با کمک GridSearchCV امتیاز حالات مختلف را بررسی میکنیم
# 
# 
# </div>

# In[20]:


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
def my_roc_auc_score(model, X, y):
    return roc_auc_score(y, model.predict_proba(X)[:,1])
params = {
    'learning_rate': [0.02, 0.05, 0.1, 0.3],
    'max_depth': [2, 3, 4],
    'n_estimators': [100, 200, 300],
    'colsample_bytree': [0.8, 1],
}

gsearch = GridSearchCV(
    estimator=model,
    param_grid=params,
    scoring=my_roc_auc_score,
    cv=3,
    n_jobs=-1,
)
gsearch.fit(X_train, y_train)



# In[21]:


print(gsearch.best_score_)
print(gsearch.score(X_train, y_train))
print(gsearch.score(X_test, y_test))
print(gsearch.best_estimator_)

print("\ngsearch best params:")
print(gsearch.best_params_)


# 
# <div dir=rtl>
# بهترین مجموعه پارامتر هارا بدست آوردیم.
# 
# :دقت مدل و ماتریس درهم ریختگی را محاسبه میکنیم
# 
# </div>
# 
# 

# In[22]:


model2 = gsearch
# model2 = XGBClassifier(
#     learning_rate =0.,
#     max_depth=,
#     n_estimators=200,
#     subsample=0.5,
#     colsample_bytree=1,
#     random_state=123,
#     eval_metric='auc',
#     verbosity=1,
# )
# model2.fit(
#     X_train, y_train, 
#     eval_set=[(X_train, y_train), (X_test, y_test)], 
#     early_stopping_rounds=10,
# )



## cal train accuracy
y_pred = model2.predict(X_train)
predictions = [value for value in y_pred]
accuracy = accuracy_score(y_train, predictions)
print("Train Accuracy: %.2f%%" % (accuracy * 100.0))

## cal test accuracy
y_pred = model2.predict(X_test)
predictions = [value for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (accuracy * 100.0))

## confusion matrix
data = {'y_test': y_test, 'y_pred': y_pred,}
confusion_matrix_df = pd.DataFrame(data, columns=['y_test','y_pred'])
confusion_matrix = pd.crosstab(confusion_matrix_df['y_test'], confusion_matrix_df['y_pred'], rownames=['Actual'], colnames=['Predicted'])
print()
print(confusion_matrix)

## cal precision & recall
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred)
precision = precision[1];recall = recall[1]
print('\nrecall_Score:', recall)
print('precision_Score:', precision)


# <div dir=rtl>
# 
#     
#     
# 
# هر پلات رابطه یک پارامتر با امتیاز میانگین را نشان میدهد.
# 
# مثلا ضریب یادگیری روند کاهشی داشته اما n_estimators افزایش است
#     
#     
#    
# </div>
# 
# 
# 
# 
# 

# In[23]:


def plot_search_results(grid):
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']

    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid

    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Score per parameter')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()

plot_search_results(gsearch)

