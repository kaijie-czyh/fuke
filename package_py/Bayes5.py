import numpy as np

# 混合式
class NaiveBayes:
    def __init__(self, alpha=1.0):
        """
        初始化混合朴素贝叶斯分类器，alpha 是拉普拉斯平滑的参数（仅用于离散特征）。
        """
        self.alpha = alpha
        self.class_prior = {}  # P(c)
        self.discrete_cond_prob = {}  # 离散特征的条件概率（多项式分布）
        self.continuous_mean = {}  # 连续特征的均值
        self.continuous_var = {}  # 连续特征的方差
        self.feature_types = {}  # 记录每个特征是离散还是连续
        self.classes = []  # 样本标签们

    def fit(self, X, y, feature_types):
        """
        训练混合朴素贝叶斯分类器。

        参数:
        X -- 训练数据的特征矩阵，形状为 (n_samples, n_features)
        y -- 训练数据的标签向量，形状为 (n_samples,)
        feature_types -- 一个长度为 n_features 的列表，指定每个特征是离散（True）还是连续（False）
        """
        n_samples, n_features = X.shape
        self.classes, counts = np.unique(y, return_counts=True)

        # 计算类先验概率
        self.class_prior = {cls: count / n_samples for cls, count in zip(self.classes, counts)}

        # 初始化条件概率和连续特征的统计量
        self.discrete_cond_prob = {cls: {} for cls in self.classes}
        self.continuous_mean = {cls: np.zeros(n_features) for cls in self.classes}
        self.continuous_var = {cls: np.zeros(n_features) for cls in self.classes}
        self.feature_types = feature_types

        for cls in self.classes:
            X_cls = X[y == cls]
            n_samples_cls = X_cls.shape[0]

            for feature_idx in range(n_features):
                
                if self.feature_types[feature_idx]:  # 离散特征
                    feature_values = X_cls[:, feature_idx]
                    unique_values, counts = np.unique(feature_values, return_counts=True)
                    # 多项式贝叶斯条件概率计算
                    total_count_cls_feature = counts.sum() + self.alpha * len(unique_values)
                    prob = {val: (count + self.alpha) / total_count_cls_feature
                            for val, count in zip(unique_values, counts)}
                    self.discrete_cond_prob[cls][feature_idx] = prob
                else:  # 连续特征
                    mean = X_cls[:, feature_idx].mean()
                    var = X_cls[:, feature_idx].var()
                    self.continuous_mean[cls][feature_idx] = mean
                    self.continuous_var[cls][feature_idx] = var

    def _gaussian_prob(self, x, mean, var):
        """
        计算高斯分布的概率密度函数。

        参数:
        x -- 要计算概率的点
        mean -- 高斯分布的均值
        var -- 高斯分布的方差

        返回:
        prob -- 给定点在高斯分布下的概率密度
        """
        var = np.maximum(var, 1e-10)  # 确保方差为正
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(- (x - mean) ** 2 / (2 * var))
        return np.maximum(exponent * coeff, 1e-20)

    def predict_proba(self, X):
        """
        计算新数据属于每个类别的概率。

        参数:
        X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)

        返回:
        prob -- 预测结果的概率矩阵，形状为 (n_samples, n_classes)
        """
        prob = []

        for sample in X:
            class_prob = {}

            for cls in self.classes:
                prior = np.log(self.class_prior[cls])
                conditional = 0.0

                for feature_idx, value in enumerate(sample):
                    if self.feature_types[feature_idx]:  # 离散特征
                        if value in self.discrete_cond_prob[cls][feature_idx]:
                            conditional += np.log(self.discrete_cond_prob[cls][feature_idx][value])
                        else:
                            # 如果值不在离散特征的观测值中，可以使用拉普拉斯平滑处理（这里简单处理为取最接近的值）
                            closest_value = min(self.discrete_cond_prob[cls][feature_idx].keys(), key=lambda x: abs(x - value))
                            conditional += np.log(self.discrete_cond_prob[cls][feature_idx][closest_value])
                    else:  # 连续特征
                        mean = self.continuous_mean[cls][feature_idx]
                        var = self.continuous_var[cls][feature_idx]
                        conditional += np.log(self._gaussian_prob(value, mean, var))

                class_prob[cls] = prior + conditional

            # 将对数概率转换为概率并归一化
            class_prob_exp = np.exp(np.array(list(class_prob.values())))
            class_prob_norm = class_prob_exp / class_prob_exp.sum()

            prob.append(class_prob_norm)

        return np.array(prob)

    def predict(self, X):
        """
        对新的数据进行预测，并返回类别标签。

        参数:
        X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)

        返回:
        y_pred -- 预测结果的标签向量，形状为 (n_samples,)
        """
        prob = self.predict_proba(X)
        indices = np.argmax(prob, axis=1)  # 获取每个样本最可能类别的索引
        y_pred = self.classes[indices]     # 使用索引从self.classes中获取对应的标签

        return y_pred
    
# import numpy as np
# # 混合式
# class NaiveBayes:
#     def __init__(self, alpha=1.0):
#         """
#         初始化混合朴素贝叶斯分类器，alpha 是拉普拉斯平滑的参数（仅用于离散特征）。
#         """
#         self.alpha = alpha
#         self.class_prior = {}  # P(c)
#         self.discrete_cond_prob = {}  # 离散特征的条件概率
#         self.continuous_mean = {}  # 连续特征的均值
#         self.continuous_var = {}  # 连续特征的方差
#         self.feature_types = {}  # 记录每个特征是离散还是连续
#         self.classes = []  # 样本标签们

#     def fit(self, X, y, feature_types):
#         """
#         训练混合朴素贝叶斯分类器。

#         参数:
#         X -- 训练数据的特征矩阵，形状为 (n_samples, n_features)
#         y -- 训练数据的标签向量，形状为 (n_samples,)
#         feature_types -- 一个长度为 n_features 的列表，指定每个特征是离散（True）还是连续（False）
#         """
#         n_samples, n_features = X.shape
#         self.classes, counts = np.unique(y, return_counts=True)

#         # 计算类先验概率
#         self.class_prior = {cls: count / n_samples for cls, count in zip(self.classes, counts)}

#         # 初始化条件概率和连续特征的统计量
#         self.discrete_cond_prob = {cls: {} for cls in self.classes}
#         self.continuous_mean = {cls: np.zeros(n_features) for cls in self.classes}
#         self.continuous_var = {cls: np.zeros(n_features) for cls in self.classes}
#         self.feature_types = feature_types

#         for cls in self.classes:
#             X_cls = X[y == cls]
#             n_samples_cls = X_cls.shape[0]

#             for feature_idx in range(n_features):
                

#                 if self.feature_types[feature_idx]:  # 离散特征
#                     feature_values = X_cls[:, feature_idx]
#                     unique_values, counts = np.unique(feature_values, return_counts=True)
#                     prob = {val: (count + self.alpha) / (n_samples_cls + self.alpha * len(unique_values))
#                             for val, count in zip(unique_values, counts)}
#                     self.discrete_cond_prob[cls][feature_idx] = prob
#                 else:  # 连续特征
#                     mean = X_cls[:, feature_idx].mean()
#                     var = X_cls[:, feature_idx].var()
#                     self.continuous_mean[cls][feature_idx] = mean
#                     self.continuous_var[cls][feature_idx] = var
#         # print(feature_types)

#     def _gaussian_prob(self, x, mean, var):
#         """
#         计算高斯分布的概率密度函数。

#         参数:
#         x -- 要计算概率的点
#         mean -- 高斯分布的均值
#         var -- 高斯分布的方差

#         返回:
#         prob -- 给定点在高斯分布下的概率密度
#         """
#         var = np.maximum(var, 1e-10)  # 确保方差为正
#         coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
#         exponent = np.exp(- (x - mean) ** 2 / (2 * var))
#         return np.maximum(exponent * coeff, 1e-20)

#     def predict_proba(self, X):
#         """
#         计算新数据属于每个类别的概率。

#         参数:
#         X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)

#         返回:
#         prob -- 预测结果的概率矩阵，形状为 (n_samples, n_classes)
#         """
#         prob = []

#         for sample in X:
#             class_prob = {}

#             for cls in self.classes:
#                 prior = np.log(self.class_prior[cls])
#                 conditional = 0.0

#                 for feature_idx, value in enumerate(sample):
#                     if self.feature_types[feature_idx]:  # 离散特征
#                         if value in self.discrete_cond_prob[cls][feature_idx]:
#                             conditional += np.log(self.discrete_cond_prob[cls][feature_idx][value])
#                         else:
#                             # 如果值不在离散特征的观测值中，可以使用拉普拉斯平滑处理（这里简单处理为取最接近的值）
#                             closest_value = min(self.discrete_cond_prob[cls][feature_idx].keys(), key=lambda x: abs(x - value))
#                             conditional += np.log(self.discrete_cond_prob[cls][feature_idx][closest_value])
#                     else:  # 连续特征
#                         mean = self.continuous_mean[cls][feature_idx]
#                         var = self.continuous_var[cls][feature_idx]
#                         conditional += np.log(self._gaussian_prob(value, mean, var))
#                 # class_prob[cls] =conditional
#                 class_prob[cls] = prior + conditional

#             # 将对数概率转换为概率并归一化
#             class_prob_exp = np.exp(np.array(list(class_prob.values())))
#             class_prob_norm = class_prob_exp / class_prob_exp.sum()

#             prob.append(class_prob_norm)

#         return np.array(prob)

#     def predict(self, X):
#         """
#         对新的数据进行预测，并返回类别标签。

#         参数:
#         X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)

#         返回:
#         y_pred -- 预测结果的标签向量，形状为 (n_samples,)
#         """
#         prob = self.predict_proba(X)
#         indices = np.argmax(prob, axis=1)  # 获取每个样本最可能类别的索引
#         y_pred = self.classes[indices]     # 使用索引从self.classes中获取对应的标签

#         return y_pred

# # 高斯贝叶斯
# import numpy as np  
  
# class NaiveBayes:  
#     def __init__(self):  
#         """  
#         初始化高斯朴素贝叶斯分类器。  
#         """  
#         self.class_prior = {} #P（c） 
#         self.mean = {}  
#         self.var = {}  
#         self.classes = []  #样本标签们
  
#     def fit(self, X, y):  
#         """  
#         训练高斯朴素贝叶斯分类器。  
  
#         参数:  
#         X -- 训练数据的特征矩阵，形状为 (n_samples, n_features)  
#         y -- 训练数据的标签向量，形状为 (n_samples,)  
#         """  
#         n_samples, n_features = X.shape  
#         self.classes, counts = np.unique(y, return_counts=True)  
  
#         # 计算类先验概率  
#         self.class_prior = {cls: count / n_samples for cls, count in zip(self.classes, counts)}  
  
#         # 初始化均值和方差  
#         self.mean = {cls: np.zeros(n_features) for cls in self.classes}  
#         self.var = {cls: np.zeros(n_features) for cls in self.classes}  
  
#         for cls in self.classes:  
#             X_cls = X[y == cls]  
#             self.mean[cls] = X_cls.mean(axis=0)  
#             self.var[cls] = X_cls.var(axis=0)  
#     def _gaussian_prob(self, x, mean, var):  
#         var = np.maximum(var, 1e-10)  # Ensure variance is positive  
#         coeff = 1.0 / np.sqrt(2.0 * np.pi * var)  
#         exponent = np.exp(- (x - mean) ** 2 / (2 * var))  
#         return np.maximum(exponent * coeff, 1e-20)
#     # def _gaussian_prob(self, x, mean, var):  
#     #     """  
#     #     计算高斯分布的概率密度函数。  
  
#     #     参数:  
#     #     x -- 要计算概率的点  
#     #     mean -- 高斯分布的均值  
#     #     var -- 高斯分布的方差  
  
#     #     返回:  
#     #     prob -- 给定点在高斯分布下的概率密度  
#     #     """  
       
#     #     # var = np.where(var == 0, 1e-9, var)
#     #     # 检查方差是否小于阈值
#     #     var = np.where(var < 1e-10, 1e-10, var)

#     #     coeff = 1.0 / np.sqrt(2.0 * np.pi * var)  
#     #     exponent = np.exp(- (x - mean) ** 2 / (2 * var)) 
#     #     # 添加一个小的常数以避免对数下溢  
#     #     stable_out = np.maximum(exponent*coeff, -1e20)
        

#     #     return stable_out 
  
#     def predict_proba(self, X):  
#         """  
#         计算新数据属于每个类别的概率。  
  
#         参数:  
#         X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)  
  
#         返回:  
#         prob -- 预测结果的概率矩阵，形状为 (n_samples, n_classes)  
#         """  
#         prob = []  
  
#         for sample in X:  
#             class_prob = {}  
  
#             for cls in self.classes:  
#                 prior = np.log(self.class_prior[cls])  
#                 conditional = 0.0  
  
#                 for feature_idx, value in enumerate(sample):  
#                     mean = self.mean[cls][feature_idx]  
#                     var = self.var[cls][feature_idx]  

#                     conditional += np.log(self._gaussian_prob(value, mean, var))  
  
#                 class_prob[cls] = prior + conditional  
  
#             # 将对数概率转换为概率并归一化  
#             class_prob_exp = np.exp(np.array(list(class_prob.values())))  
#             class_prob_norm = class_prob_exp / class_prob_exp.sum()  
  
#             prob.append(class_prob_norm)  
  
#         return np.array(prob)  
#     def predict(self, X):  
#       """  
#       对新的数据进行预测，并返回类别标签。  
    
#       参数:  
#       X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)  
    
#       返回:  
#       y_pred -- 预测结果的标签向量，形状为 (n_samples,)  
#       """  
#       prob = self.predict_proba(X)  
#       indices = np.argmax(prob, axis=1)  # 获取每个样本最可能类别的索引  
#       y_pred = self.classes[indices]     # 使用索引从self.classes中获取对应的标签  
    
#       return y_pred






# import numpy as np  
  
# class NaiveBayes:  
#     def __init__(self, alpha=1.0):  
#         """  
#         初始化朴素贝叶斯分类器，alpha 是拉普拉斯平滑的参数。  
#         """  
#         self.alpha = alpha  
#         self.class_prior = {}  #P(c)
#         self.conditional_prob = {}  
#         self.classes = []  #样本标签们
  
#     def fit(self, X, y):  
#         """  
#         训练朴素贝叶斯分类器。  
          
#         参数:  
#         X -- 训练数据的特征矩阵，形状为 (n_samples, n_features)  
#         y -- 训练数据的标签向量，形状为 (n_samples,)  
#         """  
#         # n_samples训练样本数，n_features特征数
#         n_samples, n_features = X.shape  
#         self.classes, counts = np.unique(y, return_counts=True)  
          
#         # 计算类先验概率，返回的是类别和对应概率的字典
#         self.class_prior = {cls: count / n_samples for cls, count in zip(self.classes, counts)}  
          
#         # 初始化条件概率  
#         self.conditional_prob = {cls: {} for cls in self.classes}  
       
#         # print("self.conditional_prob",self.conditional_prob)
#         for cls in self.classes:
#             X_cls = X[y == cls]  
#             n_samples_cls = X_cls.shape[0]  
              
#             for feature_idx in range(n_features):  
#                 feature_values = X_cls[:, feature_idx]  
#                 unique_values, counts = np.unique(feature_values, return_counts=True)  
#                 # 不应用拉普拉斯平滑  
#                 prob = {val: count / n_samples_cls
#                         for val, count in zip(unique_values, counts)}    
#                 # # 应用拉普拉斯平滑  
#                 # prob = {val: (count + self.alpha) / (n_samples_cls + self.alpha * len(unique_values))  
#                 #         for val, count in zip(unique_values, counts)}  
                  
#                 self.conditional_prob[cls][feature_idx] = prob  
#         print('self.conditional_prob',self.conditional_prob)

#     def find_closest_value(self,dictionary, x):
#         # 初始化最小距离为一个较大的值
#         min_distance = float('inf')
#         closest_key = None
        
#         # 遍历字典的键
#         for key in dictionary:
#             # 计算当前键与x的绝对距离
#             distance = abs(key - x)
            
#             # 如果当前距离小于最小距离，则更新最小距离和对应的键
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_key = key
#         print("closest_key",closest_key)
#         # 返回距离x最近的键对应的值
#         return dictionary[closest_key]
  
#     def predict(self, X):  
#         """  
#         对新的数据进行预测。  
          
#         参数:  
#         X -- 要预测的数据的特征矩阵，形状为 (n_samples, n_features)  
          
#         返回:  
#         y_pred -- 预测结果的标签向量，形状为 (n_samples,)  
#         """  
#         y_pred = []  
          
#         for sample in X:  
#             print("sample",sample)
#             class_posteriors = {}  
              
#             for cls in self.classes:  
#                 prior = np.log(self.class_prior[cls])  
#                 conditional = 0.0  
                  
#                 for feature_idx, value in enumerate(sample):  
#                     if value in self.conditional_prob[cls][feature_idx]:  
#                         conditional += np.log(self.conditional_prob[cls][feature_idx][value])  
#                     else:  
#                         # delta=0.0001
#                         # value_plus=value
#                         # value_minus=value
                   
#                         # while value_plus not in self.conditional_prob[cls][feature_idx] or value_minus not in self.conditional_prob[cls][feature_idx] or value<max(self.conditional_prob[cls][feature_idx],key=self.conditional_prob[cls][feature_idx].get):
#                         #     value_plus+=delta
#                         #     value_minus-=delta
                         
                            
#                         # if value_plus in self.conditional_prob[cls][feature_idx]:
#                         #     conditional += np.log(self.conditional_prob[cls][feature_idx][value_plus])
#                         #     continue
#                         # if value_minus in self.conditional_prob[cls][feature_idx]:
#                         #     conditional += np.log(self.conditional_prob[cls][feature_idx][value_minus])
#                         #     continue
#                         print("value",value)
#                         # 如果值未在训练数据中见过，
#                         conditional += np.log(self.find_closest_value(self.conditional_prob[cls][feature_idx],value))

#                         # # 如果值未在训练数据中见过，使用拉普拉斯平滑后的概率  
#                         # conditional += np.log(self.alpha / (len(self.conditional_prob[cls][feature_idx]) * self.alpha))  
#                         # print("conditional",conditional)

#                 class_posteriors[cls] = prior + conditional
#             # print("class_posteriors",class_posteriors)
#             # 选择具有最高后验概率的类  
#             y_pred.append(max(class_posteriors, key=class_posteriors.get))  
          
#         return np.array(y_pred)  