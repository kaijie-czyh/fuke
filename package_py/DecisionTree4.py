import math
from collections import Counter  
class Node(object):
  # def __init__(self,node_D,feature_index=None,parent=None,children=[],leaf_value=None):
  
    def __init__(self, node_D, feature_val=None, feature_index=None, parent=None, children=None, leaf_value=None):  
        self.node_D = node_D #节点数据  
        self.feature_val=feature_val #属性值
        self.feature_index = feature_index #属性索引 
        self.parent = parent  #父节点
        self.children = children if children is not None else [] # 子节点们  
        self.leaf_value = leaf_value #叶子节点的值（好瓜坏瓜）

def print_tree( node, level=0):  
    print(" " * level * 4 + f"Node_D_val: {node.feature_val if node.node_D else []}, Feature Index: {node.feature_index}, node: {node.children}, Leaf Value: {node.leaf_value}")  
    if node.children==[]:
      return 0
    for child in node.children:  
        print_tree(child, level + 1)  
    return 0
class TreeGenerate(object):
  def __init__(self,D,A):

     self.D=D #数据集
     self.A=A #属性集
     self.root=None
     self._build_tree(D,A)

    # calculate_pk([[1,1],[2,2]])
    # {'1': 0.5, '2': 0.5}
  def calculate_pk(self,D):
    A=[]
    for i in range(len(D)): 
      # 提取最后一列数据
      A.append(D[i][-1])
        
    # 创建一个字典来存储每个值（转换为字符串后）出现的次数  
    count_dict = {}     
    # 统计每个值（转换为字符串后）出现的次数  
    for item in A:  
        # 将项转换为字符串（这里简单地将所有非字符串类型转换为字符串）  
        # 注意：这可能会导致类型信息的丢失，例如1和1.0会被视为相同的字符串'1'  
        item_str = str(item)  
        if item_str in count_dict:  
            count_dict[item_str] += 1  
        else:  
            count_dict[item_str] = 1       
    # 计算列表中元素的总数  
    total_count = len(A)         
    # 计算每个值（转换为字符串后）出现的次数占总次数的比例  
    proportion_dict = {item_str: count / total_count for item_str, count in count_dict.items()}     
    # 打印结果  
    return proportion_dict

  # print(calculate_Entropy(['是','是','是','是','是','是','是','是','否','否','否','否','否','否','否','否','否']))
  # 0.9975025463691153
  def calculate_Entropy(self,D):

    p = self.calculate_pk(D)
    entropy = 0
    for i in p.values():
      entropy += i*math.log(i,2)
    return -entropy
  # 计算第二个（索引1）属性的信息增益
  # calculate_Gain(0,[[1,2,3,0],[2,3,1,0],[3,1,2,0],[1,2,3,0],[2,3,1,0],[3,1,2,1]],1)
  # 0.3166890883150208 ['2', '3', '1']
  # [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
  def calculate_Gini(self,D):
    p = self.calculate_pk(D)
    gini = 1
    for i in p.values():

      gini -= i**2
    return gini
  def calculate_Gini_Index(self,D,a):
      if D==[]:
        print('D is empty')
        return 0
      if a<0 or a>=len(D[0]):
        print('a is not in D')
        return 0
      
      Ent_Di={}
      count_dict = {}  # 记录指定属性的取值的次数
      attribute_name=[]   # 记录指定属性的取值
      
      # 统计每个值（转换为字符串后）出现的次数  
      for item in D:  
      # 将项转换为字符串（这里简单地将所有非字符串类型转换为字符串）  
      # 注意：这可能会导致类型信息的丢失，例如1和1.0会被视为相同的字符串'1'  
        item_str = str(item[a])  
        if item_str in count_dict:  
          count_dict[item_str] += 1  
          Ent_Di[item_str].append(item)
        else:  
          count_dict[item_str] = 1
          Ent_Di[item_str]=[]
          Ent_Di[item_str].append(item)
          attribute_name.append(item_str)       

      # 计算列表中元素的总数
      total_count = len(D)
      # 计算每个值（转换为字符串后）出现的次数占总次数的比例
      attribute_weight = [count / total_count for count in count_dict.values()]
      # print(attribute_weight)
      # print(attribute_name)
      gini_index=0
      for i in range(len(attribute_name)):
        gini_index+=attribute_weight[i]*self.calculate_Gini(Ent_Di[attribute_name[i]])

      return gini_index,attribute_name
  def calculate_Gain(self,D,a):
      if D==[]:
        print('D is empty')
        return 0
      if a<0 or a>=len(D[0]):
        print('a is not in D')
        return 0
      
      Ent_D = self.calculate_Entropy(D)
      # print(Ent_D)
      Ent_Di={}
      count_dict = {}  # 记录指定属性的取值的次数
      attribute_name=[]   # 记录指定属性的取值
      
      # 统计每个值（转换为字符串后）出现的次数  
      for item in D:  
      # 将项转换为字符串（这里简单地将所有非字符串类型转换为字符串）  
      # 注意：这可能会导致类型信息的丢失，例如1和1.0会被视为相同的字符串'1'  
        item_str = str(item[a])  
        if item_str in count_dict:  
          count_dict[item_str] += 1  
          Ent_Di[item_str].append(item)
        else:  
          count_dict[item_str] = 1
          Ent_Di[item_str]=[]
          Ent_Di[item_str].append(item)
          attribute_name.append(item_str)       

      # 计算列表中元素的总数
      total_count = len(D)
      # 计算每个值（转换为字符串后）出现的次数占总次数的比例
      attribute_weight = [count / total_count for count in count_dict.values()]
      # print(attribute_weight)
      # print(attribute_name)
      gain=Ent_D
      for i in range(len(attribute_name)):
        gain-=attribute_weight[i]*self.calculate_Entropy(Ent_Di[attribute_name[i]])

      return gain,attribute_name

  # 选择最大信息增益属性
  # choose_A(0,[[2,2,3,0],[2,2,1,0],[2,2,2,0],[1,2,3,0],[3,3,3,0],[3,1,2,1]],[0,1,2])
  # (1, ['2', '3', '1'])
  def choose_A_by_Gini(self,D,A):
        # 由于最大的gini_index不会超过A属性取值的个数
        min_gini=len(A)
        for i in range(len(A)):
          gini_index,attribute_name=self.calculate_Gini_Index(D,i)
          gain,_=self.calculate_Gain(D,i)
          # 考虑等于0，是因为在外层遍历中并未考虑D在指定特征列的取值已经全部相同的情况
          if gini_index<=min_gini:
            min_gini=gini_index
            min_attribute_index=i
            min_attribute_name=attribute_name
        
 
        return min_attribute_index,min_attribute_name,min_gini,gain
  # def __init__(self,node_D,feature_index,parent=None,childs=[],leaf_value=None):
  # self.node_D=node_D #节点数据
  # self.feature_index = feature_index #属性索引
  # self.parent = parent #父节点
  # self.childs = childs #子节点们
  # self.leaf_value = leaf_value  #叶子节点的值（好瓜坏瓜）
  # 递归构建决策树  
  def _build_tree(self, D, A, parent=None, now_feature_val=None):  
      # 如果所有样本都属于同一类，则返回叶子节点  
      if len(set([item[-1] for item in D])) == 1:  
          leaf_value = D[0][-1]  

          node=Node(D, feature_val=now_feature_val,leaf_value=leaf_value)  
          node.parent=parent
          parent.children.append(node)
          

          return node
      # 如果没有更多特征可以选择，则返回叶子节点（多数类投票）  
      if not A:  
          leaf_value = max(self.calculate_pk(D), key=self.calculate_pk(D).get)  
     
          node=Node(D, feature_val=now_feature_val, leaf_value=leaf_value)  
          node.parent=parent
          parent.children.append(node)
          
          return node  
      if D==[]:

        leaf_value = max(self.calculate_pk(D), key=self.calculate_pk(D).get)
        node=Node(D, feature_val=now_feature_val, leaf_value=leaf_value)  
        node.parent=parent
        parent.children.append(node)
        
        return node
       

      # 选择最佳特征  
      feature_index, feature_name ,min_gini,feature_gain= self.choose_A_by_Gini(D, A) 

      if feature_gain==0:
        leaf_value = max(self.calculate_pk(D), key=self.calculate_pk(D).get)
        node=Node(D, feature_val=now_feature_val, leaf_value=leaf_value)  
        node.parent=parent
        parent.children.append(node)
        
        return node
      
      # 创建根节点 
      if self.root==None: 
       
        root = Node(D, feature_val=self.A[feature_index],feature_index=feature_index)
        self.root=root
        new_parent=root
        node=root
      else:
        
        node=Node(D, feature_val=now_feature_val,feature_index=feature_index)
        parent.children.append(node)
        node.parent=parent
        new_parent=node
      
      # 递归地为每个特征值创建子节点  
      for name in feature_name:  
    
          
          # 使用 math.isclose 进行比较，允许一个小的容差  
          sub_D = [item for item in D if math.isclose(item[feature_index], float(name))]
          # sub_D = [item for item in D if item[feature_index] == name]  
          sub_A = [i for i in A if i != feature_index]  # 移除已使用的特征  
       
          if sub_D == []:
             leaf_value = max(self.calculate_pk(D), key=self.calculate_pk(D).get)  
          
             node=Node([], feature_val=name, leaf_value=leaf_value)
            #  1111111111
             if parent!=None:
                node.parent=parent
                parent.children.append(node)
             
             continue
          
          child_node = self._build_tree(sub_D, sub_A, new_parent, name)
          # if child_node.
          # print('child_node',child_node.node_D)  
          # child_node.parent = self.now_parent  # 设置父节点  
          # self.now_parent.children.append(child_node)  # 使用标准的 children 而不是 childs  
          # print('children',self.now_parent.children)
      
            
      return child_node
  # 剪枝
  def check_accuracy(self,D,value):
    # 计算预测值与真实值的差异
    accuracy = sum(1 for d in D if d[-1] == value) / len(D)
    return accuracy
  def is_del(self,node):
    now_accuracy=self.check_accuracy(node.node_D,node.leaf_value)
    # 查看父节点多数原则
    last_column_values = [sublist[-1] for sublist in node.parent.node_D]   
    counter = Counter(last_column_values)   
    parent_leaf_value,_ = counter.most_common(1)[0] 
    # 剪枝后准确率
    after_accuracy=self.check_accuracy(node.node_D,parent_leaf_value)

    if now_accuracy-0.1 >after_accuracy:
        return False
    else:
        return True
  def del_node(self,node):
    # 删除子节点，并未释放空间，remove只会删除第一个元素，但这并不会造成问题，因为子节点必然互不相同
    return node.parent.children.remove(node)
  def bhd_tree(self,now_node):

    # 如果已经是根节点并且没有子节点，退出
    if now_node == self.root and now_node.children == []:

      return True
    # 如果是叶子节点，则直接考虑是否剪枝
    if now_node.leaf_value != None:

      if self.is_del(now_node):

        self.del_node(now_node)
     
      return True
    # 下面是针对中间节点的递归
    
    for child in reversed(now_node.children):
        self.bhd_tree(child)
    # 如果中间节点的所有子节点都没了，那给该中间节点赋予leaf_value,并向上追溯
    if now_node.children==[]:
        last_column_values = [sublist[-1] for sublist in now_node.node_D]   
        counter = Counter(last_column_values)   
        leaf_value,_ = counter.most_common(1)[0] 
        now_node.leaf_value=leaf_value
        self.bhd_tree(now_node)

  # # 剪枝
  # def check_accuracy(self,D,value):
  #   # 计算预测值与真实值的差异
  #   accuracy = sum(1 for d in D if d[-1] == value) / len(D)
  #   return accuracy
  # def is_del(self,node):
  #   now_accuracy=self.check_accuracy(node.node_D,node.leaf_value)
  #   # 查看父节点多数原则
  #   last_column_values = [sublist[-1] for sublist in node.parent.node_D]   
  #   counter = Counter(last_column_values)   
  #   parent_leaf_value,_ = counter.most_common(1)[0] 
  #   # 剪枝后准确率
  #   after_accuracy=self.check_accuracy(node.node_D,parent_leaf_value)
    
  #   if now_accuracy >after_accuracy:
  #       return False
  #   else:
  #       return True
  # def del_node(self,node):
  #   # 删除子节点，并未释放空间，remove只会删除第一个元素，但这并不会造成问题，因为子节点必然互不相同
  #   return node.parent.children.remove(node)
  # def bhd_tree(self,now_node):

  #   # 如果已经是根节点并且没有子节点，退出
  #   if now_node == self.root and now_node.children == []:
      

  #     return True
  #   # 如果是叶子节点，则直接考虑是否剪枝
  #   if now_node.leaf_value != None:
        
  #       if self.is_del(now_node):
          
  #         self.del_node(now_node)
        
  #       return True
  #   # 下面是针对中间节点的递归
    
  #   for child in now_node.children:
  #       self.bhd_tree(child)
  #   # 如果中间节点的所有子节点都没了，那向上追溯
  #   if now_node.children==[]:
  #       self.bhd_tree(now_node)

class predicts:
    def __init__(self,tree,X,y):
        self.tree = tree
        self.X = X
        self.y = y
        self.predict_result=[]
        # self.predict(self.X,tree.root)
    # classify=classifyer(tree=tree_gen,X=[['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑'],['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘']],y=['是','否'])
    # ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑']
    # 1
    # p匹配上了 蜷缩
    # ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘']
    # 1
    # p匹配上了 稍蜷
    # ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '软粘']
    # 0
    # p匹配上了 乌黑
    # ['是', '否']
     
    def predict(self,now_x,now_node):
       
        for x in now_x:
            # 如果是叶子节点，返回叶节点的取值。其实这个判断仅在根节点是叶子节点的特殊情况才会用到。如果考虑删除下面那句，又必然会多套一层函数，所以差别不是很大
            if now_node.leaf_value!=None:
                self.predict_result.append(int(now_node.leaf_value))
           
            feature_index=now_node.feature_index #获取本节点特征选取的index
            
            for child in now_node.children:
                # 仅进入对应取值的分支
                if math.isclose(x[feature_index],float(child.feature_val)):
                    
                    # 如果孩子是叶节点，返回叶节点的取值
                    if child.leaf_value != None:
                        self.predict_result.append(int(child.leaf_value))
                    else:
                        
                        self.predict([x],child)
                    break
                  # 1111111111
                else:
                  if child==now_node.children[-1]:
                    
                    # 提取每个子列表的最后一个元素  
                    last_column_values = [sublist[-1] for sublist in now_node.node_D]  
                      
                    # 使用 Counter 统计频次  
                    counter = Counter(last_column_values)  
                      
                    # 找到频次最高的元素及其频次  
                    most_common_value, most_common_count = counter.most_common(1)[0] 
                    
                    self.predict_result.append(int(most_common_value))
        
        return self.predict_result



        
