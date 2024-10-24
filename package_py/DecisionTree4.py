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
  def choose_A(self,D,A):
        max_gain=0
        for i in range(len(A)):
          gain,attribute_name=self.calculate_Gain(D,i)
          # 考虑等于0，是因为在外层遍历中并未考虑D在指定特征列的取值已经全部相同的情况
          if gain>=max_gain:
            max_gain=gain
            max_attribute_index=i
            max_attribute_name=attribute_name
          
        
        return max_attribute_index,max_attribute_name,max_gain
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
      feature_index, feature_name ,feature_gain= self.choose_A(D, A)  
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
    # 辅助函数：打印决策树（用于调试）  
def print_tree( node, level=0):  
    print(" " * level * 4 + f"Node_D_val: {node.feature_val if node.node_D else []}, Feature Index: {node.feature_index}, node: {node.children}, Leaf Value: {node.leaf_value}")  
    if node.children==[]:
      return 0
    for child in node.children:  
        print_tree(child, level + 1)  
    return 0

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



        
