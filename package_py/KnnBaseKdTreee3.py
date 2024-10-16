import math  
import heapq
from collections import Counter  
from collections import deque  
  
import math
# kd-tree每个结点中主要包含的数据结构如下
class KdNode(object):
    def __init__(self, dom_elt, split,parent, left, right, val):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.parent = parent  # 父节点
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree
        self.val=val  # 考虑到分类的需要，记录训练集的分类类型

class KdTree(object):
    def __init__(self, data, y):
        if not data:
            return None
        if len(data)!=len(y):
            return print("check X Y")
        k = len(data[0])  # 数据维度

        def CreateNode(parent_node,split, data_set, y_set):  # 按第split维划分数据集exset创建KdNode
            if not data_set:  # 数据集为空
                return None
            # key参数的值为一个函数，此函数只有一个参数且返回一个值用来进行比较
            # operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
            #data_set.sort(key=itemgetter(split)) # 按要进行分割的那一维数据排序
            data_set.sort(key=lambda x: x[split])
            split_pos = len(data_set) // 2  # //为Python中的整数除法
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates
            val=y_set[split_pos]

            # 递归的创建kd树
            now_node = KdNode(
                median,
                split,
                parent_node,
                None,
                None,
                val
            )  # 创建当前节点
            now_node.left = CreateNode(now_node, split_next, data_set[:split_pos],y_set[:split_pos])  # 创建左子树
            now_node.right = CreateNode(now_node, split_next, data_set[split_pos + 1:],y_set[split_pos + 1:])  # 创建右子树
            return now_node
            
        self.root = CreateNode(None,0, data, y)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt,root.split,root.val)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)
 
class Searcher:
    def __init__(self,new_points,kd_tree):
        self.new_points = new_points
        self.kd_tree = kd_tree
        self.nearest_point = []
        self.nearest_distance = []
    
    # 距离计算
    def caculate_Euclidean_distance(self,point1,point2):
        if point1 is None or point2 is None:
            return None
        if len(point1)!= len(point2):
            return None
        sum = 0
        for i in range(len(point1)):
            sum += (point1[i]-point2[i])**2
        return math.sqrt(sum)
    def caculate_Manhattan_distance(self,point1,point2):
        if point1 is None or point2 is None:
            return None
        if len(point1)!= len(point2):
            return None
        sum = 0
        for i in range(len(point1)):
            sum += math.fabs(point1[i]-point2[i])
        return sum
    
    def find_pre_nearest_point(self,root,new_point):
            
            if root is None:
                return None
            if root.left is None and root.right is None:
                return root,self.caculate_Euclidean_distance(root.dom_elt,new_point)
            split=root.split
            if new_point[split] < root.dom_elt[split] and root.left is not None:
                return self.find_pre_nearest_point(root.left,new_point)
            if new_point[split] >= root.dom_elt[split] and root.right is not None:
                return self.find_pre_nearest_point(root.right,new_point)
            else:
                 return root,self.caculate_Euclidean_distance(root.dom_elt,new_point)
    # 返回距离输入点最近的区域节点，和他们之间的距离
    def find_pre_nearest_points(self,root):
        roots=[]
        distances=[]
        new_points=self.new_points
        for new_point in new_points:
            # 给每个当前节点查找最近的初始叶子节点，并记录   
            root_temp,distance_temp=self.find_pre_nearest_point(root,new_point)
            roots.append(root_temp)
            distances.append(distance_temp)
            continue
            
            # if root is None:
            #     continue
            # if root.left is None and root.right is None:
            #     # 给每个当前节点查找最近的初始叶子节点，并记录
            #     roots.append(root)
            #     distances.append(self.caculate_Euclidean_distance(root.dom_elt,new_point))
            #     continue
            #     # return root,self.caculate_Euclidean_distance(root.dom_elt,new_point)
            # split=root.split
            # if new_point[split] < root.dom_elt[split] and root.left is not None:
            #     return self.find_pre_nearest_point(root.left)
            # if new_point[split] >= root.dom_elt[split] and root.right is not None:
            #     return self.find_pre_nearest_point(root.right)
            # else:
            #      # 给每个当前节点查找最近的初始叶子节点，并记录
            #     roots.append(root)
            #     distances.append(self.caculate_Euclidean_distance(root.dom_elt,new_point))
            #     continue
            #     # return root,self.caculate_Euclidean_distance(root.dom_elt,new_point)
        return roots,distances


    # 搜索
    def search(self, new_point, now_node, nearest_node, nearest_distance, k):
        """
        在 KD 树中搜索最近邻节点

        参数:
            now_node: 当前节点
            nearest_node: 当前最近邻节点
            nearest_distance: 当前最近邻距离
            k: 要返回的最近邻节点的数量

        返回:
            最近邻节点和最近邻距离的列表
        """
        
        # 当前节点为空或者已经是根节点，返回
        if now_node is None or now_node.parent is None:
            return None
        # print(new_point)
        
        distance = self.caculate_Euclidean_distance(now_node.dom_elt, new_point)
        if distance < nearest_distance:
            # print(1)
            # print(distance, now_node.dom_elt)
            # 为了避免重复输出父节点，此处不添加
            # 添加历史结点
            self.nearest_distance.append(distance)
            self.nearest_point.append(now_node)
            nearest_distance = distance
            nearest_node = now_node
            if now_node.left is not None:
                
                left_nearest_node, left_nearest_distance = self.search(new_point,now_node.left, nearest_node, nearest_distance, k)
                # print(1.1)
                # print(left_nearest_distance, left_nearest_node.dom_elt)

                # 添加历史结点
                self.nearest_distance.append(left_nearest_distance)
                self.nearest_point.append(left_nearest_node)
                if left_nearest_distance < nearest_distance:

                    nearest_distance = left_nearest_distance
                    nearest_node = left_nearest_node
            if now_node.right is not None:
                # print(1.2)
                # print(right_nearest_distance, right_nearest_node.dom_elt)
                right_nearest_node, right_nearest_distance = self.search(new_point, now_node.right, nearest_node, nearest_distance, k)
                # 添加历史结点
                self.nearest_distance.append(right_nearest_distance)
                self.nearest_point.append(right_nearest_node)
                if right_nearest_distance < nearest_distance:

                    nearest_distance = right_nearest_distance
                    nearest_node = right_nearest_node
            # 左右子树为空，返回父节点，回溯
            else:
                # print(1.3)
                return self.search(new_point,now_node.parent, nearest_node, nearest_distance, k)
        
        # print(2)
        # print(distance, now_node.dom_elt)
        

        self.nearest_distance.append(distance)
        self.nearest_point.append(now_node)

        return nearest_node, nearest_distance
    
    
    def predict(self,k):
         # kd子树只搜索logn个结点，故当节点比较少的时候会出现需求的查找最近k个节点超出搜索过的节点数的情况，此时我们不需考虑k，而尊重kd树的结果，唯一可能造成误差的是根节点的另一节点距离，但一般不会影响大局
        k=min(k,len(self.nearest_point))
        
        # 对数组去重，原因见下方markdown文件
        # 如果涉及到训练集中有重复的点，后期考虑在构建kd树时加入一个计数器，在搜索时对计数器进行判断，若计数器大于1，则加入历史结点，不大于1则不加入，如此下方的去重操作可以不执行
        # 使用dict.fromkeys()保持顺序地去重
        def remove_duplicates(lst):
            return list(dict.fromkeys(lst))
        self.nearest_distance = remove_duplicates(self.nearest_distance)
        self.nearest_point = remove_duplicates(self.nearest_point)
        
        # # 取前k个结点
        re1 = map(self.nearest_distance.index, heapq.nlargest(k, self.nearest_distance)) #求distance最小的k个索引    nsmallest与nlargest相反，求最小

        val_predict=[]
        for i in re1:
            val_predict.append(self.nearest_point[i].val)
        # 使用 Counter 计算每个元素的出现次数  
        counter = Counter(val_predict)  
        
        # 找出出现次数最多的元素及其次数,当数组中的元素没有严格意义上“最多”的重复次数（即多个元素具有相同的最高重复次数）时，collections.Counter 的 most_common() 方法会按照元素在数组中首次出现的顺序返回这些元素中的一个。具体来说，most_common(n) 方法会返回一个列表，其中包含前 n 个最常见元素及其计数，按计数降序排列；如果计数相同，则按元素在输入中的首次出现顺序排列。这是符合我们的需求的  
        most_common_element, count = counter.most_common(1)[0]       
        # print(f"重复次数最多的元素是: {most_common_element}，出现了 {count} 次")
        return most_common_element
def predicts(kd,X_test,k):
    searcher=Searcher(X_test,kd)
    nearest_points,nearest_distances=searcher.find_pre_nearest_points(kd.root)
    # print(list(nearest_points),list(nearest_distances))
    predict_result=[]
    # k之后处理，在search中还未写进，放在了predict之中
    for i in range(len(X_test)):
        searcher.search(X_test[i],nearest_points[i].parent,nearest_points[i],nearest_distances[i],k)
        pred_temp=searcher.predict(k)
        predict_result.append(pred_temp)
        # 恢复状态
        searcher.nearest_distance=[]
        searcher.nearest_point=[]
    return predict_result