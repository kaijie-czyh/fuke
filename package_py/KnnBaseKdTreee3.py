import math  
from collections import deque, Counter  
import heapq  # 导入 heapq 以使用优先队列（最小堆）  
  
# kd-tree每个结点中主要包含的数据结构如下  
class KdNode(object):  
    def __init__(self, dom_elt, split, parent=None, left=None, right=None):  
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)  
        self.split = split  # 整数（进行分割维度的序号）  
        self.parent = parent  # 父节点  
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree  
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree  
  
  
class KdTree(object):  
    def __init__(self, data):  
        if not data:  
            self.root = None  
            return  
        k = len(data[0])  # 数据维度  
  
        def CreateNode(parent_node, split, data_set):  # 按第split维划分数据集exset创建KdNode  
            if not data_set:  # 数据集为空  
                return None  
            data_set.sort(key=lambda x: x[split])  
            split_pos = len(data_set) // 2  # //为Python中的整数除法  
            median = data_set[split_pos]  # 中位数分割点  
            split_next = (split + 1) % k  # cycle coordinates  
  
            # 递归的创建kd树  
            now_node = KdNode(  
                median,  
                split,  
                parent_node,  
                None,  
                None  
            )  # 创建当前节点  
            now_node.left = CreateNode(now_node, split_next, data_set[:split_pos])  # 创建左子树  
            now_node.right = CreateNode(now_node, split_next, data_set[split_pos + 1:])  # 创建右子树  
            return now_node  
  
        self.root = CreateNode(None, 0, data)  # 从第0维分量开始构建kd树,返回根节点  
  
  
class Searcher:  
    def __init__(self, new_point, kd_tree):  
        self.new_point = new_point  
        self.kd_tree = kd_tree  
        self.nearest_points = []  # 存储多个最近邻点  
        self.nearest_distances = []  # 存储对应的距离  
  
    # 距离计算函数  
    def caculate_Euclidean_distance(self, point1, point2):  
        if point1 is None or point2 is None:  
            return float('inf')  # 返回无穷大表示无效距离  
        if len(point1) != len(point2):  
            raise ValueError("Points must have the same dimensions")  
        return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))  
  
    def caculate_Manhattan_distance(self, point1, point2):  
        if point1 is None or point2 is None:  
            return float('inf')  # 返回无穷大表示无效距离  
        if len(point1) != len(point2):  
            raise ValueError("Points must have the same dimensions")  
        return sum(abs(p1 - p2) for p1, p2 in zip(point1, point2))  
  
    def find_nearest_points(self, root, k=1):  
        """  
        在 KD 树中搜索最近的 k 个点  
  
        参数:  
            root: KD 树的根节点  
            k: 要返回的最近邻的数量  
  
        返回:  
            最近邻节点列表和对应的距离列表  
        """  
        if root is None:  
            return [], []  
  
        pq = []  # 优先队列（最小堆）用于存储最近的 k 个点  
  
        # 初始化搜索队列  
        search_queue = deque([(root, self.caculate_Euclidean_distance(root.dom_elt, self.new_point))])  
        visited = set()  
        visited.add(root)  
  
        while search_queue:  
            current_node, current_distance = search_queue.popleft()  
  
            # 如果堆的大小小于 k，或者当前点比堆中最远的点更近，则更新堆  
            if len(pq) < k or current_distance < pq[0][0]:  
                heapq.heappushpop(pq, (current_distance, current_node))  
  
            # 在左右子树中搜索  
            children = [current_node.left, current_node.right]  
            for child in children:  
                if child is not None and child not in visited:  
                    visited.add(child)  
                    # 判断是否应该进入左/右子树  
                    split = current_node.split  
                    if (child == current_node.left and self.new_point[split] < current_node.dom_elt[split]) or (child == current_node.right and self.new_point[split] >= current_node.dom_elt[split]):  
                        search_queue.append((child, self.caculate_Euclidean_distance(child.dom_elt, self.new_point)))  
  
        # 从优先队列中提取最近的 k 个点  
        self.nearest_distances = [dist for dist, node in pq]  
        self.nearest_points = [node.dom_elt for dist, node in pq]  
        return self.nearest_points, self.nearest_distances  
  
    def classify(self, k=1):  
        """  
        根据最近的 k 个邻居对查询点进行分类  
  
        参数:  
            k: 要考虑的最近邻的数量  
  
        返回:  
            查询点的预测类别  
        """  
        nearest_points, _ = self.find_nearest_points(self.kd_tree.root, k)  
  
        # 假设每个点都有一个类别标签（这里我们简单地假设标签是点的最后一个维度，仅用于示例）  
        # 在实际应用中，您需要一个单独的数据结构来存储点的类别标签  
        labels = [point[-1] if isinstance(point, list) else None for point in nearest_points]  # 注意：这里假设标签是点的最后一个元素  
  
        # 找到出现次数最多的标签  
        most_common_label = Counter(labels).most_common(1)[0][0]  
  
        return most_common_label  
  
# 示例用法（这部分通常不会放在类定义文件中，而是放在主脚本中）  
# if __name__ == "__main__":  
#     # 示例数据集  
#     your_dataset = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]  
#     your_query_point = [9, 2]  
#     kd_tree = KdTree(your_dataset)  
#     searcher = Searcher(your_query_point, kd_tree)  
#     predicted_label = searcher.classify(k=3)  
#     print(f"Predicted label: {predicted_label}")