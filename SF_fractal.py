from pathlib import Path
import random
import pathlib
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from networkx import *
from rtree import index
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import os
import re


def to_dual_graph(stroke_gdf, output_path=None):
    stroke_dual_graph = nx.Graph()
    print("正在初始化R-tree索引...")

    # 初始化 R-tree 索引
    rtree_idx = index.Index()
    for i, geom in enumerate(stroke_gdf.geometry):
        if geom.geom_type in ['LineString', 'MultiLineString']:
            rtree_idx.insert(i, geom.bounds)

    print("R-tree索引初始化完成，开始构建图...")

    # 遍历每个几何对象，检查相交情况
    for i, geom1 in enumerate(stroke_gdf.geometry):
        if geom1.is_empty or geom1.geom_type not in ['LineString', 'MultiLineString']:
            continue

        stroke_dual_graph.add_node(i)
        candidates = list(rtree_idx.intersection(geom1.bounds))

        for j in candidates:
            if i >= j:
                continue
            geom2 = stroke_gdf.geometry.iloc[j]
            if geom2.is_empty or geom2.geom_type not in ['LineString', 'MultiLineString']:
                continue
            if geom1.intersects(geom2):
                stroke_dual_graph.add_edge(i, j)

    if output_path:
        # 检查是否连通（无向图专用方法）
        if not nx.is_connected(stroke_dual_graph):
            # 获取最大连通子图（无向图专用）
            largest_cc = max(nx.connected_components(stroke_dual_graph), key=len)
            stroke_dual_graph = stroke_dual_graph.subgraph(largest_cc).copy()
        else:
            print("数据出错————————————————————————")

        nx.write_graphml(stroke_dual_graph, output_path)

    return stroke_dual_graph


def process_shp_to_graph(shp_path, output_folder):
    """处理单个SHP文件的独立函数"""
    try:
        output_path = pathlib.Path(output_folder) / f"{shp_path.stem}_dual.graphml"
        if output_path.exists():
            print(f"跳过已存在的文件：{output_path.name}")
            return True

        print(f"正在处理：{shp_path.name}")
        gdf = gpd.read_file(shp_path)
        _ = to_dual_graph(gdf, str(output_path))
        return True
    except Exception as e:
        print(f"处理失败 {shp_path.name}: {str(e)}")
        return False


def memb_box(_graph, box_step, cycle=0):
    """
    MEMB introduced by Song, Havlin, Makse in Nature Physics 2, 275.
    """
    # 预处理网络
    if _graph.is_directed():  # 如果图是有向的，将其转为无向图
        _graph = _graph.to_undirected()

    adj = _graph.adj  # 获取图 G 的邻接表
    number_of_nodes = _graph.number_of_nodes()  # 获取节点数量
    covered_nodes = set()  # 已覆盖节点集合
    center_nodes = set()  # 中心节点集合
    non_center_nodes = list(_graph.nodes())  # 所有非中心节点的列表
    center_node_found = 0  # 中心节点找到标志
    boxes = {}  # 存储框与节点的映射 {box_id: [框内节点]}
    central_distance_of_node = {}  # 节点的中心距离映射 {node: central_distance}
    node_box_id = {}  # 节点的框 ID 映射 {node: box_id}
    nodes_sorted_by_central_distance = {}  # 根据中心距离排序的节点字典 {central_distance: [nodes]}
    excluded_mass_of_non_centers_rb = {}  # 非中心节点的排除质量映射，针对 rb
    excluded_mass_of_non_centers_rb2 = {}  # 非中心节点的排除质量映射，针对 rb+1
    rb2 = box_step + 1  # rb+1 的值

    # 计算每个节点的排除质量
    for node in non_center_nodes:
        level = 0  # 当前层级
        nextlevel = {node: 1}  # 下一层需要检查的节点
        paths_rb = None  # 用于存储 rb 路径
        paths_rb2 = {node: [node]}  # 路径字典（从源节点到每个节点的路径）

        # BFS 遍历计算路径
        while nextlevel:
            paths_rb = deepcopy(paths_rb2)  # 深拷贝当前路径
            thislevel = nextlevel  # 当前层的节点
            nextlevel = {}  # 重置下一层节点
            for v in thislevel:
                for w in _graph.neighbors(v):  # 遍历当前节点的邻居
                    if w not in paths_rb2:  # 如果邻居未在路径字典中
                        paths_rb2[w] = paths_rb2[v] + [w]  # 更新路径
                        nextlevel[w] = 1  # 添加到下一层
            level = level + 1  # 增加层级
            if rb2 <= level:
                break  # 如果达到 rb2 层级，停止

        # 计算排除质量
        excluded_mass_of_node = len(paths_rb2)  # 当前节点的排除质量
        try:
            excluded_mass_of_non_centers_rb2[excluded_mass_of_node].append(node)  # 更新 rb+1 的排除质量映射
        except KeyError:
            excluded_mass_of_non_centers_rb2[excluded_mass_of_node] = [node]  # 创建新条目

        excluded_mass_of_node = len(paths_rb)  # 计算 rb 的排除质量
        try:
            excluded_mass_of_non_centers_rb[excluded_mass_of_node].append(node)  # 更新 rb 的排除质量映射
        except KeyError:
            excluded_mass_of_non_centers_rb[excluded_mass_of_node] = [node]  # 创建新条目

    maximum_excluded_mass = 0  # 最大排除质量
    nodes_with_maximum_excluded_mass = []  # 具有最大排除质量的节点列表
    new_covered_nodes = {}  # 新覆盖节点字典
    center_node_and_mass = []  # 中心节点及其质量
    cycle_index = 0  # 循环计数器

    # 主循环，直到所有节点被覆盖
    while len(covered_nodes) < number_of_nodes:
        cycle_index += 1  # 增加循环计数
        if cycle_index == cycle:
            rb2 = box_step + 1  # 如果达到循环限制，增加 rb2
            cycle_index = 0  # 重置循环计数
        else:
            rb2 = box_step  # 否则使用 rb

        while 1:  # 内部循环
            if rb2 == box_step + 1:  # 如果使用 rb+1
                while 1:
                    maximum_key = max(excluded_mass_of_non_centers_rb2.keys())  # 找到最大排除质量
                    node = random.choice(excluded_mass_of_non_centers_rb2[maximum_key])  # 随机选择一个节点
                    if node in center_nodes:  # 如果节点是中心节点
                        excluded_mass_of_non_centers_rb2[maximum_key].remove(node)  # 从映射中移除
                        if not excluded_mass_of_non_centers_rb2[maximum_key]:
                            del excluded_mass_of_non_centers_rb2[maximum_key]  # 如果为空，删除键
                    else:
                        break  # 找到非中心节点，退出循环

                nodes_visited = {}  # 访问节点的字典
                bfs = single_source_shortest_path(_graph, node, cutoff=rb2)  # BFS 查找节点的路径
                for i in bfs:
                    nodes_visited[i] = len(bfs[i]) - 1  # 记录每个节点的访问长度
                excluded_mass_of_node = len(set(nodes_visited.keys()).difference(covered_nodes))  # 计算新的排除质量

                if excluded_mass_of_node == maximum_key:  # 如果新的排除质量等于最大排除质量
                    center_node_and_mass = (node, maximum_key)  # 记录中心节点和质量
                    excluded_mass_of_non_centers_rb2[maximum_key].remove(node)  # 移除节点
                    if not excluded_mass_of_non_centers_rb2[maximum_key]:
                        del excluded_mass_of_non_centers_rb2[maximum_key]  # 删除空条目
                    new_covered_nodes = nodes_visited  # 更新新覆盖节点
                    break  # 退出循环
                else:
                    excluded_mass_of_non_centers_rb2[maximum_key].remove(node)  # 移除节点
                    if not excluded_mass_of_non_centers_rb2[maximum_key]:
                        del excluded_mass_of_non_centers_rb2[maximum_key]  # 删除空条目
                    try:
                        excluded_mass_of_non_centers_rb2[excluded_mass_of_node].append(node)  # 更新排除质量映射
                    except KeyError:
                        excluded_mass_of_non_centers_rb2[excluded_mass_of_node] = [node]  # 创建新条目

            else:  # 如果使用 rb
                while 1:
                    maximum_key = max(excluded_mass_of_non_centers_rb.keys())  # 找到最大排除质量
                    node = random.choice(excluded_mass_of_non_centers_rb[maximum_key])  # 随机选择一个节点
                    if node in center_nodes:  # 如果节点是中心节点
                        excluded_mass_of_non_centers_rb[maximum_key].remove(node)  # 从映射中移除
                        if not excluded_mass_of_non_centers_rb[maximum_key]:
                            del excluded_mass_of_non_centers_rb[maximum_key]  # 删除空条目
                    else:
                        break  # 找到非中心节点，退出循环

                nodes_visited = {}  # 访问节点的字典
                bfs = single_source_shortest_path(_graph, node, cutoff=box_step)  # BFS 查找节点的路径
                for i in bfs:
                    nodes_visited[i] = len(bfs[i]) - 1  # 记录每个节点的访问长度
                excluded_mass_of_node = len(set(nodes_visited.keys()).difference(covered_nodes))  # 计算新的排除质量

                if excluded_mass_of_node == maximum_key:  # 如果新的排除质量等于最大排除质量
                    center_node_and_mass = (node, maximum_key)  # 记录中心节点和质量
                    excluded_mass_of_non_centers_rb[maximum_key].remove(node)  # 移除节点
                    if not excluded_mass_of_non_centers_rb[maximum_key]:
                        del excluded_mass_of_non_centers_rb[maximum_key]  # 删除空条目
                    new_covered_nodes = nodes_visited  # 更新新覆盖节点
                    break  # 退出循环
                else:
                    excluded_mass_of_non_centers_rb[maximum_key].remove(node)  # 移除节点
                    if not excluded_mass_of_non_centers_rb[maximum_key]:
                        del excluded_mass_of_non_centers_rb[maximum_key]

                    # 删除空条目
                    try:
                        excluded_mass_of_non_centers_rb[excluded_mass_of_node].append(node)  # 更新排除质量映射
                    except KeyError:
                        excluded_mass_of_non_centers_rb[excluded_mass_of_node] = [node]  # 创建新条目

        center_node_found = center_node_and_mass[0]  # 找到中心节点
        boxes[center_node_found] = [center_node_found]  # 在框字典中初始化中心节点
        node_box_id[center_node_found] = center_node_found  # 记录节点的框 ID
        non_center_nodes.remove(center_node_found)  # 从非中心节点列表中移除中心节点
        center_nodes.add(center_node_found)  # 将中心节点添加到中心节点集合

        covered_nodes = covered_nodes.union(set(new_covered_nodes.keys()))  # 更新已覆盖节点集合
        for i in new_covered_nodes:
            try:
                if central_distance_of_node[i] > new_covered_nodes[i]:  # 如果当前节点的中心距离大于新覆盖节点的距离
                    nodes_sorted_by_central_distance[central_distance_of_node[i]].remove(i)  # 从旧位置移除节点
                    if not nodes_sorted_by_central_distance[central_distance_of_node[i]]:
                        del nodes_sorted_by_central_distance[central_distance_of_node[i]]  # 删除空条目
                    try:
                        nodes_sorted_by_central_distance[new_covered_nodes[i]].append(i)  # 添加到新位置
                    except KeyError:
                        nodes_sorted_by_central_distance[new_covered_nodes[i]] = [i]  # 创建新条目
                    central_distance_of_node[i] = new_covered_nodes[i]  # 更新中心距离
            except KeyError:
                central_distance_of_node[i] = new_covered_nodes[i]  # 记录中心距离
                try:
                    nodes_sorted_by_central_distance[new_covered_nodes[i]].append(i)  # 添加到新位置
                except:
                    nodes_sorted_by_central_distance[new_covered_nodes[i]] = [i]  # 创建新条目

    max_distance = max(nodes_sorted_by_central_distance.keys())  # 找到最大中心距离
    for i in range(1, max_distance + 1):  # 遍历所有距离
        for j in nodes_sorted_by_central_distance[i]:  # 遍历当前距离的所有节点
            targets = list(set(adj[j].keys()).intersection(set(nodes_sorted_by_central_distance[i - 1])))  # 找到目标节点
            if targets:  # 检查目标节点列表是否非空
                node_box_id[j] = node_box_id[random.choice(targets)]  # 随机选择目标节点的框 ID
                boxes[node_box_id[j]].append(j)  # 将当前节点添加到其框中
            else:
                node_box_id[j] = j  # 默认将节点保留在自身的框中

    boxes_subgraphs = {}  # 存储每个框的子图字典
    for i in boxes:
        boxes_subgraphs[i] = subgraph(_graph, boxes[i])  # 为每个框生成子图

    return boxes_subgraphs


def calculate_box_s_dimension(_graph):
    """
    计算图的盒覆盖维度并返回结果。

    参数:
    _graph : networkx.Graph
        需要进行盒覆盖计算的图。

    返回:
    results_df : pandas.DataFrame
        包含盒覆盖结构维度计算结果的 DataFrame。
    """
    local_results = []
    box_num_count = {}  # 用于统计每个box_num出现的次数

    # 遍历0到50的盒子尺度
    for box_step in [0, 1, 2, 3, 4]:
        box_size = box_step * 2 + 1  # 计算盒子的大小
        if box_size == 1:
            box_num = _graph.number_of_nodes()  # 盒子大小为1时，节点数即为盒子数
        else:
            box_num = len(memb_box(_graph, box_step))  # MEMB方法计算盒子的数量

        print(f"已处理盒子尺度为 {box_step}，盒子大小为 {box_size}，盒子数：{box_num}")

        # 更新 box_num 的出现次数
        if box_num in box_num_count:
            box_num_count[box_num] += 1
        else:
            box_num_count[box_num] = 1

        # 如果box_num出现了4次，跳出循环
        if box_num_count[box_num] == 3:
            print(f"盒子数 {box_num} 已经出现3次，跳出循环")
            break

        log_box_size = np.log2(box_size)
        log_box_num = np.log2(box_num) if box_num > 0 else None

        local_results.append({
            'box_step': box_step,
            'box_size': box_size,
            'box_num': box_num,
            'log_box_size': log_box_size,
            'log_box_num': log_box_num
        })

        if log_box_num == 0:
            print(f"log_box_num 为 0，跳出循环")
            break

    # 将计算结果转换为DataFrame
    results_df = pd.DataFrame(local_results)

    return results_df


def calculate_volume_s_dimension(_graph):
    """
    计算图的体积分形维度并返回结果。

    参数:
    graph : networkx.Graph
        需要进行体积计算的图。

    返回:
    results_df : pandas.DataFrame
        包含体积维度计算结果的 DataFrame。
    """
    local_results = []

    # 遍历1到10的体积步长
    for volume_step in range(1, 5):
        __volume = np.array([
            len(nx.single_source_shortest_path_length(_graph, node, cutoff=volume_step))
            for node in _graph.nodes
        ])
        average_volume = np.mean(__volume)  # 计算该步长下的平均体积
        print(f"已处理体积步长为 {volume_step}")

        log_volume_step = np.log2(volume_step)
        log_average_volume = np.log2(average_volume) if average_volume > 0 else None

        local_results.append({
            'volume_step': volume_step,
            'average_volume': average_volume,
            'log_volume_step': log_volume_step,
            'log_average_volume': log_average_volume
        })

        if log_average_volume == 0:
            print(f"log_average_volume 为 0，跳出循环")
            break

    # 将计算结果转换为 DataFrame
    results_df = pd.DataFrame(local_results)

    return results_df


def calculate_degree_volume_s_dimension(_graph):
    """
    计算图的度体积分形维度并返回结果。

    参数:
    graph : networkx.Graph
        需要进行度体积计算的图。

    返回:
    results_df : pandas.DataFrame
        包含度体积维度计算结果的 DataFrame。
    """
    local_results = []

    # 遍历0到10的度体积步长
    for volume_step in range(1, 5):
        # 计算每个节点的度体积
        __degree_volume = np.array([
            np.sum([_graph.degree(n) for n in
                    nx.single_source_shortest_path_length(_graph, node, cutoff=volume_step)])
            for node in _graph.nodes
        ])
        average_degree_volume = np.mean(__degree_volume)  # 计算该步长下的平均度体积

        log_volume_step = np.log2(volume_step)
        log_average_degree_volume = np.log2(average_degree_volume) if average_degree_volume > 0 else None

        local_results.append({
            'volume_step': volume_step,
            'average_degree_volume': average_degree_volume,
            'log_volume_step': log_volume_step,
            'log_average_degree_volume': log_average_degree_volume
        })

        if log_average_degree_volume == 0:
            print(f"log_average_degree_volume 为 0，跳出循环")
            break

    # 将计算结果转换为 DataFrame
    results_df = pd.DataFrame(local_results)

    return results_df


def process_single_graphml(graph_file, output_dir, processors):
    """处理单个GraphML文件的独立函数"""
    try:
        # 检查文件名是否包含目标年份
        file_name = graph_file.name
        target_years = ['2014', '2016', '2018', '2019', '2020', '2022', '2024']
        year_found = False
        for year in target_years:
            if year in file_name:
                year_found = True
                break
        if not year_found:
            print(f"跳过非目标年份文件: {file_name}")
            return True

        print(f"并行处理图文件: {graph_file.name}")
        G = nx.read_graphml(graph_file)

        base_name = graph_file.stem.replace("_stroke_dual", "")
        for suffix, processor in processors.items():
            output_file = output_dir / f"{base_name}_{suffix}.xlsx"
            if output_file.exists():
                print(f"跳过已存在的结果文件：{output_file.name}")
                continue

            df = processor(G)
            df.to_excel(output_file, index=False)

        return True
    except Exception as e:
        print(f"处理失败 {graph_file.name}: {str(e)}")
        return False


def process_graphml_files(input_dir, output_dir, workers=None):
    """并行版本GraphML处理"""
    input_path = pathlib.Path(input_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processors = {
        "volume": calculate_volume_s_dimension,
        "degree_volume": calculate_degree_volume_s_dimension,
        "box": calculate_box_s_dimension
    }

    graph_files = list(input_path.glob("*.graphml"))

    # 自动设置工作进程数
    if workers is None:
        workers = min(os.cpu_count(), len(graph_files))

    print(f"启动并行图处理，使用{workers}个工作进程...")

    # 使用进程池处理
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            partial(process_single_graphml,
                    output_dir=output_path,
                    processors=processors),
            graph_files
        ))

    success_count = sum(results)
    print(f"图处理完成: 成功{success_count}/{len(graph_files)} 失败{len(graph_files) - success_count}")


# 使用方法
if __name__ == "__main__":
    # 配置并行工作进程数（None为自动检测）
    PARALLEL_WORKERS = 8

    # 路径配置
    stroke_folder = r"E:\长三角2014-2024年道路网分形维计算\data\stroke_data"
    graph_folder = r"E:\长三角2014-2024年道路网分形维计算\data\graph_data"
    result_folder = r"E:\长三角2014-2024年道路网分形维计算\result_graph"

    target_years = ['2014', '2016', '2018', '2019', '2020', '2022', '2024']

    # # 路划转对偶图
    # input_dir = pathlib.Path(stroke_folder)
    # output_dir = pathlib.Path(graph_folder)
    # # 创建输出文件夹，如果已存在则不会报错
    # output_dir.mkdir(parents=True, exist_ok=True)
    # # 遍历输入文件夹中的所有 SHP 文件
    # for shp_path in input_dir.glob("*.shp"):
    #     process_shp_to_graph(shp_path, output_dir)

    # 并行处理GraphML文件
    process_graphml_files(graph_folder, result_folder, workers=PARALLEL_WORKERS)
