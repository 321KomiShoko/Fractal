import os
import geopandas as gpd
import networkx as nx
from rtree import index
import time


def to_dual_graph(stroke_gdf, output_path=None):
    """
    将路划数据转换为对偶图

    参数:
        stroke_gdf: GeoDataFrame，路划数据
        output_path: str，对偶图保存路径（.graphml格式）

    返回:
        networkx.Graph，生成的对偶图
    """
    # 初始化对偶图
    dual_graph = nx.Graph()

    # 验证输入数据是否为空
    if stroke_gdf.empty:
        print("输入路划数据为空，无法生成对偶图")
        return dual_graph

    # 初始化R-tree索引用于空间查询
    print("正在构建空间索引...")
    rtree_idx = index.Index()
    valid_indices = []  # 存储有效线要素的索引

    for i, geom in enumerate(stroke_gdf.geometry):
        # 只处理线要素
        if geom.geom_type in ['LineString', 'MultiLineString'] and not geom.is_empty:
            rtree_idx.insert(i, geom.bounds)
            valid_indices.append(i)
        else:
            print(f"跳过非线要素或空要素（索引：{i}）")

    if not valid_indices:
        print("没有有效线要素，无法生成对偶图")
        return dual_graph

    print(f"共发现 {len(valid_indices)} 个有效线要素，开始构建对偶图...")

    # 遍历所有有效要素，检测相交关系
    for i in valid_indices:
        geom1 = stroke_gdf.geometry.iloc[i]
        dual_graph.add_node(i)  # 添加节点

        # 查找可能相交的要素
        candidates = list(rtree_idx.intersection(geom1.bounds))
        # 过滤并只保留索引大于i的候选要素（避免重复计算）
        candidates = [j for j in candidates if j in valid_indices and j > i]

        for j in candidates:
            geom2 = stroke_gdf.geometry.iloc[j]
            # 精确检查几何相交
            if geom1.intersects(geom2):
                dual_graph.add_edge(i, j)  # 添加边

    print(f"对偶图构建完成：{dual_graph.number_of_nodes()}个节点，{dual_graph.number_of_edges()}条边")

    # 处理非连通图并保存
    if output_path:
        # 若图不连通，取最大连通子图
        if not nx.is_connected(dual_graph):
            print("检测到非连通图，提取最大连通子图...")
            largest_cc = max(nx.connected_components(dual_graph), key=len)
            dual_graph = dual_graph.subgraph(largest_cc).copy()
            print(f"最大连通子图：{dual_graph.number_of_nodes()}个节点，{dual_graph.number_of_edges()}条边")

        # 保存对偶图
        try:
            nx.write_graphml(dual_graph, output_path)
            print(f"对偶图已成功保存至：{output_path}")
        except Exception as e:
            print(f"保存对偶图失败：{str(e)}")

    return dual_graph


def batch_process_stroke_files(stroke_dir, graph_dir):
    """
    批量处理路划文件，生成对偶图

    参数:
        stroke_dir: str，路划文件所在目录（road_stroke）
        graph_dir: str，对偶图输出目录（data_graph）
    """
    # 记录总处理时间
    start_time = time.time()

    # 创建输出目录
    os.makedirs(graph_dir, exist_ok=True)
    print(f"输出目录：{graph_dir}")

    # 统计变量
    total_files = 0
    success_files = 0
    skip_files = 0
    fail_files = 0

    # 遍历路划目录中的所有shp文件
    for filename in os.listdir(stroke_dir):
        # 只处理符合格式的stroke文件（年份_城市_stroke.shp）
        if filename.endswith("_stroke.shp"):
            total_files += 1
            file_base = filename.replace(".shp", "")  # 去除扩展名

            try:
                # 解析文件名：年份_城市_stroke → 提取年份和城市
                # 处理城市名可能包含下划线的情况（如"2024_台北市_中正区_stroke"）
                year_city_part = file_base.rsplit("_", 1)[0]  # 得到"年份_城市"
                year, city_name = year_city_part.split("_", 1)  # 分割年份和城市名

                # 构建输入输出路径
                input_path = os.path.join(stroke_dir, filename)
                output_filename = f"{year}_{city_name}_graph.graphml"
                output_path = os.path.join(graph_dir, output_filename)

                # 检查输出文件是否已存在
                if os.path.exists(output_path):
                    print(f"[{total_files}] 已存在，跳过：{filename}")
                    skip_files += 1
                    continue

                # 读取路划数据并转换为对偶图
                print(f"\n[{total_files}] 开始处理：{filename}")
                stroke_gdf = gpd.read_file(input_path)
                to_dual_graph(stroke_gdf, output_path)
                success_files += 1

            except Exception as e:
                print(f"[{total_files}] 处理失败 {filename}：{str(e)}")
                fail_files += 1

    # 计算总耗时
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    # 输出统计结果
    print("\n" + "=" * 50)
    print("批量处理完成")
    print(f"总文件数：{total_files}")
    print(f"成功：{success_files} | 跳过：{skip_files} | 失败：{fail_files}")
    print(f"总耗时：{minutes}分{seconds}秒")
    print("=" * 50)


if __name__ == "__main__":
    # 路径配置
    STROKE_DIRECTORY = r"E:\城市群道路网分形\data\road_stroke"  # 路划文件所在目录
    GRAPH_DIRECTORY = r"E:\城市群道路网分形\data\road_graph"  # 对偶图输出目录

    # 执行批量处理
    batch_process_stroke_files(STROKE_DIRECTORY, GRAPH_DIRECTORY)
