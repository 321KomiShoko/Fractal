# 道路转路划
import os
import geopandas as gpd
import momepy
from shapely.geometry import LineString, MultiLineString


def merge_duplicate_lines(road_gdf):
    """删除重复的线段"""
    # 去除几何完全重复的行
    road_gdf = road_gdf[~road_gdf.geometry.duplicated()]
    # 去除几何值完全相同的行，保持第一个出现的线段
    road_gdf = road_gdf.drop_duplicates(subset='geometry', keep='first')
    return road_gdf


def create_stroke_by_every_best_fit(shp_path, output_path=None, angle_threshold=60, flow_mode=False):
    """处理路网文件并使用COINS算法生成路划"""
    # 检查输入文件是否存在
    if not os.path.exists(shp_path):
        print(f"输入文件 {shp_path} 不存在，跳过处理。")
        return None

    # 检查输出文件是否已存在
    if output_path and os.path.exists(output_path):
        print(f"输出文件 {output_path} 已存在，跳过处理。")
        return None

    print(f"开始处理文件: {shp_path}")
    try:
        road_gdf = gpd.read_file(shp_path)
        # 检查数据是否包含几何列
        if 'geometry' not in road_gdf.columns:
            print(f"输入数据 {shp_path} 不包含几何列，跳过处理。")
            return None
        print(f"处理前线条数量: {len(road_gdf)}")
    except Exception as e:
        print(f"读取文件 {shp_path} 时出错：{e}")
        return None

    # 检查是否为线要素
    if not all(road_gdf.geometry.apply(
            lambda g: isinstance(g, (LineString, MultiLineString))
    )):
        print(f"文件 {shp_path} 包含非线要素，跳过处理")
        return None

    # 删除重复的线段
    road_gdf = merge_duplicate_lines(road_gdf)
    print(f"去重后线条数量: {len(road_gdf)}")

    # 使用COINS算法处理
    try:
        print("正在应用COINS算法...")
        coins = momepy.COINS(road_gdf, angle_threshold=angle_threshold, flow_mode=flow_mode)
        stroke_gdf = coins.stroke_gdf()

        # 检查结果是否为空
        if stroke_gdf.empty:
            print(f"COINS算法未生成有效路划，跳过 {shp_path}")
            return None

        print(f"处理后线条数量: {len(stroke_gdf)}")
    except Exception as e:
        print(f"COINS 算法处理时出错：{e}")
        stroke_gdf = road_gdf

    # 保存结果
    if output_path is not None:
        try:
            # 创建输出目录（如果不存在）
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            stroke_gdf.to_file(output_path, driver='ESRI Shapefile')
            print(f"处理后数据已保存为: {output_path}")
        except Exception as e:
            print(f"保存文件 {output_path} 时出错：{e}")

    return stroke_gdf


# 路径设置 - 按照你的需求配置
centerline_folder = r"E:\城市群道路网分形\data\road_centerline"  # centerline文件所在目录
stroke_folder = r"E:\城市群道路网分形\data\\road_stroke"  # stroke结果保存目录

# 确保输出目录存在
os.makedirs(stroke_folder, exist_ok=True)

# 遍历centerline文件夹中的所有shp文件
for file in os.listdir(centerline_folder):
    if file.endswith(".shp"):
        # 解析文件名（适配"年份_城市_centerline.shp"格式）
        file_base = file.replace('.shp', '')
        parts = file_base.split('_')

        # 验证文件名格式
        if len(parts) >= 3 and parts[0].isdigit() and parts[-1] == 'centerline':
            year = parts[0]
            # 提取城市名（支持城市名包含下划线的情况）
            city_name = '_'.join(parts[1:-1])

            # 构建输入输出路径
            input_shp = os.path.join(centerline_folder, file)
            output_shp = os.path.join(stroke_folder, f"{year}_{city_name}_stroke.shp")

            # 处理文件
            create_stroke_by_every_best_fit(
                shp_path=input_shp,
                output_path=output_shp,
                angle_threshold=60,
                flow_mode=True
            )
        else:
            print(f"文件名格式不符 {file}，跳过处理")

print("所有centerline文件处理完成。")
