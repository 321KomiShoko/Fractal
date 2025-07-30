import arcpy
import os
import pandas as pd

# 清除 in_memory 中的所有数据
arcpy.management.Delete("in_memory")
# 设置工作空间（全国道路网数据的根文件夹路径）
road_network_base_folder = r"D:\Data\OSM\data\road_china"
# 设置包含全国城市面要素的Shapefile路径
city_boundary_shp = r"E:\Pycharm\pythonProject\城市群道路网\data\boundary\中国_市2.shp"
# 设置储存裁剪结果的根输出文件夹路径
base_output_folder = r"E:\Pycharm\pythonProject\城市群道路网\data\road_city"
# 待处理城市的Excel文件路径
cities_excel_path = r"E:\Pycharm\pythonProject\城市群道路网\data\boundary\待处理城市.xlsx"

# 设置要保留的道路类型
valid_fclasses = [
    "residential", "tertiary", "unclassified", "motorway", "secondary", "primary", "trunk",
    "residential_link", "tertiary_link", "motorway_link", "secondary_link", "primary_link", "trunk_link"
]

# 设置 arcpy 的环境设置
arcpy.env.overwriteOutput = True

# 从Excel读取需要处理的城市名称（假设城市名称在"name"列）
try:
    # 读取Excel文件
    cities_df = pd.read_excel(cities_excel_path)
    # 提取"name"列的城市名称，去除空值并去重
    target_cities = [str(city).strip() for city in cities_df['name'].dropna().unique()]
    print(f"从Excel中读取到 {len(target_cities)} 个待处理城市")
except Exception as e:
    print(f"读取待处理城市Excel文件失败：{e}")
    # 如果读取失败则退出程序
    exit(1)

# 提前创建输出文件夹
os.makedirs(base_output_folder, exist_ok=True)

# 遍历每个年份的道路网数据（2014 到 2024）
for year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
    # 构造当前年份道路网数据路径
    road_network_year_folder = os.path.join(road_network_base_folder, f"china_{year}")

    # 检查当前年份的道路网文件夹是否存在
    if not os.path.exists(road_network_year_folder):
        print(f"文件夹不存在：{road_network_year_folder}，跳过该年份")
        continue

    # 获取当前年份道路网数据中的所有 .shp 文件
    road_files = [f for f in os.listdir(road_network_year_folder) if f.endswith(".shp")]
    if not road_files:
        print(f"{road_network_year_folder} 中未找到shp文件，跳过该年份")
        continue

    # 对每个城市提前检查是否有需要处理的文件
    for city_name in target_cities:
        # 提前构造输出路径并检查
        output_shp = os.path.join(base_output_folder, f"{year}_{city_name}_road.shp")
        if os.path.exists(output_shp):
            print(f"{output_shp} 已存在，跳过该城市年份组合")
            continue

        # 只处理需要处理的城市-年份组合
        print(f"开始处理：{year}年 - {city_name}")

        # 遍历该年份的所有道路文件
        for file in road_files:
            road_shp = os.path.join(road_network_year_folder, file)

            # 构造SQL查询语句筛选 fclass 字段
            fclass_query = " OR ".join([f"fclass = '{fclass}'" for fclass in valid_fclasses])

            # 创建临时图层并筛选
            try:
                arcpy.MakeFeatureLayer_management(road_shp, "road_layer", fclass_query)
            except Exception as e:
                print(f"创建道路图层失败 {road_shp}：{e}")
                continue

            # 查询当前城市的边界
            try:
                city_boundary = arcpy.Select_analysis(
                    city_boundary_shp,
                    "in_memory\city_boundary",
                    f"name = '{city_name}'"
                )
            except Exception as e:
                print(f"获取 {city_name} 边界失败：{e}")
                arcpy.management.Delete("road_layer")
                continue

            # 执行裁剪操作
            try:
                arcpy.analysis.Clip("road_layer", city_boundary, output_shp)
                print(f"裁剪并筛选成功：{road_shp} -> {output_shp}")
            except Exception as e:
                print(f"处理失败：{road_shp} -> {city_name}，错误信息：{e}")

            # 删除临时图层
            arcpy.management.Delete("road_layer")

print("所有指定城市的文件处理完成。")
