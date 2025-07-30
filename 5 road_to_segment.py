import os
import arcpy
from arcpy import env

# 设置工作环境
arcpy.env.overwriteOutput = True

# 输入和输出目录设置
input_directory = r"E:\城市群道路网分形\data\road_centerline"
output_directory = r"E:\城市群道路网分形\data\road_seg"
temp_folder = r"E:\城市群道路网分形\data\temp"  # 临时文件夹

# 创建输出目录和临时文件夹
os.makedirs(output_directory, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# 获取所有道路数据文件（匹配"年份_城市_road.shp"格式）
road_files = [f for f in os.listdir(input_directory)
              if f.endswith('_road.shp') and len(f.split('_')) >= 3 and f.split('_')[0].isdigit()]

# 按年份和城市名称排序
road_files.sort(key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))

# 遍历每个道路数据文件
for road_file in road_files:
    # 解析文件名，提取年份和城市名称
    parts = road_file.split('_')
    year = parts[0]
    city_name = parts[1]
    print(f"\n处理: {year}年 - {city_name}")

    # 输入输出路径
    road_shp = os.path.join(input_directory, road_file)
    # 构建包含城市名称的输出文件名：年份_城市_segment.shp
    output_file = os.path.join(output_directory, f"{year}_{city_name}_segment.shp")

    try:
        # 1. 创建临时图层并将其保存到 temp 文件夹
        selected_road_shp = os.path.join(temp_folder, f"selected_{road_file}")
        arcpy.CopyFeatures_management(road_shp, selected_road_shp)
        print(f"数据复制到临时文件：{selected_road_shp}")

        # 2. 使用 Dissolve 聚合道路（临时文件）
        dissolved_road_shp = os.path.join(temp_folder, f"dissolved_{road_file}")
        arcpy.management.Dissolve(
            selected_road_shp,
            dissolved_road_shp,
            multi_part="SINGLE_PART"  # 确保输出为单部分要素
        )
        print(f"Dissolve 成功：{selected_road_shp} -> {dissolved_road_shp}")

        # 3. 使用 FeatureToLine 将线要素转换为线要素（最终输出文件）
        arcpy.management.FeatureToLine(
            dissolved_road_shp,
            output_file,
            cluster_tolerance="0.01 Meters"  # 根据数据精度调整容差值
        )
        print(f"FeatureToLine 成功：{dissolved_road_shp} -> {output_file}")

        print(f"✓ 已成功处理并保存: {output_file}")

    except Exception as e:
        print(f"✗ 错误: 处理 {year}_{city_name} 失败 - {str(e)}")
        continue

# 可选：清理临时文件
clean_temp = input("\n是否清理临时文件？(y/n): ").strip().lower()
if clean_temp == 'y':
    for file in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            print(f"已删除临时文件: {file_path}")
        except Exception as e:
            print(f"无法删除临时文件 {file_path}: {str(e)}")

print("\n=== 所有文件处理完成 ===")
