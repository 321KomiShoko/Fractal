# 提取道路中心线
import arcpy
import os
import time


def main():
    # 记录开始时间
    start_time = time.time()

    # 配置路径 - 根据你的需求设置
    input_directory = r"E:\城市群道路网分形\data\road_city"  # 输入：年份_城市_road.shp所在目录
    output_directory = r"E:\城市群道路网分形\data\road_centerline"  # 输出：转换后的文件目录
    temp_folder = r"E:\城市群道路网分形\data\temp"  # 临时文件目录
    buffer_threshold = 25  # 缓冲区半径（米），可根据道路宽度调整

    # 设置ArcGIS环境
    arcpy.env.overwriteOutput = True
    arcpy.env.workspace = temp_folder  # 临时工作空间

    # 创建必要的目录
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(temp_folder, exist_ok=True)

    # 筛选符合格式的输入文件，并提前过滤已存在输出的文件
    road_files = []
    for f in os.listdir(input_directory):
        if f.endswith("_road.shp") and len(f.split("_")) >= 3 and f.split("_")[0].isdigit():
            # 解析文件名，提前判断输出是否存在
            parts = f.split("_")
            year, city = parts[0], parts[1]
            output_shp = os.path.join(output_directory, f"{year}_{city}_centerline.shp")
            if not arcpy.Exists(output_shp):
                road_files.append(f)  # 只保留需要处理的文件
            else:
                print(f"已存在，提前过滤：{output_shp}")

    # 按年份和城市排序
    road_files.sort(key=lambda x: (int(x.split("_")[0]), x.split("_")[1]))

    # 显示处理信息
    total_files = len(road_files)
    print(f"===== 开始处理双线转单线 =====")
    print(f"输入目录: {input_directory}")
    print(f"输出目录: {output_directory}")
    print(f"需要处理的文件: {total_files} 个（已过滤已存在文件）")
    print(f"缓冲区半径: {buffer_threshold} 米")
    print("================================\n")

    # 记录处理结果
    success_count = 0
    fail_count = 0

    # 批量处理文件
    for idx, road_file in enumerate(road_files, 1):
        # 解析文件名
        file_parts = road_file.split("_")
        year = file_parts[0]
        city_name = file_parts[1]
        print(f"处理 {idx}/{total_files}: {year}年 - {city_name}")

        # 构建文件路径
        input_shp = os.path.join(input_directory, road_file)
        output_shp = os.path.join(output_directory, f"{year}_{city_name}_centerline.shp")

        # 二次检查（防止多进程/手动创建导致的遗漏）
        if arcpy.Exists(output_shp):
            print(f"  → 已存在，跳过 ({output_shp})")
            continue

        # 执行转换处理
        try:
            # 调用处理函数
            result = convert_road(input_shp, temp_folder, buffer_threshold)

            if result and arcpy.Exists(result):
                # 导出最终结果
                arcpy.conversion.ExportFeatures(result, output_shp)
                # 清理临时结果
                arcpy.management.Delete(result)
                print(f"  → 成功生成: {output_shp}")
                success_count += 1
            else:
                print(f"  → 生成失败: 无有效输出")
                fail_count += 1

        except Exception as e:
            print(f"  → 处理错误: {str(e)}")
            fail_count += 1
            continue

    # 计算耗时
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)

    # 输出统计信息
    print("\n===== 处理完成 =====")
    print(f"总需处理文件数: {total_files}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"耗时: {minutes}分{seconds}秒")
    print("====================")


def convert_road(input_shp, temp_dir, threshold):
    """核心转换函数：处理双线道路转中心线并合并单线道路"""
    # 生成唯一临时文件名
    base_name = os.path.splitext(os.path.basename(input_shp))[0]
    double_line = os.path.join(temp_dir, f"{base_name}_double.shp")
    single_line = os.path.join(temp_dir, f"{base_name}_single.shp")
    buffer = os.path.join(temp_dir, f"{base_name}_buffer.shp")
    buffer_dissolve = os.path.join(temp_dir, f"{base_name}_buffer_dissolve.shp")
    centerline = os.path.join(temp_dir, f"{base_name}_centerline.shp")
    merged = os.path.join(temp_dir, f"{base_name}_merged.shp")

    try:
        # 1. 分离双线和单线道路
        # 双线道路：oneway=1或'True'
        arcpy.conversion.ExportFeatures(
            input_shp, double_line,
            "oneway = 1 OR oneway = 'True'"
        )

        # 单线道路：oneway=0或'False'
        arcpy.conversion.ExportFeatures(
            input_shp, single_line,
            "oneway = 0 OR oneway = 'False'"
        )

        # 2. 双线道路转中心线
        # 创建缓冲区
        arcpy.analysis.GraphicBuffer(
            double_line, buffer,
            f"{threshold} Meters",
            "BUTT", "ROUND"
        )

        # 融合缓冲区
        arcpy.management.Dissolve(buffer, buffer_dissolve)

        # 生成中心线
        arcpy.PolygonToCenterline_topographic(
            buffer_dissolve, centerline
        )

        # 3. 合并结果
        # 检查中心线是否有效
        if int(arcpy.GetCount_management(centerline)[0]) == 0:
            # 没有有效中心线，只使用单线道路
            arcpy.management.Merge([single_line], merged)
        else:
            # 合并中心线和单线道路
            arcpy.management.Merge([centerline, single_line], merged)

        return merged

    finally:
        # 清理中间临时文件
        temp_files = [double_line, single_line, buffer, buffer_dissolve, centerline]
        for file in temp_files:
            if arcpy.Exists(file):
                arcpy.management.Delete(file)


if __name__ == "__main__":
    main()