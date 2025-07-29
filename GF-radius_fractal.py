import arcpy
import pandas as pd
import os
import numpy as np
import math

# 设置工作空间
arcpy.env.overwriteOutput = True  # 允许覆盖输出文件


# 定义生成多环缓冲区的函数
def create_multiple_ring_buffer(points_shp, buffer_shp, distances, buffer_unit="Kilometers"):
    arcpy.analysis.MultipleRingBuffer(points_shp, buffer_shp, distances, buffer_unit, "distance", "NONE", "FULL")
    print(f"多环缓冲区 {buffer_shp} 创建完成！")


# 定义汇总交集的函数
def summarize_within(buffer_shp, lines_shp, output_gdb, temp_output_fc, buffer_unit="Kilometers"):
    arcpy.analysis.SummarizeWithin(buffer_shp, lines_shp, f"{output_gdb}\\{temp_output_fc}", "KEEP_ALL", None,
                                   "ADD_SHAPE_SUM", buffer_unit, None, "NO_MIN_MAJ", "NO_PERCENT")
    print(f"汇总完成，结果保存到 {output_gdb}\\{temp_output_fc}")


# 定义读取并匹配字段的函数
def read_and_match_fields(gdb, feature_class, distances, output_excel_path):
    if os.path.exists(output_excel_path):
        print(f"{output_excel_path} 已存在，跳过此数据处理。")
        return

    feature_class_path = f"{gdb}\\{feature_class}"
    fields = [f.name for f in arcpy.ListFields(feature_class_path)]
    if len(fields) < 2:
        print(f"要素类 {feature_class} 中字段不足以获取倒数第二和最后一个字段")
        return

    second_last_field = fields[-2]
    last_field = fields[-1]

    # 创建空列表来存储读取的数据
    data = []
    with arcpy.da.SearchCursor(feature_class_path, [second_last_field, last_field]) as cursor:
        for row in cursor:
            data.append(row)

    # 将数据转换为 DataFrame
    df = pd.DataFrame(data, columns=[second_last_field, last_field])

    # 动态调整 distances 的长度，确保与 DataFrame 行数一致
    distances_adjusted = distances[:len(df)]  # 截断多余的元素，保证与 DataFrame 行数一致
    df["Radius (km)"] = np.array(distances_adjusted)

    # 在保存为 Excel 文件之前，将 "Radius (km)" 列移到第一列
    if "Radius (km)" in df.columns:
        df = df[["Radius (km)"] + [col for col in df.columns if col != "Radius (km)"]]
        df["Radius (km)"] = df["Radius (km)"][::-1].values  # 原地反转该列数据顺序

    # 对数据取 log2 并添加新列
    log_df = df.copy()
    for col in df.columns:
        log_col_name = f"log2({col})"
        log_df[log_col_name] = df[col].apply(lambda x: math.log2(x) if x > 0 else np.nan)

    # 保存取 log2 后的数据，不添加 _log2 后缀
    log_df.to_excel(output_excel_path, index=False)
    print(f"取 log2 后的数据已保存到 {output_excel_path}")


# 批量处理城市的函数
def batch_process(points_shp, input_lines_folder, output_gdb, distances, output_folder):
    # 获取所有城市名称
    city_names = []
    with arcpy.da.SearchCursor(points_shp, ["city"]) as cursor:
        for row in cursor:
            city_names.append(row[0])

    for city_name in city_names:
        print(f"处理城市：{city_name}")

        # 选择当前城市的政府点要素
        where_clause = f"city = '{city_name}'"
        selected_points_shp = os.path.join(output_folder, f"{city_name}_selected_points.shp")
        arcpy.Select_analysis(points_shp, selected_points_shp, where_clause)

        # 创建多环缓冲区
        buffer_shp = os.path.join(output_folder, f"{city_name}_buffer.shp")
        if os.path.exists(buffer_shp):
            print(f"已存在缓冲区文件 {buffer_shp}，直接使用。")
        else:
            create_multiple_ring_buffer(selected_points_shp, buffer_shp, distances)

        # 删除临时选择的点要素文件
        arcpy.Delete_management(selected_points_shp)

        # 查找对应的线要素（假设文件名格式为 {年份}_{城市名称}_segment.shp）
        for year in [2014, 2016, 2018, 2020, 2022, 2024]:
            lines_file = f"{year}_{city_name}_segment.shp"
            lines_shp = os.path.join(input_lines_folder, lines_file)

            if not arcpy.Exists(lines_shp):
                print(f"未找到 {lines_shp}，跳过此年份。")
                continue

            # 构建最终的 Excel 结果文件路径
            output_excel_path = os.path.join(output_folder, f"{year}_{city_name}_radius.xlsx")

            # 检查最终的 Excel 结果文件是否已经存在
            if os.path.exists(output_excel_path):
                print(f"{output_excel_path} 已存在，跳过此年份的汇总交集和后续处理。")
                continue

            # 汇总交集
            temp_output_fc = f"{city_name}_{year}_summary"
            summarize_within(buffer_shp, lines_shp, output_gdb, temp_output_fc)

            # 读取汇总结果并输出到 Excel
            read_and_match_fields(output_gdb, temp_output_fc, distances, output_excel_path)


# 主程序
def main():
    points_shp = r"E:\长三角2014-2024年道路网分形维计算\data\gov\长三角市政府_41.shp"  # 点要素文件
    input_lines_folder = r"E:\长三角2014-2024年道路网分形维计算\data\segment_data"  # 线要素文件夹
    output_gdb = r"E:\长三角2014-2024年道路网分形维计算\data\road.gdb"  # 输出 GDB
    output_folder = r"E:\长三角2014-2024年道路网分形维计算\result_geo"  # 输出文件夹

    # 设置缓冲区的距离（2km 到 30km）
    distances = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # 批量处理城市
    batch_process(points_shp, input_lines_folder, output_gdb, distances, output_folder)


# 运行主程序
if __name__ == "__main__":
    main()