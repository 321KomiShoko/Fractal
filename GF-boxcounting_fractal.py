import os
import pathlib
import logging
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing


# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def get_utm_crs(gdf):
    """根据几何中心自动获取合适的UTM坐标系"""
    try:
        # 转换为WGS84坐标系获取经纬度
        centroid = gdf.to_crs(4326).geometry.unary_union.centroid
        lon, lat = centroid.x, centroid.y

        # 计算UTM带编号
        utm_zone = int((lon + 180) // 6 + 1)
        epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        return CRS.from_epsg(epsg_code)
    except Exception as e:
        logger.warning(f"UTM坐标计算失败，使用默认坐标系: {str(e)}")
        return CRS.from_epsg(32650)  # 默认使用UTM 50N


def generate_fishnet(bounds, size, crs):
    """使用向量化操作高效生成渔网"""
    x_min, y_min, x_max, y_max = bounds
    cols = int(np.ceil((x_max - x_min) / size))
    rows = int(np.ceil((y_max - y_min) / size))

    # 生成网格坐标
    x_coords = np.arange(x_min, x_max + size, size)
    y_coords = np.arange(y_min, y_max + size, size)

    # 创建网格多边形
    polygons = []
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            polygons.append(box(x, y, x + size, y + size))

    return gpd.GeoDataFrame(geometry=polygons, crs=crs), cols * rows


def calculate_box_g_dimension(road_gdf):
    """优化后的盒子覆盖维度计算"""
    results = []
    original_crs = road_gdf.crs

    try:
        # 自动选择UTM坐标系
        utm_crs = get_utm_crs(road_gdf)
        road_utm = road_gdf.to_crs(utm_crs)

        bounds = road_utm.total_bounds
        union_geom = road_utm.unary_union

        # 初始化网格尺寸和进度记录
        current_size = 100  # 初始尺寸500米
        max_size = 20000  # 最大尺寸20000
        step_counter = 0

        while current_size <= max_size:
            logger.debug(f"正在处理尺寸: {current_size}m")

            # 生成渔网
            fishnet, expected_boxes = generate_fishnet(bounds, current_size, utm_crs)

            # 批量相交检查
            intersects = fishnet.geometry.intersects(union_geom)
            valid_boxes = intersects.sum()

            # 记录结果
            if valid_boxes > 0:
                log_size = np.log2(current_size)
                log_num = np.log2(valid_boxes)
                results.append({
                    'box_length': current_size,
                    'box_num': valid_boxes,
                    'log_box_length': log_size,
                    'log_box_num': log_num
                })

                # 提前终止条件：连续3次相同数量或数量为1
                if len(results) >= 3 and all(r['box_num'] == valid_boxes for r in results[-3:]):
                    logger.info(f"提前终止：连续三次相同盒子数量 {valid_boxes}")
                    break
                if valid_boxes == 1:
                    logger.info("提前终止：盒子数量降为1")
                    break

            # 指数级增长尺寸
            current_size *= 2
            step_counter += 1

    except Exception as e:
        logger.error(f"维度计算失败: {str(e)}", exc_info=True)
        return pd.DataFrame()
    finally:
        # 恢复原始坐标系
        road_gdf = road_gdf.to_crs(original_crs)

    return pd.DataFrame(results)


def parse_filename(filename):
    """解析标准化文件名"""
    parts = filename.split('_')
    return parts[0], parts[1]


def process_single_shp(shp_path, output_dir):
    """处理单个SHP文件的独立函数"""
    try:
        # 解析文件名
        filename = shp_path.stem
        year, city = parse_filename(filename)
        if not year or not city:
            logger.warning(f"文件名解析失败: {filename}")
            return False

        # 构建输出路径
        output_file = output_dir / f"{year}_{city}_geo_box.xlsx"
        if output_file.exists():
            logger.info(f"跳过已存在文件: {output_file.name}")
            return True

        logger.info(f"开始处理: {year} {city}")

        # 读取数据
        gdf = gpd.read_file(shp_path)
        if len(gdf) == 0:
            logger.warning(f"空文件: {filename}")
            return False

        # 执行计算
        result_df = calculate_box_g_dimension(gdf)
        if result_df.empty:
            logger.warning(f"计算结果为空: {filename}")
            return False

        # 保存结果
        result_df.to_excel(output_file, index=False)
        logger.info(f"成功保存: {output_file.name}")

        # 显式释放内存
        del gdf
        return True

    except Exception as e:
        logger.error(f"处理失败 {shp_path.name}: {str(e)}", exc_info=True)
        return False
    finally:
        # 确保内存回收
        if 'gdf' in locals():
            del gdf


def process_directory(input_path, output_path, workers=None):
    """并行化目录处理"""
    input_dir = pathlib.Path(input_path)
    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取待处理文件列表（过滤已处理文件）
    shp_files = [f for f in input_dir.glob("*.shp")
                 if not (output_dir / f"{f.stem}_geo_box.xlsx").exists()]

    # 自动设置工作进程数
    if workers is None:
        workers = min(os.cpu_count(), len(shp_files)) or 1

    logger.info(f"启动并行处理，使用{workers}个工作进程，共{len(shp_files)}个文件...")

    # 使用进程池处理
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(
            partial(process_single_shp, output_dir=output_dir),
            shp_files
        ))

    # 统计处理结果
    success_count = sum(results)
    failure_count = len(shp_files) - success_count
    logger.info(f"处理完成: 成功{success_count} 失败{failure_count} 跳过{len(shp_files) - failure_count}")


if __name__ == "__main__":
    # 配置路径
    input_folder = r"E:\长三角2014-2024年道路网分形维计算\data\stroke_data"
    output_folder = r"E:\长三角2014-2024年道路网分形维计算\result_geo"

    # 配置并行工作进程数（None为自动检测）
    PARALLEL_WORKERS = 10

    # 输入验证
    if not pathlib.Path(input_folder).exists():
        logger.error(f"输入目录不存在: {input_folder}")
        raise FileNotFoundError(f"目录不存在: {input_folder}")

    # 执行并行处理
    logger.info("==== 开始并行批量处理 ====")
    process_directory(input_folder, output_folder, workers=PARALLEL_WORKERS)
    logger.info("==== 处理完成 ====")

