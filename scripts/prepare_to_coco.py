import os, json
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

def geojson_to_coco(image_dir, output_json, category_map, make_relative=False, skip_missing=False):
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    tif_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

    for tif_file in tqdm(tif_files):
        base = tif_file[:-4]
        geojson_file = os.path.join(image_dir, f"{base}.geojson")
        if not os.path.exists(geojson_file):
            continue

        with rasterio.open(os.path.join(image_dir, tif_file)) as src:
            width, height = src.width, src.height

        gdf = gpd.read_file(geojson_file)
        if 'roof_mater' not in gdf.columns:
            # Skip if no roof_type column
            continue
        gdf['category_id'] = gdf['roof_mater'].map(category_map)

        img_path = os.path.join(image_dir, tif_file)
        if make_relative:
            img_path = os.path.relpath(img_path)

        images.append({
            "id": img_id,
            "file_name": img_path,
            "width": width,
            "height": height
        })

        for _, row in gdf.iterrows():
            cat_id = row.get("category_id")
            if cat_id is None:
                if skip_missing:
                    continue
                # assign 0 or ignore; here we skip
                continue
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
            for poly in polys:
                if not isinstance(poly, Polygon):
                    continue
                x, y = poly.exterior.coords.xy
                coords = [c for pair in zip(x, y) for c in pair]
                xmin, ymin, xmax, ymax = poly.bounds
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cat_id),
                    "segmentation": [coords],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": poly.area,
                    "iscrowd": 0
                })
                ann_id += 1
        img_id += 1

    categories = [{"id": v, "name": k} for k, v in category_map.items()]
    coco = {"images": images, "annotations": annotations, "categories": categories}

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Saved COCO annotations to {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert roof GeoJSONs to COCO.")
    parser.add_argument("--image-dir", required=True, help="Directory with .tif/.geojson pairs")
    parser.add_argument("--out", required=True, help="Output COCO JSON path")
    parser.add_argument("--categories", nargs='+', help="Category mappings like corrugated=1 tile=2 other=3", required=True)
    parser.add_argument("--relative", action="store_true", help="Store relative image paths")
    parser.add_argument("--skip-missing", action="store_true", help="Skip features with unmapped roof_type")
    args = parser.parse_args()

    cat_map = {}
    for kv in args.categories:
        name, id_str = kv.split("=", 1)
        cat_map[name] = int(id_str)

    geojson_to_coco(args.image_dir, args.out, cat_map, make_relative=args.relative, skip_missing=args.skip_missing)
