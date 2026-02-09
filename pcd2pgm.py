import argparse
import open3d as o3d
import numpy as np
import cv2
import os



def main(args):
    pcd_file = args.pcd_file
    map_output_name = args.map_output_name
    z_min = args.z_min
    z_max = args.z_max
    resolution = args.resolution

    # Basic validation
    if resolution <= 0:
        print("ERROR: resolution must be > 0")
        exit()
    if z_min >= z_max:
        print("ERROR: --z-min must be less than --z-max")
        exit()
    # ---------------------

    print(f"Loading {pcd_file}...")
    if not os.path.exists(pcd_file):
        print(f"ERROR: {pcd_file} not found! Please copy your file here.")
        exit()

    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    print(f"Total points: {len(points)}")

    # 1. Filter by Z height (slice the map)
    mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    sliced_points = points[mask]

    if len(sliced_points) == 0:
        print("Error: No points found in the requested Z range. Check your map coordinates.")
        exit()

    print(f"Points in slice: {len(sliced_points)}")

    # 2. Project to 2D (XY plane)
    xy_points = sliced_points[:, :2]

    # 3. Calculate bounds and grid size
    min_x, min_y = np.min(xy_points, axis=0)
    max_x, max_y = np.max(xy_points, axis=0)

    width = int(np.ceil((max_x - min_x) / resolution)) + 1
    height = int(np.ceil((max_y - min_y) / resolution)) + 1

    print(f"Map Size: {width} x {height} pixels")

    # 4. Create Image (255 = Free/White, 0 = Occupied/Black)
    # Initialize with White (Free space assumption for simplicity)
    map_img = np.ones((height, width), dtype=np.uint8) * 255

    # Fill occupied pixels
    for x, y in xy_points:
        px = int((x - min_x) / resolution)
        py = int((y - min_y) / resolution)
        # Flip Y for image coordinates (bottom-left origin vs top-left image)
        py = height - 1 - py
        map_img[py, px] = 0  # Black

    # 5. Save PGM
    cv2.imwrite(f"{map_output_name}.pgm", map_img)
    print(f"Saved {map_output_name}.pgm")

    # 6. Save YAML
    yaml_content = f"""image: {map_output_name}.pgm
    resolution: {resolution}
    origin: [{min_x}, {min_y}, 0.0]
    negate: 0
    occupied_thresh: 0.65
    free_thresh: 0.196
    """
    with open(f"{map_output_name}.yaml", "w") as f:
        f.write(yaml_content)
    print(f"Saved {map_output_name}.yaml")


if __name__ == "__main__":
    # --- CLI / CONFIGURATION ---
    # required: pcd input path and output map basename
    # optional: z_min, z_max, resolution (defaults preserved)
    parser = argparse.ArgumentParser(
        description="Convert a PCD pointcloud to a 2D occupancy map (PGM + YAML)."
    )
    parser.add_argument("pcd_file", help="input .pcd file path")
    parser.add_argument("map_output_name", help="output map basename (no extension)")
    parser.add_argument("--z-min", dest="z_min", type=float, default=-0.5,
                        help="minimum Z for slice (default: -0.5)")
    parser.add_argument("--z-max", dest="z_max", type=float, default=-0.2,
                        help="maximum Z for slice (default: -0.2)")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="meters per pixel (default: 0.05)")
    args = parser.parse_args()