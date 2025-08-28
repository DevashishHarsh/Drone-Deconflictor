import argparse
import json
from DronePath import DronePath

parser = argparse.ArgumentParser(description="CLI for DronePath Deconflictor (Linear mode)")

parser.add_argument("--dfile", "-d", type=str, required=True,
                    help="Path to the drones dataset (JSON)")
parser.add_argument("--lfile", "-l", type=str, required=True,
                    help="Path to the leader drone dataset (JSON)")
parser.add_argument("--point", "-p", action="store_true",
                    help="Show detailed collision info instead of just Collision/Clear")
parser.add_argument("--tmin", "-n", type=int, default=None,
                    help="Tmin value for collision detection")
parser.add_argument("--tmax", "-m", type=int, default=None,
                    help="Tmax value for collision detection")
parser.add_argument("--dist", "-s", type=float, default=1.0,
                    help="Distance factor threshold for collision detection (default: 1.0)")

args = parser.parse_args()

# Instantiate DronePath
d = DronePath(300)
d.load_json_points(args.dfile)

# Create leader drone spline
sp_data = d.create_drone(args.lfile)

# Run collision check
result = d.check_spline_with_drones(
    sp_data=sp_data,
    dist=args.dist,
    plot_tmin=args.tmin,
    plot_tmax=args.tmax,
    detailed=args.point
)

if args.point:
    # Detailed output -> JSON formatted
    print(json.dumps(result, indent=2))
else:
    # Summary only
    print("Collision" if result else "Clear")
