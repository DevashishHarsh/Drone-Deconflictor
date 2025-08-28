import argparse
import json
from GaussianPoints import GaussianPoints

parser = argparse.ArgumentParser(description="CLI for GaussianPoints Deconflictor (Gaussian mode)")

parser.add_argument("--dfile", "-d", type=str, required=True,
                    help="Path to the drones dataset (JSON)")
parser.add_argument("--lfile", "-l", type=str, required=True,
                    help="Path to the leader drone dataset (JSON)")
parser.add_argument("--type", "-t", type=int, choices=[0, 1], default=0,
                    help="Dataset type: 0 = Position only, 1 = Position + Velocity")
parser.add_argument("--point", "-p", action="store_true",
                    help="Show detailed collision info instead of just Collision/Clear")
parser.add_argument("--tmin", "-n", type=int, default=None,
                    help="Tmin value for collision detection")
parser.add_argument("--tmax", "-m", type=int, default=None,
                    help="Tmax value for collision detection")
parser.add_argument("--conf", "-c", type=float, default=0.95,
                    help="Confidence level (default: 0.95)")

args = parser.parse_args()

g = GaussianPoints(300)
g.load_json_points(args.dfile)
sp_data = g.create_drone(args.lfile, type=args.type)

result = g.check_spline_with_drones(
    sp_data=sp_data,
    confidence_level=args.conf,
    plot_tmin=args.tmin,
    plot_tmax=args.tmax,
    detailed=args.point
)

if args.point:
    print(json.dumps(result, indent=2))
else:
    print("Collision" if result else "Clear")
