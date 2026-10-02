"""Two captures of the same place, same tier: does the same room give the same walls?"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roomscan.compare_plans import align_plans, match_walls   # noqa: E402


def main():
    a, b = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    R2, t = align_plans(a, b)
    rows = match_walls(a, b, R2, t)
    ok = sum(r["pass"] for r in rows)
    print(f"matched walls: {len(rows)}, within gate (<=1 cm or <=0.5%): {ok}")
    for r in sorted(rows, key=lambda r: -r["diff_m"])[:15]:
        print(f"  {r['a']:>12} vs {r['b']:>12}: {r['len_a']:.3f} vs {r['len_b']:.3f} m, diff {r['diff_m']*100:.1f} cm"
              f" {'PASS' if r['pass'] else 'FAIL'}")
    Path(sys.argv[3] if len(sys.argv) > 3 else "outputs/repeatability.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
