import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from roomscan.draw_plan import setup_axes


def test_plan_axes_not_mirrored():
    fig, ax = plt.subplots()
    setup_axes(ax)
    assert ax.yaxis_inverted()      # screen-up is -z when looking down from +y
    plt.close(fig)
