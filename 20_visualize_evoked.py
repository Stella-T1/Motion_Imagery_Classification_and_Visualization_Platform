# License: BSD-3-Clause
# Copyright the MNE-Python contributors.
# %%
import numpy as np
import mne
import matplotlib.pyplot as plt

def analyze_evokeds(evokeds_list, conditions, times_topo, times_anim, ts_args, topomap_args, joint_plot):
    evks = dict(zip(conditions, evokeds_list))
    
    # Append additional topomap plot
    fig, axes = plt.subplots(1, 5, figsize=(8, 2))
    kwargs = dict(times=0.1, show=False, time_unit='s')

    for idx, condition in enumerate(conds):
        evks[condition].plot_topomap(times=0.1, axes=[axes[idx], axes[-1]], show=False, time_unit='s')

    for ax, title in zip(axes, ['Aud/L', 'Aud/R', 'Vis/L', 'Vis/R', 'Colorbar']):
        ax.set_title(title)

    plt.show()
    
    # Plotting scalp topographies
    for condition in conditions:
        evks[condition].plot_topomap(times_topo, ch_type="mag", ncols=8, nrows="auto")

    # Animating the topomap
    for condition in conditions:
        fig, anim = evks[condition].animate_topomap(times=times_anim, ch_type="mag", frame_rate=2, blit=True)

    # Joint plot
    ts_args = ts_args
    topomap_args = topomap_args
    titles = conditions

    for condition, title in zip(conditions, titles):
        evks[condition].plot_joint(title=title, times=joint_plot, ts_args=ts_args, topomap_args=topomap_args)

root = mne.datasets.sample.data_path() / "MEG" / "sample"
evoked_file = root / "sample_audvis-ave.fif"
evokeds_list = mne.read_evokeds(evoked_file, baseline=(None, 0), proj=True, verbose=False)

conds = ("aud/left", "aud/right", "vis/left", "vis/right")
all_times = np.arange(-0.2, 0.5, 0.03)
specific_times = np.arange(0.04, 0.35, 0.01)

ts_args = dict(gfp=True, time_unit='s')
topomap_args = dict(sensors=False, time_unit='s')

analyze_evokeds(evokeds_list, conds, all_times, specific_times, ts_args, topomap_args, [0.08, 0.20])

# %%
