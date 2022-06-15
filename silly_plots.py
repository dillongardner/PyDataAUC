import matplotlib.pyplot as plt
import numpy as np


def make_skill_mix_plot():
    with plt.xkcd():

        technical_skills = [85, 50]
        other_skills = [15, 50]
        second_tick = 1.0
        width = 0.5

        fig = plt.figure()
        ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
        ax.bar([0, second_tick], technical_skills, width, label='Technical Skills')
        ax.bar([0, second_tick], other_skills, width, bottom=technical_skills, label='Other Skills')
        ax.spines.right.set_color('none')
        ax.spines.top.set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks([0, second_tick])
        ax.set_xticklabels(['WE THINK LEADS\nTO SUCCESS', 'ACTUALLY LEADS\nTO SUCCESS'])
        ax.set_xlim([-0.5, 1.9])
        ax.set_yticks([])
        ax.set_ylim([0, 110])
        ax.legend(loc=(0.59,0.12))

        ax.set_title("BUSINESS + DATA SCIENCE")

        return ax, fig

if __name__ == '__main__':
    _ = make_skill_mix_plot()
    plt.show()