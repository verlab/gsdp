import numpy as np

# ------------------------------------------------------------------------------------ Plotting helpers


def do_plot_top_n(ax, idx_class, index, img):
           
            ax.set_title('Label: {}, Index: {}'.format(idx_class, index))
            ax.imshow(img, cmap='gray')

# -------------------------------------------------------------------------------a stats---plotting helpers


def _do_plot_a_stats(ax, main_title, titles, key, value, cmarker, num_classes):
        dx = np.arange(int(num_classes))
        ax.set_title(main_title)  # main_title = 'Class {}'.format(idx_class)
        ax.set_ylabel(titles[key])
        ax.set_xlabel(titles['xlabel'])
        ax.set_xlim((-1, num_classes + 1))  # plot X axes in center
        if key == 'meanstd':
            # data_v = value[0] + value[1]
            lmax = np.max(value[0] + value[1])
            lmin = np.min(value[0] - value[1])
            ax.set_ylim((lmin - 2, lmax + 2))  # plot y axes in center
            ax.errorbar(dx, value[0], value[1], linestyle='None', marker='o')
        else:
            ax.plot(dx, value, cmarker) 

# --------------------------------------------------------------------------------- f-stats helpers ---plotting helpers


def _do_plot_f_stats_cc(fig, ax, title, value, features_len):
        names = list(range(features_len))
        ticks = np.arange(0, features_len, 1)
        ax.set_title(title)  # title = 'Covariance Class_{}'.format(class_i)
        cax = ax.matshow(value, vmin=-1, vmax=1)  # value = cov
        fig.colorbar(cax)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)  

# ------------------------------------------------------------------------------------- fstats


def _do_plot_f_stats(ax, idx_class, titles, key, value, cmarker, features_len):

        dx = np.arange(int(features_len))
        # ax.plot([1,2,3], [4,5,6], 'k.')
        ax.set_title('Class {}'.format(idx_class))
        ax.set_ylabel(titles[key])
        ax.set_xlabel(titles['xlabel'])
        ax.set_xlim((-1, features_len + 1))
        if key == 'meanstd':
            ax.errorbar(dx, value[0], value[1], linestyle='None', marker='o')
        else:
            ax.plot(dx, value, cmarker)                            