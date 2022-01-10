import matplotlib.pyplot as plt
import numpy as np


def plot_result(save_path, image, proba, attack_image, attack_proba):
    left = np.arange(0, 10)
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(np.array(np.squeeze(image * 255), dtype="uint8"), "gray")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.array(np.squeeze(attack_image * 255), dtype="uint8"), "gray")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.bar(left, proba)
    ax3.set_xticks(left)
    ax3.set_ylim(0, 1)
    ax3.set_title(np.argmax(proba))

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.bar(left, attack_proba)
    ax4.set_xticks(left)
    ax4.set_ylim(0, 1)
    ax4.set_title(np.argmax(attack_proba))

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
