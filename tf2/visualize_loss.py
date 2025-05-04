import matplotlib.pyplot as plt
import numpy as np
import os


# Kaydedilen loss'ları görselleştir
def plot_saved_loss():
    loss_file = 'loss_history.npy'

    if os.path.exists(loss_file):
        loss_history = np.load(loss_file)

        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title('SimCLR Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Loss grafiği kaydedildi: loss_plot.png")
        print(f"Total steps: {len(loss_history)}")
        print(f"Initial loss: {loss_history[0]:.4f}")
        print(f"Final loss: {loss_history[-1]:.4f}")
    else:
        print(f"Loss dosyası bulunamadı: {loss_file}")
        print("Önce simclr_basic.py dosyasını çalıştırarak eğitim yapın.")


if __name__ == "__main__":
    plot_saved_loss()