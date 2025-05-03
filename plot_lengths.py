import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lib.labels import label_count, number_to_label

matplotlib.rc("font", size=18)
lengths = np.load("lengths.npy")

print(np.sum(lengths < 10) / len(lengths))

plt.hist(lengths, bins=30, range=(0, 30))
plt.xlim(0, 30)
plt.xlabel("Duration of clip (s)")
plt.ylabel("Number of clips")
plt.title("Audio clip durations")
plt.tight_layout()
plt.show()


tra = np.load("tra.npy")
vaa = np.load("vaa.npy")

iters = np.arange(0, 400)
iters_v = np.arange(0, 400, 5)

plt.plot(iters, tra)
plt.plot(iters_v, vaa)
plt.ylim(50, 100)
plt.xlim(0, 400)
plt.title("Accuracy during training")
plt.legend(["Training", "Validation"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()

trl = np.load("trl.npy")
vll = np.load("val.npy")

plt.plot(iters, trl)
plt.plot(iters_v, vll)
plt.xlim(0, 400)
plt.ylim(0, 3)
plt.title("Training loss")
plt.legend(["Training", "Validation"])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.show()

label_names = [number_to_label(i)[-20:] for i in range(label_count())]
tacc = np.load("tacc.npy")
plt.bar(label_names, 100 * tacc, align="edge")
plt.xlim(-0.3, label_count())
plt.xticks(size=5, rotation=90)
plt.ylim(0, 100)
plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.title("Training accuracy per class")
plt.tight_layout()
plt.show()

vacc = np.load("vacc.npy")
plt.bar(label_names, 100 * vacc, align="edge")
plt.xlim(-0.3, label_count())
plt.xticks(size=5, rotation=90)
plt.ylim(0, 100)
plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.title("Validation accuracy per class")
plt.tight_layout()
plt.show()

testacc = np.load("testacc.npy")
plt.bar(label_names, 100 * testacc, align="edge")
plt.xlim(-0.3, label_count())
plt.xticks(size=5, rotation=90)
plt.ylim(0, 100)
plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.title("Test accuracy per class")
plt.tight_layout()
plt.show()
