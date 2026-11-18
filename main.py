"""Fashion-MNIST CNN ödevi — Ilgın Göl, 02220224035."""

import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.10
OUTPUT = Path(__file__).resolve().parent / "ciktilar"


def convolution_demo():
    """Ödevdeki 16 filtrenin eğitimden önceki ağırlıklarını çiz."""
    tf.keras.utils.set_random_seed(SEED)
    conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation="relu",
        kernel_initializer="he_normal",
    )
    inputs = tf.random.normal((1, 128, 128, 1))
    outputs = conv(inputs)
    weights, biases = conv.get_weights()

    figure, axes = plt.subplots(4, 4, figsize=(7, 5.65), layout="constrained")
    limit = float(np.max(np.abs(weights)))
    for index, axis in enumerate(axes.flat):
        values = weights[:, :, 0, index]
        heatmap = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_title(f"Filtre {index + 1}", fontsize=9, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
        for (row, column), value in np.ndenumerate(values):
            color = "white" if abs(value) > limit * 0.66 else "#152339"
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      fontsize=7.5, color=color)
    figure.colorbar(heatmap, ax=axes, shrink=0.84, label="Ağırlık değeri")
    figure.savefig(OUTPUT / "01_baslangic_filtreleri.png", dpi=200, facecolor="white")
    plt.close(figure)
    return {
        "giris": list(inputs.shape), "cikis": list(outputs.shape),
        "agirlik_boyutu": list(weights.shape), "bias_boyutu": list(biases.shape),
        "ilk_filtre": weights[:, :, 0, 0].tolist(), "bias": biases.tolist(),
    }


def pooling_demo():
    """Disjoint 2×2 blokların maksimumlarını elle hesaplanan değerlerle karşılaştır."""
    matrix = [[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 8, 3], [4, 1, 5, 9]]
    inputs = tf.reshape(tf.constant(matrix, dtype=tf.float32), (1, 4, 4, 1))
    outputs = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(inputs)
    pooled = outputs.numpy()[0, :, :, 0].astype(int)
    manual = [[6, 4], [7, 9]]
    np.testing.assert_array_equal(pooled, manual)

    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.65), layout="constrained")
    for axis, values, title in zip(axes, [np.array(matrix), pooled],
                                   ["Giriş  •  4 × 4", "Max Pooling  •  2 × 2"]):
        axis.imshow(values, cmap="Blues", vmin=0, vmax=12)
        axis.set_title(title, fontsize=12, fontweight="bold", pad=9)
        axis.set_xticks([])
        axis.set_yticks([])
        for (row, column), value in np.ndenumerate(values):
            axis.text(column, row, str(value), ha="center", va="center",
                      fontsize=14, color="white" if value >= 7 else "#152339")
    for direction in (axes[0].axhline, axes[0].axvline):
        direction(1.5, color="#C15739", linewidth=2)
    figure.savefig(OUTPUT / "02_max_pooling.png", dpi=200, facecolor="white")
    plt.close(figure)
    return {"giris": matrix, "cikis": pooled.tolist(), "elle": manual,
            "giris_boyutu": list(inputs.shape), "cikis_boyutu": list(outputs.shape)}


def activations_demo():
    inputs = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    relu = tf.keras.layers.ReLU()(inputs)
    logits = tf.constant([[2.0, 1.0, 0.1]])
    probabilities = tf.keras.layers.Softmax()(logits)
    return {
        "relu_giris": inputs.numpy().tolist(), "relu_cikis": relu.numpy().tolist(),
        "softmax_giris": logits.numpy()[0].tolist(),
        "softmax_cikis": probabilities.numpy()[0].tolist(),
        "softmax_toplam": float(tf.reduce_sum(probabilities)),
        "softmax_tahmin": int(tf.argmax(probabilities[0])),
    }


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32") / 255.0
    x_test = x_test[..., np.newaxis].astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def train_experiment(first_filters, data):
    x_train, y_train, x_test, y_test = data
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(first_filters, (3, 3), activation="relu", name="conv_1"),
        tf.keras.layers.MaxPooling2D((2, 2), name="pool_1"),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu", name="conv_2"),
        tf.keras.layers.MaxPooling2D((2, 2), name="pool_2"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(64, activation="relu", name="dense_64"),
        tf.keras.layers.Dense(10, activation="softmax", name="softmax_10"),
    ], name=f"fashion_mnist_{first_filters}_filtre")
    model.summary(line_length=76)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    start = perf_counter()
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=VALIDATION_SPLIT, shuffle=True, verbose=2)
    fit_seconds = perf_counter() - start
    test = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0, return_dict=True)
    result = {
        "ilk_katman_filtre": first_filters, "sure_saniye": fit_seconds,
        "test_dogrulugu": float(test["accuracy"]), "test_kaybi": float(test["loss"]),
        "parametre": model.count_params(), "gecmis": dict(history.history),
        "katmanlar": [
            {"ad": layer.name, "cikis": list(layer.output.shape[1:]),
             "parametre": layer.count_params()}
            for layer in model.layers
        ],
    }
    print(f"{first_filters} filtre | test doğruluğu: {test['accuracy']:.4%} | "
          f"test kaybı: {test['loss']:.4f} | eğitim: {fit_seconds:.2f} sn", flush=True)
    return result


def plot_training(experiments, filename):
    rows = len(experiments)
    figure, axes = plt.subplots(rows, 2, figsize=(8.1, 2.55 * rows + 0.2),
                                layout="constrained", squeeze=False)
    epochs = range(1, EPOCHS + 1)
    for row, result in enumerate(experiments):
        count = result["ilk_katman_filtre"]
        for axis, metric, title, scale in zip(axes[row], ["accuracy", "loss"],
                                              ["Doğruluk (%)", "Çapraz entropi kaybı"], [100, 1]):
            history = result["gecmis"]
            axis.plot(epochs, np.array(history[metric]) * scale, color="#23578A",
                      marker="o", label="Eğitim", linewidth=1.9)
            axis.plot(epochs, np.array(history[f"val_{metric}"]) * scale, color="#BC562E",
                      linestyle="--", marker="s", label="Doğrulama", linewidth=1.7)
            axis.set_title(f"{count} filtre  •  {title}", fontweight="bold", fontsize=10)
            axis.set_xlabel("Epoch")
            axis.set_xticks(list(epochs))
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
    for column, metric, scale in [(0, "accuracy", 100), (1, "loss", 1)]:
        values = [value * scale for result in experiments for key in [metric, f"val_{metric}"]
                  for value in result["gecmis"][key]]
        margin = (max(values) - min(values)) * 0.11
        for axis in axes[:, column]:
            axis.set_ylim(min(values) - margin, max(values) + margin)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    figure.savefig(OUTPUT / filename, dpi=220, facecolor="white")
    plt.close(figure)


def save_text_report(results):
    def fmt(value, precision=2):
        return f"{value:.{precision}f}".replace(".", ",")

    first, second = results["deneyler"]
    pooling = results["pooling"]
    lines = [
        "FASHION-MNIST DENEY ÇIKTISI",
        "Ilgın Göl | 02220224035", "",
        "MAX POOLING — pencere 2 × 2, adım 2",
        "Giriş matrisi",
        *(" ".join(str(value) for value in row) for row in pooling["giris"]),
        "TensorFlow çıktısı ve elle hesaplanan sonuç",
        *(" ".join(str(value) for value in row) for row in pooling["cikis"]),
        "", "TEST SONUÇLARI VE DENEY KARŞILAŞTIRMASI",
        "54.000 eğitim, 6.000 doğrulama, 10.000 test görseli. Beş epoch, 64 örneklik grup, Adam.",
        "İkinci konvolüsyon katmanında iki deneyde de 16 filtre kullanıldı.",
        f"{'Ölçüt':<36} {'8 filtre':>12} {'16 filtre':>12}",
    ]
    measurements = [
        ("Eğitim doğruluğu · son epoch", "%", [first["gecmis"]["accuracy"][-1] * 100,
                                                second["gecmis"]["accuracy"][-1] * 100], 2),
        ("Doğrulama doğruluğu · son epoch", "%", [first["gecmis"]["val_accuracy"][-1] * 100,
                                                   second["gecmis"]["val_accuracy"][-1] * 100], 2),
        ("Test doğruluğu", "%", [r["test_dogrulugu"] * 100 for r in (first, second)], 2),
        ("Test kaybı", "", [r["test_kaybi"] for r in (first, second)], 4),
        ("Eğitim süresi (sn)", "", [r["sure_saniye"] for r in (first, second)], 2),
        ("Eğitilebilir parametre", "", [r["parametre"] for r in (first, second)], 0),
    ]
    for name, unit, (left, right), precision in measurements:
        left_value = unit + fmt(left, precision)
        right_value = unit + fmt(right, precision)
        lines.append(f"{name:<36} {left_value:>12} {right_value:>12}")
    lines.extend(["", "Süre, doğrulama dahil model.fit çağrısını kapsar.",
                  "Tüm epoch değerleri ve katman boyutları deney_sonuclari.json dosyasındadır."])
    (OUTPUT / "04_test_ve_karsilastirma.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    convolution = convolution_demo()
    pooling = pooling_demo()
    activations = activations_demo()
    for label, demo in [("Konvolüsyon", convolution), ("Max Pooling", pooling), ("Aktivasyonlar", activations)]:
        print(f"{label}: {json.dumps(demo, ensure_ascii=False)}", flush=True)

    data = load_data()
    print(f"Eğitim verisi: {data[0].shape} | Test verisi: {data[2].shape}", flush=True)
    experiments = [train_experiment(first_filters, data) for first_filters in (8, 16)]
    plot_training(experiments, "03_egitim_ve_dogrulama.png")
    for experiment in experiments:
        filters = experiment["ilk_katman_filtre"]
        plot_training([experiment], f"03_egitim_ve_dogrulama_{filters}_filtre.png")
    results = {
        "ortam": {"python": platform.python_version(), "tensorflow": tf.__version__,
                  "keras": tf.keras.__version__, "isletim_sistemi": platform.system(),
                  "aygit": "GPU" if tf.config.list_physical_devices("GPU") else "CPU"},
        "ayarlar": {"seed": SEED, "epoch": EPOCHS, "batch_size": BATCH_SIZE,
                    "validation_split": VALIDATION_SPLIT,
                    "egitim_boyutu": list(data[0].shape), "test_boyutu": list(data[2].shape)},
        "konvolusyon": convolution, "pooling": pooling,
        "aktivasyonlar": activations, "deneyler": experiments,
    }
    (OUTPUT / "deney_sonuclari.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    save_text_report(results)


if __name__ == "__main__":
    main()
