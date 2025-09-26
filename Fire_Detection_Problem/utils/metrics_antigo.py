import numpy as np
from sklearn.utils import compute_class_weight
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from Fire_Detection_Problem.utils.logs import *
import csv, os
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)



# Precisão (Precision): Mede a proporção de verdadeiros positivos entre todos os exemplos classificados como positivos.
# Indica o quanto das previsões positivas realmente são corretas.
def calcula_precisao(y_true, y_pred):
    y_pred = tf.round(y_pred)  # arrendonda as previsões para 0 ou 1
    tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    fp = tf.reduce_sum(tf.cast((y_true == 0) & (y_pred == 1), tf.float32))
    precisao = tp / (tp + fp + 1e-8)
    return precisao.numpy()

# Recall (Sensibilidade): Mede a proporção de verdadeiros positivos entre todos os exemplos que realmente são positivos.
# Indica o quanto dos casos positivos foram corretamente identificados pelo modelo.
def calcula_recall(y_true, y_pred):
    y_pred = tf.round(y_pred) # arrendonda as previsões para 0 ou 1
    tp = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 1), tf.float32))
    fn = tf.reduce_sum(tf.cast((y_true == 1) & (y_pred == 0), tf.float32))
    recall = tp / (tp + fn + 1e-8)
    return recall.numpy()

# F1-score: Média harmônica entre precisão e recall.
# É uma métrica que equilibra precisão e recall, sendo útil quando há desbalanceamento entre classes.
def calcula_f1(y_true, y_pred):
    p = calcula_precisao(y_true, y_pred)
    r = calcula_recall(y_true, y_pred)
    f1Score = 2 * (p * r) / (p + r + 1e-8)
    return f1Score


def plot_matriz_de_confusao(y_true, y_pred):
    y_pred = tf.round(y_pred).numpy().astype(int)
    y_true = y_true.numpy().astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()


def calcular_pesos_classes(ds_treino):
    logCalculandoPesos()##########################################################

    # Extrai todos os rótulos do conjunto de treino
    rotulos_treino = tf.concat([labels for _, labels in ds_treino], axis=0).numpy().astype(int)
    # Calcula os pesos das classes
    classes = np.unique(rotulos_treino)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=rotulos_treino)
    # Cria o dicionário para usar no fit
    class_weight_dict = dict(zip(classes, class_weights))
    # exibe os pesos calculados
    for classe, peso in class_weight_dict.items():
        print(f"Classe {classe}: Peso {peso:.4f}")
        print("")
    return class_weight_dict




def contar_labels(dataset, nome):
    labels = [int(label.numpy()) for _, label in dataset.unbatch()]
    contagem = Counter(labels)
    print(f"\nContagem de labels em {nome}:")
    for k in sorted(contagem.keys()):
        print(f"Classe {k}: {contagem[k]}")
    return contagem

def _plot_side_by_side_confusions(keras_cm, tflite_cm, labels=("NoFire", "Fire"), save_path: str | None = None):
    import numpy as np
    import matplotlib.pyplot as plt
    # garante numpy arrays
    kcm = np.array(keras_cm)
    tcm = np.array(tflite_cm)
    vmax = max(kcm.max(), tcm.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, cm, title in zip(axes, [kcm, tcm], ["Keras", "TFLite"]):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Matriz de Confusão - {title}")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)

        # grid das células
        ax.set_xticks(np.arange(-.5, cm.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, cm.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

        # anotações com contraste
        thresh = vmax * 0.6
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm[i, j])
                color = "white" if val > thresh else "#111111"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10)

    # colorbar único e discreto
    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.04)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def _print_table_comparacao(km, tm):
    def fmt(v): return f"{v:0.4f}"
    def d(a, b): return f"{(a-b):+0.4f}"
    print("\n========== COMPARAÇÃO FINAL ==========")
    print(f"Configuração detectada: {km.get('config','N/A')}\n")
    print(f"{'Métrica':<12} {'Keras':>9} {'TFLite':>9} {'Δ(K - T)':>9}")
    print("-"*39)
    print(f"{'Accuracy':<12} {fmt(km['acc']):>9} {fmt(tm['acc']):>9} {d(km['acc'], tm['acc']):>9}")
    print(f"{'Precision':<12} {fmt(km['prec']):>9} {fmt(tm['prec']):>9} {d(km['prec'], tm['prec']):>9}")
    print(f"{'Recall':<12} {fmt(km['rec']):>9} {fmt(tm['rec']):>9} {d(km['rec'], tm['rec']):>9}")
    print(f"{'F1-Score':<12} {fmt(km['f1']):>9} {fmt(tm['f1']):>9} {d(km['f1'], tm['f1']):>9}")
    if 'roc_auc' in km and 'roc_auc' in tm:
        print(f"{'ROC-AUC':<12} {fmt(km['roc_auc']):>9} {fmt(tm['roc_auc']):>9} {d(km['roc_auc'], tm['roc_auc']):>9}")
    if 'PR-AUC' in km and 'PR-AUC' in tm:
        print(f"{'PR-AUC':<12} {fmt(km['pr_auc']):>9} {fmt(tm['pr_auc']):>9} {d(km['pr_auc'], tm['pr_auc']):>9}")

def _plot_barras(km, tm, save_path: str | None = None):
    labels = ["Acc", "Prec", "Rec", "F1"]
    k_vals = [km['acc'], km['prec'], km['rec'], km['f1']]
    t_vals = [tm['acc'], tm['prec'], tm['rec'], tm['f1']]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x - w/2, k_vals, width=w, label="Keras")
    ax.bar(x + w/2, t_vals, width=w, label="TFLite")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Comparação de Métricas")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()

def _calc_binary_probs_to_metrics(y_true, y_prob, y_pred):
    # y_true: (N,), y_prob: (N,), y_pred binário
    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # Com probabilidade, dá pra calcular áreas
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = np.nan
    return acc, prec, rec, f1, roc_auc, pr_auc

def avaliar_keras(modelo, dataset, nome="Keras"):
    # coleta probs e verdadeiros
    y_true, y_prob = [], []
    loss, acc = modelo.evaluate(dataset, verbose=0)
    for images, labels in dataset:
        probs = modelo.predict(images, verbose=0).ravel()
        y_true.extend(labels.numpy()); y_prob.extend(probs)
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob > 0.5).astype(int)
    acc2, prec, rec, f1, roc_auc, pr_auc = _calc_binary_probs_to_metrics(y_true, y_prob, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n========== Avaliando {nome} ==========")
    print(f"[{nome}] Loss: {loss:.4f}, Acc: {acc2:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_true, y_pred))
    return {
        "name": nome, "loss": float(loss), "acc": float(acc2), "prec": float(prec),
        "rec": float(rec), "f1": float(f1), "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "cm": cm, "y_true": y_true, "y_prob": y_prob
    }

def avaliar_tflite(model_path, dataset, nome="TFLite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true, y_prob = [], []
    for images, labels in dataset:
        for i in range(images.shape[0]):
            x = np.expand_dims(images[i].numpy(), axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            y_hat = interpreter.get_tensor(output_details[0]['index']).ravel()[0]
            y_prob.append(float(y_hat))
            y_true.append(int(labels[i].numpy()))
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob > 0.5).astype(int)

    acc, prec, rec, f1, roc_auc, pr_auc = _calc_binary_probs_to_metrics(y_true, y_prob, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n========== Avaliando {nome} ==========")
    print(f"[{nome}] Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_true, y_pred))
    return {
        "name": nome, "acc": float(acc), "prec": float(prec), "rec": float(rec), "f1": float(f1),
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "cm": cm,
        "y_true": y_true, "y_prob": y_prob
    }

def salvar_resultados_csv(resultados, config, total_amostras, arquivo="resultados_comparacao.csv"):
    cabecalho = ["DataHora", "Configuração", "TotalAmostras", "Modelo", "Acc", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    existe = os.path.isfile(arquivo)
    with open(arquivo, mode="a", newline="") as f:
        w = csv.writer(f)
        if not existe: w.writerow(cabecalho)
        for r in resultados:
            # r = dict de métricas
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                config, total_amostras, r["name"], r["acc"], r["prec"], r["rec"], r["f1"], r.get("roc_auc", np.nan), r.get("pr_auc", np.nan)
            ])
    print(f"[INFO] Resultados salvos em {arquivo}")

def comparar_e_mostrar(keras_dict, tflite_dict, config="N/A", labels=("Classe 0","Classe 1"), salvar_png: str | None = None):
    # acopla config para impressão
    keras_dict = {**keras_dict, "config": config}
    tflite_dict = {**tflite_dict, "config": config}
    _print_table_comparacao(keras_dict, tflite_dict)
    _plot_side_by_side_confusions(keras_dict["cm"], tflite_dict["cm"], labels=labels)
    _plot_barras(keras_dict, tflite_dict, save_path=salvar_png)
    return keras_dict, tflite_dict


# ===============================================================
# Utilitários
# ===============================================================

def detectar_configuracao(modelo):
    """Verifica se o modelo está com fine-tuning (camadas do MobileNetV2 treináveis)"""
    base_model = modelo.layers[0]  # MobileNetV2 é a primeira camada
    trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
    frozen_layers = len(base_model.layers) - trainable_layers

    if trainable_layers > 0:
        return f"MobileNetV2 Fine-Tuning ({trainable_layers} camadas treináveis, {frozen_layers} congeladas)"
    else:
        return "MobileNetV2 Congelado (sem fine-tuning)"

def plotar_matrizes(keras_cm, tflite_cm, labels=("Classe 0", "Classe 1")):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    for ax, cm, title in zip(axes, [keras_cm, tflite_cm], ["Keras", "TFLite"]):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Matriz de Confusão - {title}")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()


##########################################################################