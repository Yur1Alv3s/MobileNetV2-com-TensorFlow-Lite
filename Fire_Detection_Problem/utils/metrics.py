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
    def fmt(v):
        try:
            if v is None:
                return "   -  "
            return f"{float(v):0.4f}"
        except Exception:
            return "   -  "
    def diff(a, b):
        try:
            if a is None or b is None:
                return "   -  "
            return f"{(float(a)-float(b)): +0.4f}"
        except Exception:
            return "   -  "
    print("\n========== COMPARAÇÃO FINAL ==========")
    print(f"Configuração detectada: {km.get('config','N/A')}\n")
    print(f"{'Métrica':<14} {'Keras':>10} {'TFLite':>10} {'Δ(K - T)':>11}")
    print("-"*49)
    def row(label, k_key, t_key):
        k_val = km.get(k_key)
        t_val = tm.get(t_key)
        print(f"{label:<14} {fmt(k_val):>10} {fmt(t_val):>10} {diff(k_val, t_val):>11}")

    row('Accuracy', 'acc', 'acc')
    row('Precision', 'prec', 'prec')
    row('RecallPos', 'rec', 'rec')
    # Recall negativo (especificado pelo usuário)
    if ('rec_neg' in km) or ('rec_neg' in tm):
        row('RecallNeg', 'rec_neg', 'rec_neg')
    # Balanced Accuracy
    if ('balanced_acc' in km) or ('balanced_acc' in tm):
        row('BalancedAcc', 'balanced_acc', 'balanced_acc')
    row('F1-Score', 'f1', 'f1')
    if 'roc_auc' in km or 'roc_auc' in tm:
        row('ROC-AUC', 'roc_auc', 'roc_auc')
    if 'pr_auc' in km or 'pr_auc' in tm:
        row('PR-AUC', 'pr_auc', 'pr_auc')

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


def comparar_modelos_keras_tflite(keras_model, tflite_model_path, dataset, nome_keras="Keras", nome_tflite="TFLite", warmup=10, num_samples=None, medir_memoria=True, salvar_csv=True, csv_config="comparacao"):
    """
    Compara um modelo Keras e um modelo TFLite no mesmo `dataset` (tf.data.Dataset que yield (images, labels)).

    Parâmetros:
        keras_model: objeto Keras carregado (tf.keras.Model) ou caminho para arquivo .keras
        tflite_model_path: caminho para arquivo .tflite
        dataset: tf.data.Dataset com (images, labels). Deve ser o mesmo usado para as duas avaliações.
        warmup: número de amostras (aprox.) para aquecer o modelo antes de medir latência
        num_samples: número máximo de amostras a usar para medições (None = usar todo o dataset)
        medir_memoria: se True tenta medir uso de memória extra ao carregar/avaliar (usa psutil se disponível)
        salvar_csv: se True salva as métricas gerais via `salvar_resultados_csv`
        csv_config: string descritiva salva no CSV

    Retorna:
        dict com chaves: "keras", "tflite", "operacional" onde cada item contém métricas e arrays (y_true, y_prob)
    """
    import os, time, math
    import numpy as np
    from tensorflow.keras.models import load_model # type: ignore
    import tensorflow as tf

    # Carrega keras_model se for um caminho
    if isinstance(keras_model, (str, os.PathLike)):
        keras_model = load_model(str(keras_model))

    # Preparar lista de amostras (images, labels) até num_samples
    y_true = []
    imgs = []
    for images, labels in dataset:
        for i in range(images.shape[0]):
            imgs.append(images[i].numpy())
            y_true.append(int(labels[i].numpy()))
            if num_samples is not None and len(imgs) >= num_samples:
                break
        if num_samples is not None and len(imgs) >= num_samples:
            break

    imgs = np.array(imgs)
    y_true = np.array(y_true)
    N = imgs.shape[0]

    # --- KERAS evaluation ---
    # Warmup
    for i in range(min(warmup, N)):
        _ = keras_model.predict(np.expand_dims(imgs[i], axis=0), verbose=0)

    # Measure latencies
    keras_times = []
    keras_probs = []
    for i in range(N):
        t0 = time.perf_counter()
        p = keras_model.predict(np.expand_dims(imgs[i], axis=0), verbose=0).ravel()[0]
        t1 = time.perf_counter()
        keras_times.append(t1 - t0)
        keras_probs.append(float(p))
    keras_times = np.array(keras_times)
    keras_probs = np.array(keras_probs)

    # Predictive metrics Keras
    keras_pred = (keras_probs > 0.5).astype(int)
    km_acc, km_prec, km_rec, km_f1, km_roc, km_pr = _calc_binary_probs_to_metrics(y_true, keras_probs, keras_pred)
    km_cm = confusion_matrix(y_true, keras_pred)
    # Balanced Accuracy ( (Recall_pos + Recall_neg) / 2 )
    if km_cm.shape == (2, 2):
        tn, fp, fn, tp = km_cm.ravel()
        km_rec_pos = km_rec  # já calculado (tp/(tp+fn))
        km_rec_neg = tn / (tn + fp + 1e-8)
        km_bal_acc = (km_rec_pos + km_rec_neg) / 2.0
    else:  # fallback se algo inesperado ocorrer
        km_rec_neg = np.nan
        km_bal_acc = np.nan

    # --- TFLite evaluation ---
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Warmup
    for i in range(min(warmup, N)):
        x = np.expand_dims(imgs[i].astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])

    tflite_times = []
    tflite_probs = []
    for i in range(N):
        x = np.expand_dims(imgs[i].astype(np.float32), axis=0)
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        y_hat = interpreter.get_tensor(output_details[0]['index']).ravel()[0]
        t1 = time.perf_counter()
        tflite_times.append(t1 - t0)
        tflite_probs.append(float(y_hat))
    tflite_times = np.array(tflite_times)
    tflite_probs = np.array(tflite_probs)

    tflite_pred = (tflite_probs > 0.5).astype(int)
    tm_acc, tm_prec, tm_rec, tm_f1, tm_roc, tm_pr = _calc_binary_probs_to_metrics(y_true, tflite_probs, tflite_pred)
    tm_cm = confusion_matrix(y_true, tflite_pred)
    if tm_cm.shape == (2, 2):
        tn, fp, fn, tp = tm_cm.ravel()
        tm_rec_pos = tm_rec
        tm_rec_neg = tn / (tn + fp + 1e-8)
        tm_bal_acc = (tm_rec_pos + tm_rec_neg) / 2.0
    else:
        tm_rec_neg = np.nan
        tm_bal_acc = np.nan

    # --- Operacional: tamanhos em disco ---
    try:
        keras_size = None
        # tentar extrair path se objeto tem atributo 'saved_model' ou 'filepath' (fallback)
        if hasattr(keras_model, 'save') and hasattr(keras_model, 'weights'):
            # o usuário pode ter passado o objeto; não há arquivo associado
            keras_size = None
        if isinstance(keras_model, tf.keras.Model):
            keras_path = None
        else:
            keras_path = str(keras_model)
        tflite_size = os.path.getsize(str(tflite_model_path)) if os.path.exists(str(tflite_model_path)) else None

    except Exception:
        keras_size = None
        tflite_size = None

    # --- Operacional: memória (RSS delta ao carregar/interp) ---
    mem_info = {}
    if medir_memoria:
        try:
            import psutil
            p = psutil.Process()
            before = p.memory_info().rss
            # carregar tflite numa nova instância e medir
            interpreter2 = tf.lite.Interpreter(model_path=str(tflite_model_path))
            interpreter2.allocate_tensors()
            after = p.memory_info().rss
            mem_info['tflite_rss_delta_bytes'] = after - before
            # keras model já carregado; medir custo aproximado não-trivial (skip precise)
            mem_info['note'] = 'rss delta measured for tflite interpreter only; keras in-memory size not measured precisely'
        except Exception:
            mem_info['error'] = 'psutil not available or measurement failed'

    # --- Agrupa resultados ---
    result_keras = {
        'name': nome_keras,
        'acc': float(km_acc), 'prec': float(km_prec), 'rec': float(km_rec), 'f1': float(km_f1),
        'balanced_acc': float(km_bal_acc) if not np.isnan(km_bal_acc) else None,
        'rec_neg': float(km_rec_neg) if not np.isnan(km_rec_neg) else None,
        'roc_auc': float(km_roc) if not math.isnan(km_roc) else None,
        'pr_auc': float(km_pr) if not math.isnan(km_pr) else None,
        'cm': km_cm, 'y_true': y_true, 'y_prob': keras_probs,
        'latency_ms_mean': float(keras_times.mean()*1000), 'latency_ms_std': float(keras_times.std()*1000)
    }

    result_tflite = {
        'name': nome_tflite,
        'acc': float(tm_acc), 'prec': float(tm_prec), 'rec': float(tm_rec), 'f1': float(tm_f1),
        'balanced_acc': float(tm_bal_acc) if not np.isnan(tm_bal_acc) else None,
        'rec_neg': float(tm_rec_neg) if not np.isnan(tm_rec_neg) else None,
        'roc_auc': float(tm_roc) if not math.isnan(tm_roc) else None,
        'pr_auc': float(tm_pr) if not math.isnan(tm_pr) else None,
        'cm': tm_cm, 'y_true': y_true, 'y_prob': tflite_probs,
        'latency_ms_mean': float(tflite_times.mean()*1000), 'latency_ms_std': float(tflite_times.std()*1000)
    }

    operacional = {
        'tflite_size_bytes': tflite_size,
        'keras_size_bytes': keras_size,
        'mem_info': mem_info
    }

    # salvar resumo via salvar_resultados_csv
    if salvar_csv:
        try:
            salvar_resultados_csv([result_keras, result_tflite], csv_config, total_amostras=N)
        except Exception:
            pass

    return {'keras': result_keras, 'tflite': result_tflite, 'operacional': operacional}
