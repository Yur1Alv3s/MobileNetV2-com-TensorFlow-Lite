"""
Treino curto da cabeça do EfficientNetV2
- Congela base (pretrained)
- Congela BatchNorms no base
- Treina somente a cabeça por 5 epochs com lr=1e-3
Salva modelo em Modelos/fire_effnV2_head_short.keras
"""
import os
from pathlib import Path
import tensorflow as tf
from models.efficientnetv2 import build_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_preproc
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

THIS_DIR = Path(__file__).resolve().parent  # Fire_Detection_Problem folder
PROJECT_ROOT = THIS_DIR.parent
DATA_DIR = THIS_DIR / 'data' / 'Flame2'
MODEL_DIR = PROJECT_ROOT / 'Modelos'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 5

# Carregar datasets (assumindo estrutura train/val com subpastas de classes)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(DATA_DIR / 'train'),
    labels='inferred', label_mode='int', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    str(DATA_DIR / 'val'),
    labels='inferred', label_mode='int', image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False
)

# Normalização: por padrão usa /255.0, mas se houver preprocess específico para EfficientNetV2, usa-lo
if eff_preproc is not None:
    train_ds = train_ds.map(lambda x,y: (eff_preproc(tf.cast(x, tf.float32)), y))
    val_ds = val_ds.map(lambda x,y: (eff_preproc(tf.cast(x, tf.float32)), y))
else:
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x,y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x,y: (normalization_layer(x), y))

# Construir o modelo (base congelado)
model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), fine_tune=False)

# Congelar BatchNorms do base_model (se existirem)
for layer in model.layers[0].layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Compilar com LR maior para a cabeça
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Treinar só a cabeça (já que base está congelado)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

save_path = MODEL_DIR / 'fire_effnV2_head_short.keras'
model.save(save_path)
print('Modelo salvo em', save_path)

# Avaliar e gerar estatísticas de predição completas
import numpy as np

def evaluate_and_report(ds, name):
    y_true = []
    y_pred = []
    probs = []
    for x,y in ds:
        p = model.predict(x)
        preds = (p.flatten() > 0.5).astype(int)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        probs.extend(p.flatten().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    probs = np.array(probs)
    print(f"--- {name} ---")
    print('probs stats:', probs.min(), probs.max(), probs.mean(), probs.std())
    print('unique preds:', np.unique(y_pred, return_counts=True))
    print('confusion matrix:\n', confusion_matrix(y_true, y_pred))
    try:
        print(classification_report(y_true, y_pred))
    except Exception:
        pass

evaluate_and_report(train_ds, 'train')
evaluate_and_report(val_ds, 'val')
