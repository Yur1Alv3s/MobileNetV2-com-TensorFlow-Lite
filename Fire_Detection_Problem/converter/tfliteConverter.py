from pathlib import Path
import os
from typing import Callable, Optional, List, Tuple, Union
from Fire_Detection_Problem.loaders.loader import classification_representative_dataset_generator
from Fire_Detection_Problem.loaders.mdcount_data import regression_representative_dataset_generator
import numpy as np
import tensorflow as tf


class TFLiteConverter:
    def __init__(self, modelo: Union[Path, tf.keras.Model], nome_saida: str = "modelo_convertido"):
        """
        modelo: objeto Keras em memória OU Path para arquivo .keras
        """
        self.modelo = modelo
        self.nome_saida = nome_saida

    def converter_tflite(
        self,
        caminho: Path,
        quantizacao: Optional[str] = None,
        representative_data_dir: Optional[Path] = None,
        rep_samples: int = 100,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 1,
        # --- novos parâmetros para escolher o gerador correto ---
        task: str = "classification",                 # "classification" | "regression"
        split_list: Optional[List[str]] = None,       # usado na regressão (opcional)
        preprocess_fn: Optional[Callable] = None,     # passe o mesmo preprocess do treino (ex.: MobileNetV2.preprocess_input)
    ) -> Path:
        """
        Converte o modelo Keras para TFLite.

        Args:
            caminho: pasta de saída.
            quantizacao:
                - None (FP32)
                - "FLOAT16"
                - "INT8_FULL" (full integer, requer representative_dataset)
                - "INT8_DR" | "INT8_DYNAMIC" | "DYNAMIC" (INT8 Dynamic Range - sem representative_dataset)
            representative_data_dir:
                - Classificação: dir contendo subpastas 'fire' e 'nofire'
                - Regressão: dir com as imagens (ex.: .../train_data/images)
            rep_samples: nº de amostras para calibração INT8 full.
            img_size: (H, W) de entrada do modelo.
            batch_size: batch para o gerador de calibração (cada yield sai como [1,H,W,3]).
            task: "classification" ou "regression" para escolher o gerador.
            split_list: (regressão) lista de nomes de imagens a usar; se None, usa todas da pasta.
            preprocess_fn: função de preprocess usada no treino; se None, usa o default do gerador.

        Retorna:
            Path do arquivo .tflite gerado.
        """
        caminho.mkdir(parents=True, exist_ok=True)
        arquivo = caminho / f"{self.nome_saida}.tflite"

        # Carrega modelo se necessário
        modelo = self.modelo
        if isinstance(modelo, Path):
            print(f"[INFO] Carregando modelo Keras de: {modelo}")
            modelo = tf.keras.models.load_model(str(modelo), compile=False)

        converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

        q = (quantizacao or "FP32").upper()

        # ======= Sem quantização (FP32) =======
        if q in ("FP32", "NONE"):
            print("[INFO] Conversão FP32 (padrão).")

        # ======= Float16 =======
        elif q in ("FP16", "FLOAT16"):
            print("[INFO] Quantização Float16 (pesos em float16, I/O em float32).")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        # ======= INT8 Dynamic Range (sem representative dataset) =======
        elif q in ("INT8_DR"):
            print("[INFO] Quantização INT8 Dynamic Range (pesos int8, ativações em float32).")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Não definir representative_dataset
            # Não forçar supported_ops para INT8-only
            # Não alterar inference_input_type/output_type

        # ======= INT8 Full Integer (com representative dataset) =======
        elif q == "INT8_FULL":
            print("[INFO] Quantização INT8 Full-Integer (requer representative dataset).")
            if representative_data_dir is None:
                raise ValueError(
                    "Para INT8 full é necessário 'representative_data_dir' apontando para as imagens."
                )

            # Otimização + ops inteiros + I/O int8
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            # Seleciona o gerador conforme a tarefa
            if task.lower() in ("classification", "cls", "bin", "binary"):
                if classification_representative_dataset_generator is None:
                    raise ImportError(
                        "Não foi possível importar 'classification_representative_dataset_generator' do módulo loader."
                    )
                print(f"[INFO] Usando generator de CLASSIFICAÇÃO (dir={representative_data_dir}).")
                rep_gen = classification_representative_dataset_generator(
                    base_dir=representative_data_dir,
                    img_size=img_size,
                    rep_samples=rep_samples,
                    batch_size=batch_size,
                    use_augmentation=False,
                    preprocess_fn=preprocess_fn,
                )

            elif task.lower() in ("regression", "reg", "count", "crowd"):
                if regression_representative_dataset_generator is None:
                    raise ImportError(
                        "Não foi possível importar 'regression_representative_dataset_generator' (mdcount_data)."
                    )
                print(f"[INFO] Usando generator de REGRESSÃO (dir={representative_data_dir}).")
                rep_gen = regression_representative_dataset_generator(
                    images_dir=representative_data_dir,
                    split_list=split_list,
                    img_size=img_size,
                    rep_samples=rep_samples,
                    shuffle=True,
                    preprocess_fn=preprocess_fn,
                )

            else:
                raise ValueError("Parâmetro 'task' deve ser 'classification' ou 'regression'.")

            converter.representative_dataset = rep_gen

        else:
            raise ValueError(f"Tipo de quantização desconhecido: {quantizacao}")

        # ---- Executa a conversão ----
        tflite_bytes = converter.convert()

        # Salva arquivo
        with open(arquivo, "wb") as f:
            f.write(tflite_bytes)

        print(f"[INFO] Modelo TFLite salvo em: {arquivo}")
        self._comparar_tamanho(modelo_path=self.modelo, tflite_path=arquivo)
        return arquivo

    def _comparar_tamanho(self, modelo_path, tflite_path):
        """Compara tamanho do .keras vs .tflite (se possível)."""
        if isinstance(modelo_path, Path):
            keras_path = modelo_path
        else:
            keras_path = Path(f"{self.nome_saida}.keras")

        try:
            size_keras = os.path.getsize(keras_path) / (1024 * 1024)
            size_tflite = os.path.getsize(tflite_path) / (1024 * 1024)
            reducao = ((size_keras - size_tflite) / size_keras) * 100 if size_keras > 0 else 0.0

            print("\n========== COMPARAÇÃO DE TAMANHO ==========")
            print(f"Modelo original (.keras): {size_keras:.2f} MB")
            print(f"Modelo convertido (.tflite): {size_tflite:.2f} MB")
            print(f"Redução de tamanho: {reducao:.2f}%")
            print("==========================================\n")

        except Exception as e:
            print(f"[WARN] Não foi possível calcular tamanhos: {e}")
