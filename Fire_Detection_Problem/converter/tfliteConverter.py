from pathlib import Path
import tensorflow as tf
import os
import numpy as np
from typing import Callable, Optional


class TFLiteConverter:
    def __init__(self, modelo, nome_saida="modelo_convertido"):
        """
        modelo: pode ser um objeto Keras carregado em memória ou um Path para o arquivo .keras
        """
        self.modelo = modelo
        self.nome_saida = nome_saida

    def converter_tflite(
        self,
        caminho: Path,
        quantizacao: Optional[str] = None,
        representative_data_dir: Optional[Path] = None,
        rep_samples: int = 100,
        img_size: tuple = (224, 224),
        batch_size: int = 1,
    ) -> Path:
        """
        Converte o modelo para TFLite.

        Args:
            caminho: pasta de saída (Path)
            quantizacao: None|'FLOAT16'|'INT8' - tipo de quantização
            representative_data_dir: Path para pasta com imagens (necessário para INT8)
            rep_samples: número máximo de amostras representativas (INT8)
            img_size: tamanho das imagens esperadas pelo modelo
            batch_size: batch size para o generator representativo

        Retorna:
            Path do arquivo .tflite gerado
        """
        caminho.mkdir(parents=True, exist_ok=True)
        arquivo = caminho / f"{self.nome_saida}.tflite"

        # Se self.modelo for um Path, carregue o modelo Keras
        modelo = self.modelo
        if isinstance(modelo, Path):
            print(f"[INFO] Carregando modelo Keras de: {modelo}")
            modelo = tf.keras.models.load_model(str(modelo))

        # Criar conversor
        converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

        # Sem quantização (FP32) -> conversão padrão
        if quantizacao is None:
            print("[INFO] Conversão FP32 (padrão)")

        elif quantizacao.upper() == 'FLOAT16':
            print("[INFO] Aplicando quantização Float16 para reduzir tamanho e manter precisão" )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        elif quantizacao.upper() == 'INT8':
            print("[INFO] Aplicando quantização INT8 (requere representative dataset)")
            if representative_data_dir is None:
                raise ValueError("Para INT8 é necessário passar representative_data_dir com imagens para calibração")

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            # Forçar ops inteiros
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Definir tipos de entrada/saída como int8 (compatível com full-integer quantization)
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            # Criar generator representativo
            def _representative_gen():
                # Usar tf.data para carregar imagens do diretório (sem labels)
                ds = tf.keras.preprocessing.image_dataset_from_directory(
                    str(representative_data_dir),
                    labels=None,
                    image_size=img_size,
                    batch_size=batch_size,
                    shuffle=True
                )

                count = 0
                for batch in ds:
                    # batch pode ser apenas imagens se labels=None
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        images = batch[0]
                    else:
                        images = batch

                    images = tf.cast(images, tf.float32) / 255.0
                    for i in range(images.shape[0]):
                        if count >= rep_samples:
                            return
                        img = images[i:i+1].numpy().astype(np.float32)
                        yield [img]
                        count += 1

            converter.representative_dataset = _representative_gen

        else:
            raise ValueError(f"Tipo de quantização desconhecido: {quantizacao}")

        modelo_tflite = converter.convert()

        # Salvar modelo convertido
        with open(arquivo, "wb") as f:
            f.write(modelo_tflite)

        print(f"[INFO] Modelo TFLite salvo em: {arquivo}")

        # Comparar tamanhos (opcional)
        self._comparar_tamanho(modelo_path=self.modelo, tflite_path=arquivo)

        return arquivo

    def _comparar_tamanho(self, modelo_path, tflite_path):
        """Compara o tamanho do arquivo original (.keras) com o .tflite"""
        # Descobrir caminho do .keras (se for Path)
        if isinstance(modelo_path, Path):
            keras_path = modelo_path
        else:
            keras_path = Path(f"{self.nome_saida}.keras")

        try:
            size_keras = os.path.getsize(keras_path) / (1024 * 1024)  # MB
            size_tflite = os.path.getsize(tflite_path) / (1024 * 1024)  # MB

            reducao = ((size_keras - size_tflite) / size_keras) * 100 if size_keras > 0 else 0

            print("\n========== COMPARAÇÃO DE TAMANHO ==========")
            print(f"Modelo original (.keras): {size_keras:.2f} MB")
            print(f"Modelo convertido (.tflite): {size_tflite:.2f} MB")
            print(f"Redução de tamanho: {reducao:.2f}%")
            print("==========================================\n")

        except Exception as e:
            print(f"[WARN] Não foi possível calcular tamanhos: {e}")


if __name__ == '__main__':
    # Exemplo de uso rápido (converte o modelo EfficientNetV2-B0 criado em memória)
    try:
        from ..models.efficientnetv2 import build_model
    except Exception:
        # import relativo falhou quando executado diretamente; tenta import absoluto local
        try:
            from Fire_Detection_Problem.models.efficientnetv2 import build_model
        except Exception:
            build_model = None

    if build_model is not None:
        print('[INFO] Construindo modelo EfficientNetV2-B0 em memória para teste de conversão...')
        m = build_model(input_shape=(224, 224, 3), variant='b0', fine_tune=False)
        c = TFLiteConverter(m, nome_saida='efficientnetv2_b0_fp32')
        out = c.converter_tflite(Path('.'), quantizacao=None)
        print(f'[INFO] Conversão de teste salva em: {out}')
