from __future__ import annotations
import os
from typing import Optional, Dict
import os, tensorflow as tf
from utils.model_info import summary
from convert.tf_lite_converter import convert_like_classifier
from evaluation import eval_jhu
from data.verify import verify_integrity
from utils.paths import load_cfg
from train.trainer import train_model
from evaluation.eval_jhu import eval_jhu, eval_jhu_lite
from pathlib import Path
from evaluation.evaluate import evaluate_keras, evaluate_tflite, compare_keras_vs_tflite
from utils.predict_test_gen import predict_test_gen  # gera CSV no split test (sem GT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel("ERROR")


def main():
    cfg = load_cfg()  # lê seu YAML padrão
    keras_path = "Crowd_Counting_Problem/artifacts/models/crowd_mnv2_s8_best.keras"
    tflite_path = "Crowd_Counting_Problem/artifacts/models/crowd_mnv2_s8.tflite"

    metrics_keras = eval_jhu(model_path=keras_path,jhu_root="Crowd_Counting_Problem/data/JHU-Test")
    print("\n[TEST-JHU / Keras]")
    print(f"  n: {metrics_keras['Qtd_imgs']}")
    print(f"  MAE:  {metrics_keras['MAE']:.2f}")
    print(f"  RMSE: {metrics_keras['RMSE']:.2f}")
    print(f"  CSV:  {metrics_keras['out_csv']}")

    metrics_tflite = eval_jhu_lite(tflite_path=tflite_path,jhu_root="Crowd_Counting_Problem/data/JHU-Test",save_csv=True)

    print("\n[TEST-JHU / TFLite]")
    print(f"  n: {metrics_tflite['Qtd_imgs']}")
    print(f"  MAE:  {metrics_tflite['MAE']:.2f}")
    print(f"  RMSE: {metrics_tflite['RMSE']:.2f}")
    print(f"  CSV:  {metrics_tflite['out_csv']}")

    summary(keras_path, details=True, benchmark=True, runs=100, warmup=20)
    summary(tflite_path, details=True, benchmark=True, runs=100, warmup=20)



    # 1) AVALIAR O .KERAS NO SPLIT DE VALIDAÇÃO (com GT)

    res_val = evaluate_keras(keras_path, split="val", cfg=cfg)
    print("\n[VAL / Keras]")
    print(f"  n: {res_val['n']}")
    print(f"  MAE:  {res_val['MAE']:.2f}")
    print(f"  RMSE: {res_val['RMSE']:.2f}")
    print(f"  CSV:  {res_val['csv_path']}")

    # 2) (OPCIONAL) AVALIAR UMA VERSÃO TFLite NO VAL
    tflite_path = "Crowd_Counting_Problem/artifacts/models/crowd_mnv2_s8.tflite"
    res_tfl = evaluate_tflite(tflite_path, split="", cfg=cfg, delegate="cpu")
    print("\n[VAL / TFLite]")
    print(f"  n: {res_tfl['n']}")
    print(f"  MAE:  {res_tfl['MAE']:.2f}")
    print(f"  RMSE: {res_tfl['RMSE']:.2f}")
    print(f"  CSV:  {res_tfl['csv_path']}")

    # 3) (OPCIONAL) COMPARAR KERAS × TFLite (consistência de contagem)
    res_cmp = compare_keras_vs_tflite(keras_path, tflite_path, split="val", cfg=cfg)
    print("\n[Compare Keras × TFLite]")
    print(f"  n: {res_cmp['n']}")
    print(f"  mean_abs_delta: {res_cmp['mean_abs_delta']:.3f}")
    print(f"  p95_abs_delta:  {res_cmp['p95_abs_delta']:.3f}")
    print(f"  CSV:            {res_cmp['csv_path']}")

    # 4) (TESTE SEM GT) RODAR INFERÊNCIA CEGA E GERAR CSV DE SUBMISSÃO
    #    -> requer que você já tenha implementado utils/predict_test_gen.py
        #   (essa função deve ler data/lists/test.txt e salvar id,pred_count)
    csv_test = predict_test_gen(model_path=keras_path, cfg=cfg, split="test",
                                save_maps=False, previews=0)
    print("\n[TEST / Keras — sem GT]")
    print(f"  CSV de predições: {csv_test}")

    # TESTE COM O JHU-TEST (se você já tiver implementado evaluation/eval_jhu.py)
    metrics = eval_jhu(model_path=keras_path,jhu_root="data/JHU-Test")
    print("\n[TEST-JHU / Keras]")
    print(f"  n: {metrics['Qtd_imgs']}")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  CSV:  {metrics['out_csv']}")
    

if __name__ == "__main__":
    main()
