!python run_smap.py 
Found Kaggle dataset at: /kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
Copying /kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/labeled_anomalies.csv to /kaggle/working/carla_bilstm_v1/datasets/smap/labeled_anomalies.csv...
Copying /kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data/data/train to /kaggle/working/carla_bilstm_v1/datasets/smap/train...
Copying /kaggle/input/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data/data/test to /kaggle/working/carla_bilstm_v1/datasets/smap/test...

==============================
STARTING EXPERIMENTS
==============================
GPU available: Tesla T4

Running dataset: P-1
Max GPU Memory after P-1: 936.97 MB

Running dataset: S-1
Max GPU Memory after S-1: 936.97 MB

Running dataset: E-1
Max GPU Memory after E-1: 937.67 MB

Running dataset: E-2
Max GPU Memory after E-2: 937.67 MB

Running dataset: E-3
Max GPU Memory after E-3: 937.67 MB

Running dataset: E-4
Max GPU Memory after E-4: 937.67 MB

Running dataset: E-5
Max GPU Memory after E-5: 937.67 MB

Running dataset: E-6
Max GPU Memory after E-6: 937.67 MB

Running dataset: E-7
Max GPU Memory after E-7: 937.67 MB

Running dataset: E-8
Max GPU Memory after E-8: 937.67 MB

Running dataset: E-9
Max GPU Memory after E-9: 937.67 MB

Running dataset: E-10
Max GPU Memory after E-10: 937.67 MB

Running dataset: E-11
Max GPU Memory after E-11: 937.67 MB

Running dataset: E-12
Max GPU Memory after E-12: 937.67 MB

Running dataset: E-13
Max GPU Memory after E-13: 937.67 MB

Running dataset: A-1
Max GPU Memory after A-1: 937.67 MB

Running dataset: D-1
Max GPU Memory after D-1: 937.67 MB

Running dataset: P-2
Max GPU Memory after P-2: 937.67 MB

Running dataset: P-3
Max GPU Memory after P-3: 937.67 MB

Running dataset: D-2
Max GPU Memory after D-2: 937.67 MB

Running dataset: D-3
Max GPU Memory after D-3: 937.67 MB

Running dataset: D-4
Max GPU Memory after D-4: 937.67 MB

Running dataset: A-2
Max GPU Memory after A-2: 937.67 MB

Running dataset: A-3
Max GPU Memory after A-3: 937.67 MB

Running dataset: A-4
Max GPU Memory after A-4: 937.67 MB

Running dataset: G-1
Max GPU Memory after G-1: 937.67 MB

Running dataset: G-2
Max GPU Memory after G-2: 937.67 MB

Running dataset: D-5
Max GPU Memory after D-5: 937.67 MB

Running dataset: D-6
Max GPU Memory after D-6: 937.67 MB

Running dataset: D-7
Max GPU Memory after D-7: 937.67 MB

Running dataset: F-1
Max GPU Memory after F-1: 937.67 MB

Running dataset: P-4
Max GPU Memory after P-4: 937.67 MB

Running dataset: G-3
Max GPU Memory after G-3: 937.67 MB

Running dataset: T-1
Max GPU Memory after T-1: 937.67 MB

Running dataset: T-2
Max GPU Memory after T-2: 937.67 MB

Running dataset: D-8
Max GPU Memory after D-8: 937.67 MB

Running dataset: D-9
Max GPU Memory after D-9: 937.67 MB

Running dataset: F-2
Max GPU Memory after F-2: 937.67 MB

Running dataset: G-4
Max GPU Memory after G-4: 937.67 MB

Running dataset: T-3
Max GPU Memory after T-3: 937.67 MB

Running dataset: D-11
Max GPU Memory after D-11: 937.67 MB

Running dataset: D-12
Error running pretext for D-12: Command '['/usr/bin/python3', 'carla_pretext.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/pretext/carla_pretext_smap.yml', '--fname', 'D-12']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/kaggle/working/carla_bilstm_v1/carla_pretext.py", line 240, in <module>
    main()
  File "/kaggle/working/carla_bilstm_v1/carla_pretext.py", line 196, in main
    tmp_loss = pretext_train(train_dataloader, model, criterion, optimizer, epoch, prev_loss, device=device)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/carla_bilstm_v1/utils/train_utils.py", line 50, in pretext_train
    return loss
           ^^^^
UnboundLocalError: cannot access local variable 'loss' where it is not associated with a value

Error running classification for D-12: Command '['/usr/bin/python3', 'carla_classification.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/classification/carla_classification_smap.yml', '--fname', 'D-12']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/kaggle/working/carla_bilstm_v1/carla_classification.py", line 216, in <module>
    main()
  File "/kaggle/working/carla_bilstm_v1/carla_classification.py", line 52, in main
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/carla_bilstm_v1/utils/common_config.py", line 173, in get_aug_train_dataset
    dataloader = torch.load(p['contrastive_dataset'], weights_only=False)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1500, in load
    with _open_file_like(f, "rb") as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 768, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 749, in __init__
    super().__init__(open(name, mode))  # noqa: SIM115
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'results/smap/D-12/pretext/con_train_dataset.pth'

Max GPU Memory after D-12: 937.67 MB

Running dataset: B-1
Max GPU Memory after B-1: 937.67 MB

Running dataset: G-6
Max GPU Memory after G-6: 937.75 MB

Running dataset: G-7
Max GPU Memory after G-7: 937.75 MB

Running dataset: P-7
Max GPU Memory after P-7: 937.75 MB

Running dataset: R-1
Max GPU Memory after R-1: 937.75 MB

Running dataset: A-5
Max GPU Memory after A-5: 937.75 MB

Running dataset: A-6
Max GPU Memory after A-6: 937.75 MB

Running dataset: A-7
Max GPU Memory after A-7: 937.75 MB

Running dataset: D-13
Max GPU Memory after D-13: 937.75 MB

Running dataset: P-2
Max GPU Memory after P-2: 937.75 MB

Running dataset: A-8
Max GPU Memory after A-8: 937.75 MB

Running dataset: A-9
Max GPU Memory after A-9: 937.75 MB

Running dataset: F-3
Max GPU Memory after F-3: 937.75 MB

==============================
DONE ALL SMAP DATASETS
Total time: 13680.31 s
Avg / dataset: 248.73 s
==============================

Time results saved to results/smap/time_results.json

==============================
STARTING EVALUATION (PAPER STYLE)
==============================
P-1: PR-AUC=0.3317, TP=673, FP=1033, FN=975
S-1: PR-AUC=0.6467, TP=354, FP=9, FN=393
E-1: PR-AUC=0.1763, TP=632, FP=2604, FN=474
E-2: PR-AUC=0.7977, TP=1168, FP=157, FN=529
E-3: PR-AUC=0.9147, TP=2569, FP=0, FN=643
E-4: PR-AUC=0.9302, TP=2535, FP=0, FN=368
E-5: PR-AUC=0.2291, TP=155, FP=306, FN=465
E-6: PR-AUC=0.8655, TP=294, FP=0, FN=71
E-7: PR-AUC=0.6202, TP=290, FP=63, FN=290
E-8: PR-AUC=0.1423, TP=482, FP=2368, FN=440
E-9: PR-AUC=0.2339, TP=340, FP=973, FN=310
E-10: PR-AUC=0.1349, TP=668, FP=3538, FN=252
E-11: PR-AUC=0.1269, TP=618, FP=2948, FN=275
E-12: PR-AUC=0.2376, TP=536, FP=1347, FN=645
E-13: PR-AUC=0.1190, TP=886, FP=5094, FN=165
A-1: PR-AUC=0.7987, TP=300, FP=0, FN=84
D-1: PR-AUC=0.9515, TP=2879, FP=16, FN=379
P-2: PR-AUC=0.9132, TP=1298, FP=1, FN=227
P-3: PR-AUC=0.3089, TP=1324, FP=1471, FN=311
D-2: PR-AUC=0.9289, TP=3808, FP=761, FN=467
D-3: PR-AUC=0.9958, TP=3280, FP=3, FN=134
D-4: PR-AUC=0.9958, TP=3187, FP=0, FN=60
A-2: PR-AUC=0.2494, TP=136, FP=337, FN=274
A-3: PR-AUC=0.0711, TP=329, FP=3453, FN=156
A-4: PR-AUC=0.0635, TP=286, FP=3182, FN=124
G-1: PR-AUC=0.0872, TP=308, FP=2121, FN=112
G-2: PR-AUC=0.0764, TP=265, FP=2507, FN=75
D-5: PR-AUC=0.1603, TP=66, FP=87, FN=284
D-6: PR-AUC=0.1242, TP=58, FP=220, FN=322
D-7: PR-AUC=0.6875, TP=2252, FP=1456, FN=449
F-1: PR-AUC=0.1208, TP=110, FP=611, FN=290
P-4: PR-AUC=0.4003, TP=1122, FP=651, FN=218
G-3: PR-AUC=0.0484, TP=302, FP=4493, FN=48
T-1: PR-AUC=0.4249, TP=1655, FP=2038, FN=479
T-2: PR-AUC=0.5465, TP=1457, FP=428, FN=327
D-8: PR-AUC=0.2716, TP=300, FP=306, FN=50
D-9: PR-AUC=0.6348, TP=927, FP=181, FN=228
F-2: PR-AUC=0.5593, TP=1859, FP=490, FN=1097
G-4: PR-AUC=0.8439, TP=275, FP=0, FN=55
T-3: PR-AUC=0.8094, TP=588, FP=0, FN=194
D-11: PR-AUC=0.5132, TP=232, FP=52, FN=128
Skip D-12 (missing files)
B-1: PR-AUC=0.1542, TP=263, FP=506, FN=107
G-6: PR-AUC=0.4867, TP=229, FP=38, FN=171
G-7: PR-AUC=0.9156, TP=998, FP=0, FN=167
P-7: PR-AUC=0.9547, TP=1721, FP=253, FN=229
R-1: PR-AUC=0.2091, TP=300, FP=412, FN=80
A-5: PR-AUC=0.7555, TP=249, FP=0, FN=101
A-6: PR-AUC=0.7694, TP=248, FP=0, FN=92
A-7: PR-AUC=0.9806, TP=2319, FP=0, FN=111
D-13: PR-AUC=0.6736, TP=288, FP=10, FN=172
P-2: PR-AUC=0.9132, TP=1298, FP=1, FN=227
A-8: PR-AUC=0.9758, TP=3476, FP=0, FN=329
A-9: PR-AUC=1.0000, TP=3864, FP=0, FN=0
F-3: PR-AUC=0.0226, TP=340, FP=7736, FN=0

==============================
FINAL RESULTS (PAPER STYLE)
==============================
PRECISION: 0.5096
RECALL: 0.7938
F1: 0.6207
AUPR_MEAN: 0.5167
AUPR_STD: 0.3436
TP: 56396
TN: 294607
FP: 54261
FN: 14653
TOTAL_DATASETS: 54

================ SUMMARY ================
Precision : 0.5096
Recall    : 0.7938
F1-score  : 0.6207
AUPR mean : 0.5167
AUPR std  : 0.3436

Total time     : 13680.31 s
Avg / dataset  : 248.73 s
GPU max memory : 937.75 MB

fix lỗi này và thêm đoạn auto reduce window size tương tự vậy:
if len(self.data) < wsz:
            if len(self.data) >= 4096:
                wsz = 4096
            elif len(self.data) >= 2048:
                wsz = 2048