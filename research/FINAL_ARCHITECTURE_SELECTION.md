# FINAL ARCHITECTURE SELECTION

ARCHITECTURE SELECTION IS CLOSED. No further backbone proposals.

{
  "status": "architecture_selection_closed",
  "selected_run": "s13_s12_identity_fp32_control_exploratory_seed42",
  "models": [
    {
      "run_id": "s03_efficientnet_b0_none_exploratory_seed42",
      "model": "efficientnet_b0",
      "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
      "checkpoint": "checkpoints/structured/s03_efficientnet_b0_none_exploratory_seed42/best.pt",
      "checkpoint_sha256": "6b7a270de98912a7f7a70ca3c1a78e55f8b5a0fe1406762760699cb23d4ca39c"
    },
    {
      "run_id": "s06_convnext_tiny_none_exploratory_seed42",
      "model": "convnext_tiny",
      "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
      "checkpoint": "checkpoints/structured/s06_convnext_tiny_none_exploratory_seed42/best.pt",
      "checkpoint_sha256": "92381249b6486f54a8db6c7d808e5d421bf7ba1c0f8c7ea370d24542ec247ef7"
    },
    {
      "run_id": "s10_efficientnet_v2_s_none_exploratory_seed42",
      "model": "efficientnet_v2_s",
      "weights": "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
      "checkpoint": "checkpoints/structured/s10_efficientnet_v2_s_none_exploratory_seed42/best.pt",
      "checkpoint_sha256": "120b621b6ddc280c0929273121b149e4c7e90aec4c088c4da0c5b41a1dce62f5"
    }
  ],
  "number_of_models": 3,
  "ensemble_method": "fixed probability soft voting",
  "ensemble_weights": [
    0.3333333333333333,
    0.3333333333333333,
    0.3333333333333333
  ],
  "metrics": {
    "loss": 0.3609271708617435,
    "accuracy": 0.9248170326014638,
    "macro_precision": 0.9032601879784398,
    "macro_recall": 0.8586992323421929,
    "macro_f1": 0.8774541331925992,
    "class_order": [
      "akiec",
      "bcc",
      "bkl",
      "df",
      "mel",
      "nv",
      "vasc"
    ],
    "confusion_matrix": [
      [
        35,
        9,
        4,
        0,
        0,
        1,
        0
      ],
      [
        0,
        73,
        0,
        0,
        1,
        3,
        0
      ],
      [
        5,
        1,
        133,
        0,
        9,
        17,
        0
      ],
      [
        0,
        2,
        1,
        14,
        0,
        0,
        0
      ],
      [
        0,
        0,
        3,
        0,
        132,
        32,
        0
      ],
      [
        0,
        2,
        5,
        0,
        15,
        983,
        2
      ],
      [
        0,
        0,
        0,
        0,
        0,
        1,
        20
      ]
    ],
    "per_class": {
      "akiec": {
        "precision": 0.875,
        "recall": 0.7142857142857143,
        "f1": 0.7865168539325843,
        "support": 49
      },
      "bcc": {
        "precision": 0.8390804597701149,
        "recall": 0.948051948051948,
        "f1": 0.8902439024390244,
        "support": 77
      },
      "bkl": {
        "precision": 0.910958904109589,
        "recall": 0.806060606060606,
        "f1": 0.8553054662379421,
        "support": 165
      },
      "df": {
        "precision": 1.0,
        "recall": 0.8235294117647058,
        "f1": 0.9032258064516129,
        "support": 17
      },
      "mel": {
        "precision": 0.8407643312101911,
        "recall": 0.7904191616766467,
        "f1": 0.8148148148148148,
        "support": 167
      },
      "nv": {
        "precision": 0.9479267116682739,
        "recall": 0.9761668321747765,
        "f1": 0.961839530332681,
        "support": 1007
      },
      "vasc": {
        "precision": 0.9090909090909091,
        "recall": 0.9523809523809523,
        "f1": 0.9302325581395349,
        "support": 21
      }
    }
  },
  "metrics_path": "results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/validation_metrics_identity_fp32.json",
  "confusion_matrix_reference": "results/structured_experiments/s13_s12_four_flip_tta_exploratory_seed42/figures_identity_fp32/confusion_matrix.csv",
  "protocol": "exploratory_image_level",
  "split_sha256": "75ebfcb011d8822e372283218dbb95a89b3e3d3e9323c83519004e2da1790fbb",
  "material_gain": false,
  "material_gate": ">=1percentage point accuracy without F1loss OR >=.01macroF1 without accuracy loss; akiec recall drop<=.05,mel<=.03,df/vasc<=.10; single-seed development heuristic not significance",
  "batch_candidate_status": {
    "s16_efficientnet_b3_none_exploratory_seed42": {
      "status": "failed",
      "attempts": 1,
      "pid": 24144,
      "create_time": 1791097545.2182,
      "command": [
        "C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\.venv\\Scripts\\python.exe",
        "-u",
        "-m",
        "research.train",
        "--config",
        "research/configs/phase3/s16_efficientnet_b3_none_exploratory_seed42.json"
      ],
      "resumed": false,
      "console": "results\\architecture_selection\\final_v1\\s16_efficientnet_b3_none_exploratory_seed42.console.log",
      "exit_code": 1,
      "reason": "\": 7, \"logits_inf\": 0, \"bad_logit_rows\": [6], \"loss_finite\": false, \"probabilities_nonfinite\": 7}\nTraceback (most recent call last):\n  File \"<frozen runpy>\", line 198, in _run_module_as_main\n  File \"<frozen runpy>\", line 88, in _run_code\n  File \"C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\research\\train.py\", line 565, in <module>\n    if __name__=='__main__': main()\n                             ~~~~^^\n  File \"C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\research\\train.py\", line 562, in main\n    with run_lock(config['experiment_id']): run(config,args.resume,fix,args.stop_after_epoch)\n                                            ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\research\\train.py\", line 467, in run\n    all_y,all_p,ids,val_loss,validation_precision=evaluate_validation(model,val_loader,weights,config,experiment.path,epoch,\n                                                  ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n        allow_fp32_recovery=bool(amendment),training_state=dict(optimizer=optimizer.state_dict(),\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(True)))\n            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\research\\train.py\", line 197, in evaluate_validation\n    try:return collect(False)\n               ~~~~~~~^^^^^^^\n  File \"C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\research\\train.py\", line 189, in collect\n    try:num,p=checked_validation_batch(model,images,y,weights,fp32)\n              ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\research\\train.py\", line 163, in checked_validation_batch\n    raise NonfiniteValidation(dict(reason='nonfinite_forward_or_loss',precision='fp32' if fp32 else 'amp_fp16',\n    ...<2 lines>...\n        loss_finite=bool(torch.isfinite(num)),probabilities_nonfinite=int((~torch.isfinite(probabilities)).sum())))\nNonfiniteValidation: Nonfinite validation batch: {\"reason\": \"nonfinite_forward_or_loss\", \"precision\": \"amp_fp16\", \"logits_nan\": 7, \"logits_inf\": 0, \"bad_logit_rows\": [6], \"loss_finite\": false, \"probabilities_nonfinite\": 7}\n"
    },
    "s19_resnet101_none_exploratory_seed42": {
      "status": "completed",
      "attempts": 2,
      "pid": 10172,
      "create_time": 1791104074.0195427,
      "command": [
        "C:\\Users\\MARS\\Desktop\\College\\Minor Project\\Project Development\\.venv\\Scripts\\python.exe",
        "-u",
        "-m",
        "research.train",
        "--config",
        "research/configs/phase3/s19_resnet101_none_exploratory_seed42.json",
        "--resume"
      ],
      "resumed": true,
      "console": "results\\architecture_selection\\final_v1\\s19_resnet101_none_exploratory_seed42.console.log",
      "exit_code": 0
    }
  },
  "test_evaluated": false,
  "architecture_search_closed": true,
  "no_more_backbones": true
}

## Why these models
- efficientnet_b0: accuracy 0.863606, macro-F1 0.770660; compound-scaled CNN. Strongest class scores: nv F1=0.9321, vasc F1=0.8261, bcc F1=0.7805. Corrects these other-member errors: {'s06_convnext_tiny_none_exploratory_seed42': 52, 's10_efficientnet_v2_s_none_exploratory_seed42': 76}. Fixed ensemble selection is supported by saved joint performance, not standalone rank alone.
- convnext_tiny: accuracy 0.917498, macro-F1 0.862831; modern CNN. Strongest class scores: nv F1=0.9581, vasc F1=0.9524, bkl F1=0.8626. Corrects these other-member errors: {'s03_efficientnet_b0_none_exploratory_seed42': 133, 's10_efficientnet_v2_s_none_exploratory_seed42': 89}. Fixed ensemble selection is supported by saved joint performance, not standalone rank alone.
- efficientnet_v2_s: accuracy 0.894877, macro-F1 0.823162; fused/MBConv CNN. Strongest class scores: nv F1=0.9471, vasc F1=0.9333, bcc F1=0.8428. Corrects these other-member errors: {'s03_efficientnet_b0_none_exploratory_seed42': 123, 's06_convnext_tiny_none_exploratory_seed42': 55}. Fixed ensemble selection is supported by saved joint performance, not standalone rank alone.

## Standalone evidence and rejection context
| Model | Attention | Accuracy | Macro_F1 | Outcome |
| --- | --- | --- | --- | --- |
| mobilenet_v3_large | none | 0.8602794411177644 | 0.7861111676525011 | weaker standalone than Tiny; complementary value separate |
| efficientnet_b0 | none | 0.863606121091151 | 0.7706600163648087 | weaker standalone than Tiny; complementary value separate |
| efficientnet_b0 | cbam | 0.863606121091151 | 0.7693010875155818 | weaker standalone than Tiny; complementary value separate |
| efficientnet_v2_s | none | 0.8948769128409847 | 0.8231616637229359 | weaker standalone than Tiny; complementary value separate |
| densenet201 | none | 0.8962075848303394 | 0.8075191800514074 | weaker standalone than Tiny; complementary value separate |
| convnext_tiny | none | 0.9174983366600133 | 0.862830691053104 | competitive standalone; ensemble gain not assumed |
| convnext_small | none | 0.9221556886227545 | 0.8549484177398866 | accuracy gain with macro-F1 trade-off |
| resnet101 | none | 0.8809048569527611 | 0.7920318955927467 | weaker standalone than Tiny; complementary value separate |
| efficientnet_b3 |  |  |  | FAILED/UNVERIFIED: ": 7, "logits_inf": 0, "bad_logit_rows": [6], "loss_finite": false, "probabilities_nonfinite": 7}
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "C:\Users\MARS\Desktop\College\Minor Project\Project Development\research\train.py", line 565, in <module>
    if __name__=='__main__': main()
                             ~~~~^^
  File "C:\Users\MARS\Desktop\College\Minor Project\Project Development\research\train.py", line 562, in main
    with run_lock(config['experiment_id']): run(config,args.resume,fix,args.stop_after_epoch)
                                            ~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\MARS\Desktop\College\Minor Project\Project Development\research\train.py", line 467, in run
    all_y,all_p,ids,val_loss,validation_precision=evaluate_validation(model,val_loader,weights,config,experiment.path,epoch,
                                                  ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        allow_fp32_recovery=bool(amendment),training_state=dict(optimizer=optimizer.state_dict(),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(True)))
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\MARS\Desktop\College\Minor Project\Project Development\research\train.py", line 197, in evaluate_validation
    try:return collect(False)
               ~~~~~~~^^^^^^^
  File "C:\Users\MARS\Desktop\College\Minor Project\Project Development\research\train.py", line 189, in collect
    try:num,p=checked_validation_batch(model,images,y,weights,fp32)
              ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\MARS\Desktop\College\Minor Project\Project Development\research\train.py", line 163, in checked_validation_batch
    raise NonfiniteValidation(dict(reason='nonfinite_forward_or_loss',precision='fp32' if fp32 else 'amp_fp16',
    ...<2 lines>...
        loss_finite=bool(torch.isfinite(num)),probabilities_nonfinite=int((~torch.isfinite(probabilities)).sum())))
NonfiniteValidation: Nonfinite validation batch: {"reason": "nonfinite_forward_or_loss", "precision": "amp_fp16", "logits_nan": 7, "logits_inf": 0, "bad_logit_rows": [6], "loss_finite": false, "probabilities_nonfinite": 7}
 |

No candidate passed the material balanced-gain gate; retain the proven S12 FP32 identity reference rather than chase a few images.

## Rejected models and alternatives
| experiment_id | accuracy | macro_f1 | selected |
| --- | --- | --- | --- |
| s06_convnext_tiny_none_exploratory_seed42 | 0.9174983366600133 | 0.862830691053104 | False |
| s18_convnext_small_none_exploratory_seed42 | 0.9221556886227545 | 0.8549484177398866 | False |
| s12_s03_s06_s10_equal_probability_exploratory_seed42 | 0.9241516966067864 | 0.877023379276669 | False |
| s13_s12_identity_fp32_control_exploratory_seed42 | 0.9248170326014638 | 0.8774541331925992 | True |
| s17_s06_s15_equal_probability_exploratory_seed42 | 0.9261477045908184 | 0.8770606134548798 | False |
| s19_resnet101_none_exploratory_seed42 | 0.8809048569527611 | 0.7920318955927467 | False |
| s20_final_equal_01_exploratory_seed42 | 0.9261477045908184 | 0.8725039609097829 | False |
| s21_final_equal_02_exploratory_seed42 | 0.9194943446440452 | 0.865239148650957 | False |
| s22_final_equal_03_exploratory_seed42 | 0.9228210246174318 | 0.8478816281164497 | False |
| s23_final_equal_04_exploratory_seed42 | 0.9254823685961411 | 0.8641600811694099 | False |
| s26_final_weighted_01_exploratory_seed42 | 0.9268130405854956 | 0.8718470001691984 | False |
| s27_final_weighted_02_exploratory_seed42 | 0.927478376580173 | 0.8663550330406329 | False |

Unselected higher/lower accuracy methods are rejected as the final choice because they fail the material/balance/complexity gate; retain all research evidence. MobileNet/B0 controls and CBAM ablation remain valid; standalone DenseNet was weaker. Candidate failures are disclosed above.

## Class-wise performance
| class_name | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| akiec | 0.875 | 0.7142857142857143 | 0.7865168539325843 | 49 |
| bcc | 0.8390804597701149 | 0.948051948051948 | 0.8902439024390244 | 77 |
| bkl | 0.910958904109589 | 0.806060606060606 | 0.8553054662379421 | 165 |
| df | 1.0 | 0.8235294117647058 | 0.9032258064516129 | 17 |
| mel | 0.8407643312101911 | 0.7904191616766467 | 0.8148148148148148 | 167 |
| nv | 0.9479267116682739 | 0.9761668321747765 | 0.961839530332681 | 1007 |
| vasc | 0.9090909090909091 | 0.9523809523809523 | 0.9302325581395349 | 21 |

## Limitations
7009train/1503val/1503lockedtest. Train/val share563lesions affecting596validation images. Many prior development decisions increase selection bias. Bounded weighting is still validation-based method selection, not independent evidence. No statistical superiority or robustness claim from one seed. FP32 identity reference reused; new fusions use stored AMP parent probabilities. Test untouched. Strict confirmation must start fresh from external ImageNet weights; exploratory checkpoints cannot confirm lesion independence. No new architecture follows this report.
