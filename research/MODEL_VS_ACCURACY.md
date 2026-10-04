# Final standalone architecture comparison

Same exploratory validation; seven classes; no test results. Accuracy-selected checkpoints; native pretrained packages differ. Failed candidates disclosed.

| Model | Architecture_family | Pretrained_weights | Transfer_learning | Attention | Input_size | Training_images | Validation_images | Test_status | Protocol | Best_epoch | Accuracy | Macro_precision | Macro_recall | Macro_F1 | Parameters | Runtime_seconds | Training_epochs | Outcome | experiment_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mobilenet_v3_large | mobile CNN | MobileNet_V3_Large_Weights.IMAGENET1K_V2 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 16 | 0.8602794411177644 | 0.7949701703544128 | 0.780099667659236 | 0.7861111676525011 | 2978679 | 896.6182624000357 | 20 | weaker standalone than Tiny; complementary value separate | s02_mobilenet_v3_large_none_exploratory_seed42 |
| efficientnet_b0 | compound-scaled CNN | EfficientNet_B0_Weights.IMAGENET1K_V1 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 20 | 0.863606121091151 | 0.7534286077687018 | 0.7916330829131726 | 0.7706600163648087 | 4016515 | 938.3005253999727 | 20 | weaker standalone than Tiny; complementary value separate | s03_efficientnet_b0_none_exploratory_seed42 |
| efficientnet_b0 | compound-scaled CNN | EfficientNet_B0_Weights.IMAGENET1K_V1 | ImageNet; full fine-tuning | cbam | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 19 | 0.863606121091151 | 0.7599204214006077 | 0.7812601338875816 | 0.7693010875155818 | 4221413 | 743.0220448999899 | 20 | weaker standalone than Tiny; complementary value separate | s05_efficientnet_b0_cbam_exploratory_seed42 |
| efficientnet_v2_s | fused/MBConv CNN | EfficientNet_V2_S_Weights.IMAGENET1K_V1 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 15 | 0.8948769128409847 | 0.8429931782712218 | 0.8099490560479351 | 0.8231616637229359 | 20186455 | 1713.7837136001326 | 20 | weaker standalone than Tiny; complementary value separate | s10_efficientnet_v2_s_none_exploratory_seed42 |
| densenet201 | dense CNN | DenseNet201_Weights.IMAGENET1K_V1 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 12 | 0.8962075848303394 | 0.81691264404218 | 0.805484253908005 | 0.8075191800514074 | 18106375 | 1898.7574629000155 | 17 | weaker standalone than Tiny; complementary value separate | s15_densenet201_none_exploratory_seed42 |
| convnext_tiny | modern CNN | ConvNeXt_Tiny_Weights.IMAGENET1K_V1 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 17 | 0.9174983366600133 | 0.8804352604512183 | 0.8533257129249898 | 0.862830691053104 | 27825511 | 896.4832069000695 | 20 | competitive standalone; ensemble gain not assumed | s06_convnext_tiny_none_exploratory_seed42 |
| convnext_small | modern CNN | ConvNeXt_Small_Weights.IMAGENET1K_V1 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 17 | 0.9221556886227545 | 0.8793870459091774 | 0.8371484433006663 | 0.8549484177398866 | 49460071 | 1470.2621479999507 | 20 | accuracy gain with macro-F1 trade-off | s18_convnext_small_none_exploratory_seed42 |
| resnet101 | residual CNN | ResNet101_Weights.IMAGENET1K_V2 | ImageNet; full fine-tuning | none | 224 | 7009 | 1503 | 1503 locked images; NOT evaluated | exploratory image-level; shared lesions | 19 | 0.8809048569527611 | 0.8055128671958757 | 0.7831234731491711 | 0.7920318955927467 | 42514503 | 1268.5103160999984 | 20 | weaker standalone than Tiny; complementary value separate | s19_resnet101_none_exploratory_seed42 |
| efficientnet_b3 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | FAILED/UNVERIFIED: ": 7, "logits_inf": 0, "bad_logit_rows": [6], "loss_finite": false, "probabilities_nonfinite": 7}
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
 | s16_efficientnet_b3_none_exploratory_seed42 |
