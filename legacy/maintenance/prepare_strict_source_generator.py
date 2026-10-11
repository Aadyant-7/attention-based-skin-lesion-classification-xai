from pathlib import Path
import json
root=Path.cwd()
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)

# Preserve the historical runner byte-for-byte; isolate the new recipe/runner.
s=(root/'research/train.py').read_text()
s=replace(s,'from .experiment import Experiment','from .strict_experiment import Experiment')
start=s.index('RECIPES=');end=s.index('\n\n\ndef validate_config',start)
s=s[:start]+"RECIPES={'final_strict_v1':ROOT/'research/configs/final_strict_recipe_v1.json'}"+s[end:]
start=s.index('def development_data(');end=s.index('\n\nclass DevelopmentImages',start)
s=s[:start]+'from .strict_protocol import development_data, partition_metadata\n'+s[end:]
s=replace(s,"return {f:sha256(ROOT/f) for f in ('research/train.py','research/models.py','research/experiment.py',", "return {f:sha256(ROOT/f) for f in ('research/strict_train.py','research/strict_protocol.py','research/models.py','research/strict_experiment.py',")
s=replace(s,"obj.validation_support=pd.read_csv(ROOT/config['split_manifest']).query(\"split=='val'\").diagnosis.value_counts().to_dict()", "obj.validation_support=partition_metadata()[0].query(\"split=='val'\").diagnosis.value_counts().to_dict()")
s=replace(s,'try:return collect(False)', 'try:return collect(True)')
s=replace(s,"if stale>=config['patience']: break", "if stale>=config['patience'] and epoch-1>=config['minimum_epochs']: break")
s=replace(s,'scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()', "scaler.unscale_(optimizer)\n                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)\n                scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()")
s=replace(s,"logger.warning('Validation epoch %d recovered with a complete FP32 pass; diagnostic artifacts saved',epoch)", "logger.info('Validation epoch %d used predefined FP32 validation',epoch)")
s=replace(s,"metadata_sha256=sha256(ROOT/'data/raw/HAM10000/HAM10000_metadata.csv'),", "metadata_scope='train/validation metadata only; test labels excluded',fresh_imagenet_initialization=True,exploratory_checkpoint_reused=False,")
s=replace(s,"if best is None: raise RuntimeError('No committed epoch; cannot publish a completed result')", """if best is None: raise RuntimeError('No committed epoch; cannot publish a completed result')
        final_y,final_p,final_ids,final_loss,_=evaluate_validation(model,val_loader,weights,config,experiment.path,len(experiment.history))
        final_metrics=metric_report(np.asarray(final_y),final_p,final_loss)
        final_predictions=[dict(image_id=i,true_class=CLASSES[y],predicted_class=CLASSES[int(np.argmax(p))],
            **{f'p_{c}':float(p[j]) for j,c in enumerate(CLASSES)}) for i,y,p in zip(final_ids,final_y,final_p)]
        write_json(experiment.path/'validation_metrics_latest.json',final_metrics)
        write_csv(experiment.path/'validation_predictions_latest.csv',final_predictions)
        from .plots import metric_figures
        metric_figures(final_metrics,experiment.path/'figures_latest','Final strict validation | latest checkpoint')
        if secondary_best:
            metric_figures(secondary_best['metrics'],experiment.path/'figures_macro_f1','Final strict validation | macro-F1 checkpoint')
        write_csv(experiment.path/'lr_history.csv',[{k:r[k] for k in ('epoch','backbone_lr','head_lr')} for r in experiment.history])
""")
s=replace(s,"secondary_best_epoch=secondary_best['epoch'] if secondary_best else None,\n                   stop_reason=", "secondary_best_epoch=secondary_best['epoch'] if secondary_best else None,\n                   stopping_epoch=len(experiment.history),runtime_seconds=runtime+time.perf_counter()-start,maximum_epochs=config['max_epochs'],minimum_epochs=config['minimum_epochs'],\n                   fp32_validation_fallback_used=False,validation_precision='fp32_from_start',gradient_clip_norm=config['gradient_clip_norm'],\n                   stop_reason=")
s=replace(s,"experiment.fail(repr(exc));progress(status='failed',error=repr(exc),committed_epochs=len(experiment.history))", """stamp=f'training_failure_{time.time_ns()}'
        write_json(experiment.path/(stamp+'.json'),dict(error=repr(exc),committed_epochs=len(experiment.history),
            model_nonfinite_tensors=tensor_nonfinite_names(model.state_dict()) if 'model' in locals() else [],
            optimizer_nonfinite_tensors=tensor_nonfinite_names(optimizer.state_dict()) if 'optimizer' in locals() else []))
        if 'model' in locals():
            atomic_checkpoint(experiment.path/(stamp+'.pt'),dict(diagnostic_only=True,model=model.state_dict(),
                optimizer=optimizer.state_dict() if 'optimizer' in locals() else {},config=config,code_hashes=code_hashes()))
        experiment.fail(repr(exc));progress(status='failed',error=repr(exc),committed_epochs=len(experiment.history))""")
s=s.replace('ready_pending_gpu_approval','ready_authorized_final_strict')
(root/'research/strict_train.py').write_text(s,encoding='utf-8')
s=(root/'research/experiment.py').read_text()
s=replace(s,'from .registry import FIELDS, upsert','from .registry import FIELDS, upsert\nfrom .strict_protocol import partition_metadata')
s=replace(s,"self.validation_support=pd.read_csv(split).query(\"split == 'val'\").diagnosis.value_counts().to_dict()", "self.validation_support=partition_metadata()[0].query(\"split == 'val'\").diagnosis.value_counts().to_dict()")
(root/'research/strict_experiment.py').write_text(s,encoding='utf-8')
base=json.loads((root/'research/configs/phase3/s03_efficientnet_b0_none_exploratory_seed42.json').read_text())
for k in ['experiment_id','model','weights','optional','question']:base.pop(k)
base.update(recipe_version='final_strict_v1',phase='final_strict_confirmation',protocol='strict_lesion_disjoint',
    split_manifest='data/splits/split_assignments.csv',split_sha256='db1ce9f8b83f176001dd89fd332fb1668c7d2ba5f58d77157d293a09e2c2e696',
    max_epochs=50,patience=12,minimum_epochs=15,gradient_clip_norm=1.0,validation_precision='fp32',status='authorized',
    checkpoint_rule='earliest maximum standalone validation accuracy; ensemble of three accuracy winners, no ensemble epoch search')
(root/'research/configs/final_strict_recipe_v1.json').write_text(json.dumps(base,indent=2)+'\n')
spec=[('s28','efficientnet_b0','EfficientNet_B0_Weights.IMAGENET1K_V1'),('s29','convnext_tiny','ConvNeXt_Tiny_Weights.IMAGENET1K_V1'),('s30','efficientnet_v2_s','EfficientNet_V2_S_Weights.IMAGENET1K_V1')]
folder=root/'research/configs/final_strict';folder.mkdir(exist_ok=True)
configs=[]
for number,model,weights in spec:
    c=dict(base,experiment_id=f'{number}_{model}_final_strict_seed42',model=model,weights=weights,
        question=f'Confirm frozen {model} generalization with fresh ImageNet training on lesion-disjoint development; no architecture search.',optional=False)
    path=folder/(c['experiment_id']+'.json');path.write_text(json.dumps(c,indent=2)+'\n');configs.append(path.relative_to(root).as_posix())
(root/'research/configs/final_strict_queue_v1.json').write_text(json.dumps(dict(configs=configs,execution_mode='sequential',ensemble_id='s31_final_equal_fp32_strict_seed42',ensemble_weights=[1/3]*3,test_evaluation=False),indent=2)+'\n')
