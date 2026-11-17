"""Fixed zero-shot CPMS evaluation without assuming undocumented label scales.

No training or threshold fitting. Source-image staff detections are frozen as
proxy traces before running correction methods. Their coverage and straightness
are diagnostics, not manually annotated geometric accuracy or OMR accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scoredewarp.core.pipeline import Config, _detect, dewarp, dewarp_document, refine_dewarp
from scoredewarp.core.structures import draw_structures
from scoredewarp.eval.geometry import map_points, network_points
from scoredewarp.eval.real_score import trace_metrics, sampled_map_folds, margin_content_points
from scoredewarp.learn.infer import load, predict, apply_kind_decision


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def quantile(records, section, key, q=.5):
    values = [r.get(section, {}).get(key) if section else r.get(key) for r in records]
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.quantile(values, q)) if values else None


def summarize(records, methods):
    result = {}
    for method in methods:
        rows = [r for r in records if r['method'] == method]
        passed = [r for r in rows if r['status'] == 'ok']
        result[method] = {'pages':len(rows), 'exceptions':len(rows)-len(passed),
                          'type_staff_rate': sum(r.get('page_type')=='staff' for r in passed)/len(rows),
                          'detected_staves_median':quantile(passed, None, 'detected_staves'),
                          'time_s_median':quantile(passed, None, 'time_s')}
        for section in ('own', 'common'):
            result[method][section] = {key:quantile(passed, section, key) for key in
                ('point_coverage','usable_staves','staff_straight','staff_spacing_cv','horizontal_reversal_rate')}
            result[method][section]['straightness_p90'] = quantile(passed, section, 'staff_straight', .9)
            result[method][section]['measurable_pages'] = sum(r.get(section, {}).get('staff_straight') is not None for r in passed)
        result[method]['coverage_below_95pct_pages'] = sum(r.get('own',{}).get('point_coverage',0) < .95 for r in rows)
        margin = [r['margin_content_coverage'] for r in passed if r.get('margin_content_coverage') is not None]
        result[method]['margin_content'] = {
            'measurable_pages': len(margin),
            'coverage_median': float(np.median(margin)) if margin else None,
            'coverage_mean': float(np.mean(margin)) if margin else None,
            'pages_below_95pct': sum(v < .95 for v in margin)}
    return result


def paired_comparison(records, methods, baseline='input'):
    lookup = {(r['sample'],r['method']):r for r in records}
    result = {}
    for method in methods:
        if method in ('input', baseline): continue
        pairs = []
        for sample in sorted({r['sample'] for r in records}):
            a, b = lookup[sample, method], lookup[sample, baseline]
            x,y = a.get('common',{}).get('staff_straight'),b.get('common',{}).get('staff_straight')
            if x is not None and y is not None:
                pairs.append({'sample':sample, 'group':a['content_group'], 'delta':x-y})
        groups = sorted({p['group'] for p in pairs})
        # Each content group (two captures) receives equal weight.
        differences = [np.mean([p['delta'] for p in pairs if p['group']==g]) for g in groups]
        if differences:
            rng = np.random.default_rng(731)
            boot = np.mean(rng.choice(differences,(1000,len(differences)),replace=True),1)
            result[method] = {'paired_pages':len(pairs),'content_groups':len(groups),
                              'improved_pages':sum(p['delta']<0 for p in pairs),
                              'page_delta_median':float(np.median([p['delta'] for p in pairs])),
                              'group_mean_delta':float(np.mean(differences)),
                              'group_bootstrap_mean_delta_ci95':np.quantile(boot,[.025,.975]).tolist()}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download-root',type=Path,default=Path('data/external/cpms'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--width',type=int,default=1240)
    ap.add_argument('--limit',type=int,default=0,help='Smoke test only; 0 evaluates all photos')
    ap.add_argument('--checkpoint',action='append',help='Optional repeated name=checkpoint overrides the default network')
    ap.add_argument('--document', action='store_true', help='include original ABC + bounded staff mesh')
    ap.add_argument('--uvdoc-root', type=Path, help='include the verified official UVDoc baseline')
    ap.add_argument('--variant',action='append',default=[],
                    help='Repeated name=network:STAGES; B=surface, C=legacy field, M=bounded line mesh (B,C,BC,M,BM)')
    args = ap.parse_args()
    if args.width < 128 or args.limit < 0: ap.error('Invalid width or limit')
    if args.out.exists() and any(args.out.iterdir()): ap.error('Use a fresh empty output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    manifest=json.loads((args.download_root/'download_manifest.json').read_text())
    source=Path(manifest['source'])
    files={r['path']:r for r in manifest['files']}
    paths=sorted((source/'data').glob('*.jpeg'))
    if args.limit:paths=paths[:args.limit]
    if not paths:ap.error('No CPMS photos')
    checkpoints={'network_robust':ROOT/'outputs/diverse_training_20261115/model/best.pt'}
    if args.checkpoint:
        checkpoints={}
        for value in args.checkpoint:
            if '=' not in value:ap.error('Use name=checkpoint')
            name,path=value.split('=',1)
            if not name or name in ('input','traditional','document','uvdoc') or name in checkpoints or not all(c.isalnum() or c=='_' for c in name):
                ap.error('Network names must be distinct alphanumeric identifiers')
            checkpoints[name]=Path(path).resolve()
    variants={}
    for value in args.variant:
        name,_,spec=value.partition('=')
        network,_,stages=spec.partition(':')
        if (not name or name in ('input','traditional','document','uvdoc') or name in checkpoints or name in variants
                or not all(c.isalnum() or c=='_' for c in name)):
            ap.error('Variant names must be new distinct alphanumeric identifiers')
        if (network not in checkpoints and not (network == 'uvdoc' and args.uvdoc_root)) or stages not in ('B','C','BC','M','BM'):
            ap.error('Use name=network:STAGES with a loaded network and STAGES in B,C,BC,M,BM')
        variants[name]={'network':network,'surface':'B' in stages,'field':'C' in stages,'mesh':'M' in stages}
    methods=['input','traditional'] + (['document'] if args.document else []) + list(checkpoints)
    if args.uvdoc_root: methods.append('uvdoc')
    methods.extend(variants)
    inputs=[]
    for path in paths:
        label=source/'label'/(path.stem+'.json')
        for p in (path,label):
            if sha(p)!=files[str(p.relative_to(source))]['sha256']:
                raise ValueError(f'Download changed: {p}')
        annotation=json.loads(label.read_text())
        content=[[(n['pitch'],n['duration']) for n in row] for row in annotation.values()]
        group=hashlib.sha256(json.dumps(content).encode()).hexdigest()
        inputs.append({'sample':path.stem,'photo_sha256':sha(path),'label_sha256':sha(label),
                       'content_group':group,'annotated_rows':len(annotation),
                       'annotated_notes':sum(map(len,annotation.values()))})
    protocol={'dataset_revision':manifest['revision'],'photos':inputs,'methods':methods,'working_width':args.width,
              'checkpoints':{k:{'path':str(v.relative_to(ROOT)),'sha256':sha(v)} for k,v in checkpoints.items()},
              'network_output_size':[args.width,round(args.width*2263/1600)],'network_field_refinement':False,
              'variants':variants,'variant_surface':'classical B fitted on the network output (identity homography), keeping the full network canvas',
              'document':'original-frame ABC -> bounded staff mesh only if coarse and final evidence both classify as staff',
              'experiment_role':'development regression; CPMS already exposed; no independent generalization claim',
              'margin_content':'ink pixels in one median staff gap above the first and below the last frozen source staff; own validity, not GT',
              'reference':'frozen robust staff detector on resized input; pseudo reference, not GT',
              'point_validity':'finite; canvas bounds; roundtrip <=2 working-image pixels; predicted page mask for networks',
              'common_support':'intersection of all successful correction methods; none if any method failed',
              'label_xy_used_for_geometry':False,'reason':'coordinate resize frame undocumented and visually inconsistent with raw photos',
              'training_on_cpms_photos':False,'parameters_fitted_to_cpms_labels':False,
              'cpms_already_used_for_development':True,'bootstrap':'paired content group mean differences, 1000 resamples, seed731',
              'git_head':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),
              'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
                  [Path(__file__),ROOT/'scoredewarp/eval/real_score.py',ROOT/'scoredewarp/core/pipeline.py',
                   ROOT/'scoredewarp/core/structures.py',ROOT/'scoredewarp/core/surface3d.py',
                   ROOT/'scoredewarp/learn/infer.py', ROOT/'scoredewarp/core/line_mesh.py', ROOT/'scoredewarp/learn/uvdoc.py']}}
    if args.uvdoc_root:
        protocol['uvdoc'] = json.loads((args.uvdoc_root/'download_manifest.json').read_text())
    save(args.out/'protocol.json',protocol)
    networks={k:load(v) for k,v in checkpoints.items()}
    if args.uvdoc_root:
        from scoredewarp.learn import uvdoc
        uvdoc_model = uvdoc.load(args.uvdoc_root)
    records=[];reference_audit=[];start_all=time.time()
    for index,(path,info) in enumerate(zip(paths,inputs)):
        image=cv2.imread(str(path))
        if image is None:raise ValueError(f'Cannot read {path}')
        image=cv2.resize(image,(args.width,round(image.shape[0]*args.width/image.shape[1])),interpolation=cv2.INTER_AREA)
        _,ink,reference=_detect(image)
        staves=[np.asarray(st.tuples) for st in reference.staves if len(st.tuples)>=8]
        content=margin_content_points(ink,staves)
        predictions={}
        destination=args.out/path.stem;destination.mkdir()
        cv2.imwrite(str(destination/'input.jpg'),image)
        cv2.imwrite(str(destination/'input_traces.jpg'),draw_structures(image,reference))
        save(destination/'source_traces.json',[s.tolist() for s in staves])
        save(destination/'margin_content_points.json',content.tolist())
        reference_audit.append({**info,'detected_source_staves':len(staves),'source_trace_points':sum(s.shape[0]*5 for s in staves)})
        mapped={'input':staves};masks={'input':[np.ones(s.shape[:2],bool) for s in staves]}
        rows=[{'sample':path.stem,'content_group':info['content_group'],'method':'input','status':'ok',
               'page_type':reference.page_type,'detected_staves':len(reference.staves),'time_s':0.,
               'own':trace_metrics(staves,masks['input']),'margin_content_points':len(content),
               'margin_content_coverage':1. if len(content) else None}]
        # Freeze the reference before corrections. Prediction never sees labels.
        for method in methods[1:]:
            row={'sample':path.stem,'content_group':info['content_group'],'method':method}
            start=time.time()
            try:
                raw=None
                prediction_time=0.
                if method=='traditional':
                    result=dewarp(image,Config(coarse_width=args.width))
                elif method=='document':
                    result=dewarp_document(image,Config(coarse_width=args.width))
                else:
                    variant=variants.get(method,{'network':method,'surface':False,'field':False,'mesh':False})
                    official = variant['network'] == 'uvdoc'
                    if not official: net,size=networks[variant['network']]
                    if variant['network'] not in predictions:
                        t=time.time()
                        if official:
                            bm,_,_=uvdoc.predict(uvdoc_model,image)
                            out=None
                        else:
                            bm,_,out=predict(net,image,size,out_size=tuple(protocol['network_output_size']))
                        predictions[variant['network']]=(bm,out,time.time()-t)
                    bm,raw,prediction_time=predictions[variant['network']]
                    start=time.time()
                    result=refine_dewarp(image,bm,Config(use_field=variant['field'],use_line_mesh=variant['mesh']),surface=variant['surface'])
                    if official:
                        result.stages['initialization']='official_uvdoc'
                        result.stages['uvdoc_revision']=uvdoc.REVISION
                    else:
                        apply_kind_decision(result,net,raw)
                production_time=time.time()-start+prediction_time
                coordinates=[];validity=[]
                for staff in staves:
                    page=None
                    if raw is not None:
                        _,page=network_points(raw,staff,(image.shape[1],image.shape[0]),1.)
                    coords,keep=map_points(result.bmap,staff,page_valid=page)
                    coordinates.append(coords);validity.append(keep)
                mapped[method]=coordinates;masks[method]=validity
                if len(content):
                    page=network_points(raw,content,(image.shape[1],image.shape[0]),1.)[1] if raw is not None else None
                    _,keep=map_points(result.bmap,content,page_valid=page)
                    row.update(margin_content_points=len(content),margin_content_coverage=float(keep.mean()))
                row.update(status='ok',time_s=production_time,total_evaluation_time_s=time.time()-start,
                           page_type=result.page_type,detected_staves=len(result.structures.staves),
                           output_size=list(result.bmap.out_size),own=trace_metrics(coordinates,validity),
                           grid=sampled_map_folds(result.bmap),stages=result.stages)
                cv2.imwrite(str(destination/(method+'.jpg')),result.image,[cv2.IMWRITE_JPEG_QUALITY,95])
                cv2.imwrite(str(destination/(method+'_debug.jpg')),draw_structures(result.image,result.structures))
                np.savez_compressed(destination/(method+'_bmap.npz'),grid=result.bmap.grid,stride=result.bmap.stride,out_size=result.bmap.out_size)
                save(destination/(method+'_mapped_traces.json'),{'points':coordinates,'valid':validity})
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}',time_s=time.time()-start)
            rows.append(row)
        if all(m in masks for m in methods):
            common=[np.logical_and.reduce([masks[m][i] for m in methods]) for i in range(len(staves))]
            for row in rows:row['common']=trace_metrics(mapped[row['method']],common)
        save(destination/'metrics.json',rows)
        records.extend(rows)
        save(args.out/'per_sample.json',records)
        print(f'{index+1}/{len(paths)} {path.stem} source_staves={len(staves)} '+
              ' '.join(f"{r['method']}={r.get('own',{}).get('staff_straight')}" for r in rows)+
              f' elapsed={time.time()-start_all:.0f}s',flush=True)
    for method,path in checkpoints.items():
        if sha(path)!=protocol['checkpoints'][method]['sha256']:raise ValueError('Checkpoint changed during evaluation')
    save(args.out/'reference_audit.json',reference_audit)
    evaluation={'summary':summarize(records,methods),'paired_vs_input':paired_comparison(records,methods),
                'paired_vs_traditional':paired_comparison(records,methods,'traditional'),
                'elapsed_s':time.time()-start_all,'checkpoint_unchanged':True,
                'reference_is_ground_truth':False,'omr_accuracy_measured':False}
    save(args.out/'evaluation.json',evaluation)
    print(json.dumps(clean(evaluation),indent=2),flush=True)


if __name__=='__main__':main()
