#!/usr/bin/env python3
"""Check PCN (Car) ply files against PCN.json; optionally write a filtered index."""
import argparse
import json
import os
import sys

import yaml

CAR_TAXONOMY = '02958343'


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def paths_for_sample(cfg, subset, model_id, n_renderings):
    partial_tpl = (
        cfg.get('PARTIAL_POINTS_PATH_MULTI')
        or cfg['PARTIAL_POINTS_PATH']
    )
    complete_tpl = cfg['COMPLETE_POINTS_PATH']
    if subset == 'train' and n_renderings > 1:
        partials = [
            partial_tpl % (subset, CAR_TAXONOMY, model_id, i)
            for i in range(n_renderings)
        ]
    else:
        partials = [cfg['PARTIAL_POINTS_PATH'] % (subset, CAR_TAXONOMY, model_id)]
    complete = complete_tpl % (subset, CAR_TAXONOMY, model_id)
    return partials, complete


def ply_header_vertex_count(path):
    count = None
    with open(path, 'rb') as f:
        for _ in range(64):
            line = f.readline()
            if not line:
                break
            text = line.decode('ascii', errors='ignore').strip()
            if text.startswith('element vertex'):
                count = int(text.split()[-1])
            if text == 'end_header':
                break
    return count


def read_ply_points(path):
    import open3d as o3d
    import numpy as np

    expected = ply_header_vertex_count(path)
    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points, dtype=np.float32)
    if pts.size == 0 or pts.shape[0] == 0:
        raise ValueError('empty point cloud')
    if pts.ndim == 1:
        pts = pts.reshape(-1, 3)
    if expected is not None and pts.shape[0] != expected:
        raise ValueError('read %d points, header says %d (truncated/corrupt ply)' % (
            pts.shape[0], expected))
    return pts


def validate_file(path, n_points, role, strict_complete_npoints=True):
    if not os.path.isfile(path):
        return 'missing', 'file not found'
    try:
        pts = read_ply_points(path)
    except Exception as exc:
        return 'corrupt', str(exc)
    if role == 'complete' and strict_complete_npoints and pts.shape[0] != n_points:
        return 'bad_npoints', 'got %d, expect %d' % (pts.shape[0], n_points)
    if role == 'partial' and pts.shape[0] < 256:
        return 'bad_npoints', 'partial too few points: %d' % pts.shape[0]
    return None, ''


def main():
    parser = argparse.ArgumentParser(description='Verify PCN Car ply dataset on disk.')
    parser.add_argument(
        '--config',
        default='cfgs/dataset_configs/PCNCars.yaml',
        help='Dataset config yaml',
    )
    parser.add_argument(
        '--out-json',
        default='',
        help='If set, write filtered PCN.json keeping only valid samples',
    )
    parser.add_argument(
        '--out-report',
        default='',
        help='If set, write full problem report to this path',
    )
    parser.add_argument(
        '--skip-read',
        action='store_true',
        help='Only check file existence, do not open/read ply (fast but misses corrupt files)',
    )
    parser.add_argument(
        '--skip-npoints',
        action='store_true',
        help='Do not require complete ply to have exactly N_POINTS (not recommended)',
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    json_path = cfg['CATEGORY_FILE_PATH']
    n_renderings = cfg.get('N_RENDERINGS', 8)
    n_points = cfg.get('N_POINTS', 16384)

    with open(json_path, 'r', encoding='utf-8') as f:
        categories = json.load(f)

    car = next(c for c in categories if c['taxonomy_id'] == CAR_TAXONOMY)
    problems = []
    ok_train, ok_test = [], []

    for subset, n_rend in [('train', n_renderings), ('test', 1)]:
        for model_id in car[subset]:
            partials, complete = paths_for_sample(cfg, subset, model_id, n_rend)
            bad = []
            for p in partials:
                if args.skip_read:
                    if not os.path.isfile(p):
                        bad.append({'path': p, 'kind': 'missing', 'detail': 'file not found'})
                else:
                    kind, detail = validate_file(p, n_points, 'partial')
                    if kind:
                        bad.append({'path': p, 'kind': kind, 'detail': detail})
            if args.skip_read:
                if not os.path.isfile(complete):
                    bad.append({'path': complete, 'kind': 'missing', 'detail': 'file not found'})
            else:
                kind, detail = validate_file(
                    complete, n_points, 'complete', strict_complete_npoints=not args.skip_npoints)
                if kind:
                    bad.append({'path': complete, 'kind': kind, 'detail': detail})

            if bad:
                problems.append({'subset': subset, 'model_id': model_id, 'bad': bad})
            elif subset == 'train':
                ok_train.append(model_id)
            else:
                ok_test.append(model_id)

    total_train = len(car['train'])
    total_test = len(car['test'])
    n_missing = sum(1 for x in problems for b in x['bad'] if b['kind'] == 'missing')
    n_corrupt = sum(1 for x in problems for b in x['bad'] if b['kind'] == 'corrupt')
    n_bad_n = sum(1 for x in problems for b in x['bad'] if b['kind'] == 'bad_npoints')

    print('=== PCN Car dataset check ===')
    print('Config:', args.config)
    print('Index:', json_path)
    print('Read ply:', not args.skip_read)
    print('Require complete N_POINTS=%d:' % n_points, not args.skip_npoints)
    print()
    print('train: %d / %d OK' % (len(ok_train), total_train))
    print('test:  %d / %d OK' % (len(ok_test), total_test))
    print('problem samples: %d' % len(problems))
    print('  missing files: %d' % n_missing)
    print('  corrupt/unreadable: %d' % n_corrupt)
    print('  wrong point count: %d' % n_bad_n)

    if problems:
        print('\nFirst 10 problem samples:')
        for item in problems[:10]:
            print('  [%s] %s' % (item['subset'], item['model_id']))
            for b in item['bad'][:4]:
                print('    [%s] %s' % (b['kind'], b['path']))
                if b['detail']:
                    print('          %s' % b['detail'])

    if args.out_report:
        lines = []
        lines.append('=== PCN Car dataset problems ===')
        lines.append('train OK: %d / %d' % (len(ok_train), total_train))
        lines.append('test OK:  %d / %d' % (len(ok_test), total_test))
        lines.append('problem samples: %d' % len(problems))
        lines.append('missing files: %d' % n_missing)
        lines.append('corrupt/unreadable: %d' % n_corrupt)
        lines.append('')
        for item in problems:
            lines.append('[%s] %s' % (item['subset'], item['model_id']))
            for b in item['bad']:
                lines.append('  [%s] %s' % (b['kind'], b['path']))
                if b['detail']:
                    lines.append('       %s' % b['detail'])
            lines.append('')
        out_path = args.out_report
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('\nFull report written to: %s' % out_path)

    if args.out_json:
        filtered = [{
            'taxonomy_id': car['taxonomy_id'],
            'taxonomy_name': car['taxonomy_name'],
            'train': ok_train,
            'test': ok_test,
            'val': car.get('val', []),
        }]
        out_path = args.out_json
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=4)
        print('\nWrote filtered index (%d train, %d test): %s' % (
            len(ok_train), len(ok_test), out_path))
        print('Set CATEGORY_FILE_PATH to this file in PCNCars.yaml.')

    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
