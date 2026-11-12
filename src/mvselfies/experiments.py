"""Run one archived experiment configuration without overwriting released artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


TRAINERS = {
    'Baseline-VAE': 'train_baseline',
    'Contrastive-VAE': 'train_multiview',
    'Rare-Aware-VAE': 'train_multiview',
}


def training_command(run, dataset, destination):
    """Translate a recorded configuration into a training subprocess argument list.

    Parameters
    ----------
    run : dict
        Catalog entry with model identifier and recorded ``training_args``.
    dataset : pathlib.Path
        Scale-specific folder containing molecule and fragment input files.
    destination : pathlib.Path
        Output directory for this new training run.

    Returns
    -------
    list of str
        Interpreter, module, and CLI arguments. Input/output paths are remapped,
        MLflow is disabled, and multiview runs explicitly enable historical
        positional pairing to match the archived experiment inputs.

    Raises
    ------
    KeyError
        If the model name is not in the trainer mapping.
    """
    args = dict(run['training_args'])
    module = TRAINERS[run['model']]
    args.update(data=str(dataset/'clean_selfies.txt'), vocab=str(dataset/'selfies_vocab.txt'),
                outdir=str(destination), log_file=str(destination/'train.log'), mlflow=False)
    for key, filename in [('frags_jsonl','frags_filled.jsonl'), ('frag_vocab','frag_vocab.txt'),
                          ('frag_freq','frag_freq.csv'), ('frag_labels_pkl','frag_labels.pkl')]:
        if args.get(key):
            args[key] = str(dataset/filename)
    if module == 'train_multiview':
        args['legacy_fragment_order'] = True
    cmd = [sys.executable, '-m', 'mvselfies.'+module]
    for key, value in args.items():
        if key.startswith('mlflow') or key in {'run_name', 'log'} or value is None or value is False:
            continue
        cmd.append('--'+key.replace('_','-'))
        if value is not True:
            cmd.append(str(value))
    return cmd


def main():
    """List, preview, or execute catalog stages in a separate run-output directory."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, default=Path('experiments/runs.json'))
    parser.add_argument('--run', help='For example: main/Baseline-VAE_10k_s1')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--stage', choices=['train','generate','analyze','evaluate','all'], default='all')
    parser.add_argument('--datasets', type=Path, default=Path('datasets'))
    parser.add_argument('--runs-dir', type=Path, default=Path('runs'))
    parser.add_argument('--n', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    options = parser.parse_args()
    runs = json.loads(options.config.read_text(encoding='utf-8'))
    if options.list:
        for run in runs:
            args = run['training_args']
            print(f"{run['id']:44} epochs={args['epochs']:3} batch={args['batch_size']:3} fragments={run['fragment_convention']}")
        return
    matches = [run for run in runs if run['id'] == options.run]
    if len(matches) != 1:
        parser.error('Select an exact --run from --list.')
    run = matches[0]
    dataset = options.datasets/run['scale']
    destination = options.runs_dir/run['id']
    if any(destination.resolve().is_relative_to(Path(name).resolve())
           for name in ['results', 'models', 'datasets']):
        parser.error('Choose an output directory outside the released results and models.')
    if options.stage in {'train','all'} and (destination/'best.pt').exists():
        parser.error('A trained checkpoint already exists here; choose another --runs-dir.')
    generated = destination/'generated_canonical_smiles.txt'
    module = [sys.executable, '-m']
    commands = {
        'train': training_command(run, dataset, destination),
        'generate': module+['mvselfies.generate','--ckpt',str(destination/'best.pt'),
                           '--out',str(generated),'--n',str(options.n),
                           '--train-smiles',str(dataset/'clean_smiles.txt'),'--seed',str(run['seed'])],
        'analyze': module+['mvselfies.analyze','--gen',str(generated),
                          '--out-csv',str(destination/'generated_props.csv'),'--seed',str(run['seed'])],
        'evaluate': module+['mvselfies.evaluate','--gen',str(generated),'--train',str(dataset/'clean_smiles.txt'),
                           '--out-dir',str(destination/'frag_eval_smiles'),'--space','smiles',
                           '--fragment-convention',run['fragment_convention'],'--workers',str(options.workers)],
    }
    stages = list(commands) if options.stage == 'all' else [options.stage]
    for stage in stages:
        cmd = commands[stage]
        print(subprocess.list2cmdline(cmd), flush=True)
        if not options.dry_run:
            destination.mkdir(parents=True, exist_ok=True)
            (destination/'release_run.json').write_text(json.dumps(run, indent=2)+'\n', encoding='utf-8')
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
