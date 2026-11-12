import csv
import json
import math
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile
from collections import Counter

import selfies as sf
import torch
from rdkit import Chem

from mvselfies.evaluate import cap_brics_fragment_smiles, compute_rare_recall, jsd_full
from mvselfies.experiments import training_command
from mvselfies.restore_resources import restore_bundle
from mvselfies.tokenizer import SelfiesTokenizer
from mvselfies.train_multiview import MolWithFragSetDataset
from mvselfies.train_multiview import FragEncoder, info_nce
from mvselfies import evaluate


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.smiles = ['CCO','CCN','CCCO','CCCN','c1ccccc1','c1ccncc1','CC(=O)O','CC(=O)N']
        self.selfies = [sf.encoder(s) for s in self.smiles]
        (self.root/'smiles.txt').write_text('\n'.join(self.smiles)+'\n', encoding='utf-8')
        (self.root/'selfies.txt').write_text('\n'.join(self.selfies)+'\n', encoding='utf-8')
        tokens = sorted({token for text in self.selfies for token in sf.split_selfies(text)})
        self.tokens = ['<pad>','<bos>','<eos>','<unk>']+tokens
        (self.root/'vocab.txt').write_text('\n'.join(self.tokens)+'\n', encoding='utf-8')
        fragments = [{'selfies':s,'frag_selfies':['[C][C]','[N]']} for s in self.selfies]
        (self.root/'frags.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in fragments), encoding='utf-8')
        (self.root/'frag_vocab.txt').write_text('[C][C]\n[N]\n', encoding='utf-8')
        (self.root/'frag_freq.csv').write_text('fragment,count\n[C][C],8\n[N],2\n', encoding='utf-8')
        with (self.root/'labels.pkl').open('wb') as stream:
            pickle.dump([[0,1] for _ in fragments], stream)

    def cli(self, module, *args, expected_code=0, cwd=None):
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
        result = subprocess.run([sys.executable,'-m','mvselfies.'+module,*map(str,args)],
                                capture_output=True, text=True, encoding='utf-8', env=env,
                                cwd=cwd, timeout=60)
        self.assertEqual(result.returncode, expected_code, result.stdout+'\n'+result.stderr)
        return result

    def test_tokenizer_and_fragment_pairing_guard(self):
        tok = SelfiesTokenizer.from_tokens(self.tokens)
        ids_in, ids_out, mask = tok.encode('[C][C][O]', 8)
        self.assertEqual(tok.decode_ids(ids_out), '[C][C][O]')
        self.assertEqual(ids_in[0], tok.bos_id)
        self.assertEqual(sum(not p for p in mask), 4)
        with self.assertRaises(ValueError):
            SelfiesTokenizer.from_tokens(self.tokens+['[C]'])
        rows = (self.root/'frags.jsonl').read_text(encoding='utf-8').splitlines()
        (self.root/'frags.jsonl').write_text('\n'.join(reversed(rows))+'\n', encoding='utf-8')
        args = (self.root/'selfies.txt', self.root/'frags.jsonl', tok, 12)
        with self.assertRaisesRegex(ValueError, 'Fragment row 1'):
            MolWithFragSetDataset(*args)
        with self.assertWarns(UserWarning):
            ds = MolWithFragSetDataset(*args, legacy_fragment_order=True,
                                     frag_labels_pkl=self.root/'labels.pkl', frag_vocab_size=2)
        self.assertEqual(ds.labels[0], [0,1])
        self.assertTrue(torch.equal(ds[0][-1], torch.ones(2)))

    def test_metric_definitions_and_fragment_conventions(self):
        self.assertEqual(cap_brics_fragment_smiles('[16*]c1ccccc1', 'capped'), 'c1ccccc1')
        self.assertIn('*', cap_brics_fragment_smiles('[16*]c1ccccc1', 'labelled'))
        self.assertAlmostEqual(jsd_full(Counter(a=1), Counter(b=1)), math.sqrt(math.log(2)), places=9)
        self.assertEqual(compute_rare_recall(Counter(a=100,b=1,c=1), Counter(b=4)), 0.5)
        for left, right in [(Counter(), Counter()), (Counter(), Counter(a=1)),
                            (Counter(a=1), Counter()), (Counter(a=0), Counter(b=1))]:
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(ValueError, 'positive counts'):
                    jsd_full(left, right)

    def test_training_rejects_a_split_smaller_than_a_full_batch(self):
        common = ['--data',self.root/'selfies.txt','--vocab',self.root/'vocab.txt',
                  '--max-len','12','--epochs','1','--num-workers','0']
        for module in ['train_baseline', 'train_multiview']:
            with self.subTest(module=module):
                destination = self.root/module
                options = common+['--outdir',destination]
                if module == 'train_multiview':
                    options += ['--frags-jsonl',self.root/'frags.jsonl']
                result = self.cli(module,*options,expected_code=1)
                self.assertIn('Training split has 7 examples',result.stderr)
                self.assertFalse((destination/'best.pt').exists())

    def test_half_precision_attention_has_finite_empty_rows_and_gradients(self):
        encoder = FragEncoder(vocab_size=5,pad_id=0,emb=4,hid=4,z_dim=4).half()
        fragments = torch.ones((3,2,3),dtype=torch.long)
        padding = torch.zeros_like(fragments,dtype=torch.bool)
        padding[0] = True
        query = torch.randn((3,4),dtype=torch.float16,requires_grad=True)
        encoded = torch.ones((6,4),dtype=torch.float16)
        with mock.patch.object(encoder,'encode_seq',return_value=encoded):
            pooled,attention,valid = encoder(fragments,padding,attn_query=query,
                                            return_attn=True,training=False)
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.isfinite(attention).all())
        self.assertTrue(torch.equal(attention[0],torch.zeros(2,dtype=torch.float16)))
        selected = valid >= 2
        loss = info_nce(query.detach()[selected],pooled[selected])
        loss.backward()
        for name,parameter in encoder.named_parameters():
            if parameter.grad is not None:
                with self.subTest(parameter=name):
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_fragment_worker_errors_and_missing_rows_are_rejected(self):
        messages = queue.Queue()
        runner,connection = mock.Mock(),mock.Mock()
        runner.is_alive.return_value = False
        connection.poll.side_effect = OSError('failed helper pipe')
        with mock.patch.object(evaluate,'_start_runner',return_value=(runner,connection)):
            evaluate._frag_worker_chunk(['CC','CCC'],messages,1,6,1.0,'capped')
        self.assertEqual(messages.get()[0],'ERROR')
        self.assertEqual(messages.get()[0],'DONE')
        for emitted,error in [([('ITEM',['CC']),('DONE',None)],'processed 1 of 8'),
                              ([('ERROR','failed helper')],'Fragment worker failed')]:
            with self.subTest(error=error):
                output = queue.Queue()
                for message in emitted:
                    output.put(message)
                process = mock.Mock()
                process.is_alive.return_value = False
                context = mock.Mock()
                context.Queue.return_value = output
                context.Process.return_value = process
                with mock.patch.object(evaluate.mp,'get_context',return_value=context):
                    with self.assertRaisesRegex(RuntimeError,error):
                        evaluate._bounded_controller_build_counter(
                            str(self.root/'smiles.txt'),min_heavy_atoms=1,max_frags_per_mol=6,
                            n_workers=1,chunk_size=80,queue_mult=2,frag_timeout_sec=1.0,
                            desc='Worker failure test')

    def test_generation_and_filling_reject_input_output_collisions(self):
        checkpoint = self.root/'checkpoint.pt'
        original = b'checkpoint must not be overwritten'
        checkpoint.write_bytes(original)
        result = self.cli('generate','--ckpt',checkpoint,'--out',checkpoint,
                          '--n','1',expected_code=2)
        self.assertIn('Output paths must differ',result.stderr)
        self.assertEqual(checkpoint.read_bytes(),original)
        metrics_checkpoint = self.root/'generated.metrics.json'
        metrics_checkpoint.write_bytes(original)
        self.cli('generate','--ckpt',metrics_checkpoint,'--out',self.root/'generated.smi',
                 '--n','1',expected_code=2)
        self.assertEqual(metrics_checkpoint.read_bytes(),original)
        fragments = self.root/'frags.jsonl'
        before = fragments.read_bytes()
        result = self.cli('fill_fragments','--frags-jsonl',fragments,
                          '--frag-vocab',self.root/'frag_vocab.txt','--out-jsonl',fragments,
                          '--out-labels-pkl',self.root/'labels-out.pkl',expected_code=2)
        self.assertIn('distinct paths',result.stderr)
        self.assertEqual(fragments.read_bytes(),before)
        self.cli('fill_fragments','--frags-jsonl',fragments,'--frag-vocab',self.root/'frag_vocab.txt',
                 '--out-jsonl',self.root/'same.out','--out-labels-pkl',self.root/'same.out',expected_code=2)

    def test_damaged_resource_archive_preserves_an_existing_file(self):
        destination = self.root/'models/main/test/best.pt'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'original checkpoint')
        archive = self.root/'models.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr('models/main/test/best.pt',b'replacement checkpoint')
        data = archive.read_bytes()
        offset = data.index(b'replacement checkpoint')
        archive.write_bytes(data[:offset]+b'X'+data[offset+1:])
        with self.assertRaises(zipfile.BadZipFile):
            restore_bundle(archive,self.root)
        self.assertEqual(destination.read_bytes(),b'original checkpoint')
        self.assertEqual([path.name for path in destination.parent.iterdir()],['best.pt'])

    def test_experiment_uses_recorded_settings_and_separate_output(self):
        run = {'model':'Rare-Aware-VAE','training_args':{'epochs':149,'seed':1,'frags_jsonl':'old/frags.jsonl',
                                            'mlflow':True,'attn_pooling':True}}
        cmd = training_command(run, Path('datasets/10k'), Path('runs/new'))
        self.assertIn('--legacy-fragment-order', cmd)
        self.assertIn('--attn-pooling', cmd)
        self.assertEqual(cmd[cmd.index('--epochs')+1], '149')
        self.assertNotIn('--mlflow', cmd)
        for model, module in [('Baseline-VAE','train_baseline'),
                              ('Contrastive-VAE','train_multiview'),
                              ('Rare-Aware-VAE','train_multiview')]:
            with self.subTest(model=model):
                run['model'] = model
                cmd = training_command(run, Path('datasets/10k'), Path('runs/new'))
                self.assertEqual(cmd[2], 'mvselfies.'+module)
                self.assertEqual('--legacy-fragment-order' in cmd, model != 'Baseline-VAE')
        run['model'] = 'unknown-model'
        with self.assertRaises(KeyError):
            training_command(run, Path('datasets/10k'), Path('runs/new'))

    def test_resource_restore_preserves_metadata_and_rejects_unsafe_paths(self):
        results = self.root/'results'
        results.mkdir()
        for name in ['README.md', 'manifest.json', 'summary.csv']:
            (results/name).write_text('current', encoding='utf-8')
        archive = self.root/'results.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('results/main/run/counts.csv', 'fragment,count\na,1\n')
            bundle.writestr('results/tables/table.csv', 'a,b\n1,2\n')
            bundle.writestr('results/replicates/run/counts.csv', 'replicate')
            for name in ['README.md', 'manifest.json', 'summary.csv']:
                bundle.writestr('results/'+name, 'archived')
        restore_bundle(archive, self.root)
        self.assertEqual((results/'main/run/counts.csv').read_bytes(), b'fragment,count\na,1\n')
        self.assertTrue((results/'tables/table.csv').is_file())
        self.assertEqual((results/'replicates/run/counts.csv').read_text(), 'replicate')
        for name in ['README.md', 'manifest.json', 'summary.csv']:
            self.assertEqual((results/name).read_text(), 'current')
        for name in ['results/../../escaped.txt', 'models/wrong-bundle.txt',
                     'results/..\\escaped.txt', 'results/file:stream']:
            with self.subTest(name=name):
                with zipfile.ZipFile(archive, 'w') as bundle:
                    bundle.writestr(name, 'invalid')
                with self.assertRaises(ValueError):
                    restore_bundle(archive, self.root)

    def test_training_and_generation_for_all_model_variants(self):
        common = ['--data',self.root/'selfies.txt','--vocab',self.root/'vocab.txt',
                  '--max-len','12','--epochs','1','--batch-size','4','--num-workers','0',
                  '--emb','8','--hid','16','--z','4','--enc-layers','1','--dec-layers','1',
                  '--dropout','0']
        for model in ['Baseline-VAE','Contrastive-VAE','Rare-Aware-VAE']:
            with self.subTest(model=model):
                destination = self.root/model
                options = common+['--outdir',destination]
                if model == 'Baseline-VAE':
                    module = 'train_baseline'
                    options += ['--log-file','training.log','--log-interval','0',
                                '--sample-interval','0','--save-interval','0','--sample-n','2']
                else:
                    module = 'train_multiview'
                    options += ['--log-interval','1','--save-interval','1000',
                                '--frags-jsonl',self.root/'frags.jsonl','--max-frag-len','6',
                                '--frag-vocab',self.root/'frag_vocab.txt','--frag-freq',self.root/'frag_freq.csv']
                    if model == 'Rare-Aware-VAE':
                        options += ['--attn-pooling','--rare-aware-dropout','--frag-dropout','0.15',
                                    '--lambda-frag-bow','0.05','--frag-labels-pkl',self.root/'labels.pkl']
                self.cli(module, *options, cwd=self.root)
                checkpoint = torch.load(destination/'best.pt',map_location='cpu',weights_only=True)
                self.assertGreater(checkpoint['step'],0)
                if model == 'Baseline-VAE':
                    self.assertTrue((self.root/'training.log').is_file())
                self.cli('generate','--ckpt',destination/'best.pt','--out',destination/'generated.smi',
                         '--n','5','--batch-size','2','--seed','7')
                metrics = json.loads((destination/'generated.metrics.json').read_text())
                self.assertEqual(metrics['n_total'],5)
                self.assertEqual(metrics['batch_size'],2)
                self.assertLessEqual(metrics['n_valid'],5)

    def test_preprocessing_preserves_order_and_evaluation_matches_reference(self):
        self.cli('sanity','--input-txt',self.root/'smiles.txt',
                 '--outdir',self.root/'sanity','--stereo','--salt-split')
        self.assertEqual(len((self.root/'sanity/clean_selfies.txt').read_text().splitlines()), len(self.smiles))
        # Three fragment chunks exercise ordering across concurrent workers.
        preparation_smiles = self.smiles+['C'*n+'O' for n in range(5,165)]
        (self.root/'preparation.smi').write_text('\n'.join(preparation_smiles)+'\n', encoding='utf-8')
        self.cli('extract_fragments','--train-smiles',self.root/'preparation.smi',
                 '--out-jsonl',self.root/'extracted.jsonl','--out-vocab',self.root/'extracted_vocab.txt',
                 '--max-workers','2','--min-heavy-atoms','1','--batch-size','80')
        rows = [json.loads(s) for s in (self.root/'extracted.jsonl').read_text().splitlines()]
        self.assertEqual([d['smiles'] for d in rows], [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in preparation_smiles])
        self.cli('fill_fragments','--frags-jsonl',self.root/'extracted.jsonl',
                 '--frag-vocab',self.root/'extracted_vocab.txt','--out-jsonl','filled.jsonl',
                 '--out-labels-pkl','filled.pkl',cwd=self.root)
        filled = (self.root/'filled.jsonl').read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(filled),len(preparation_smiles))
        self.cli('evaluate','--gen',self.root/'smiles.txt','--train',self.root/'smiles.txt',
                 '--out-dir',self.root/'eval','--workers','1','--min-heavy-atoms','1')
        report = json.loads((self.root/'eval/frag_eval_report.json').read_text())
        self.assertAlmostEqual(report['jsd_full_vocab'],0.0)
        self.assertEqual(report['rare_fragment_recall'],1.0)
        self.cli('analyze','--gen',self.root/'smiles.txt','--out-csv',self.root/'props.csv')
        with (self.root/'props.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), len(self.smiles))


if __name__ == '__main__':
    unittest.main()
