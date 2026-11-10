"""Known, deliberately tiny examples. These fixtures are not project results."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.evaluation.io import validate_labels, validate_prediction, validate_split
from src.evaluation.io import sha256, write_json
from src.evaluation.metrics import average_ranks, baselines, evaluate, pearson, top_indices
from src.evaluation.statistics import across_seeds, add_strength_strata, paired_bootstrap, summarize
from src.evaluation.pipeline import score_independently


class MetricsTests(unittest.TestCase):
    def score(self, p, t, mean=None, genes=None, targets=None):
        p, t = np.asarray(p, float), np.asarray(t, float)
        n, g = p.shape
        return evaluate(p, t, np.zeros(g) if mean is None else mean,
                        np.asarray(targets or [f"t{i}" for i in range(n)]),
                        np.asarray(genes or [f"g{i}" for i in range(g)]),
                        np.ones(n, bool), np.zeros(n, bool), "tiny_fixture", 42)

    def test_mse_per_target_then_macro(self):
        rows = self.score([[1, 3], [2, 4]], [[0, 1], [2, 2]])
        self.assertEqual([r['mse'] for r in rows], [2.5, 2])
        self.assertEqual(summarize(pd.DataFrame(rows)).query("subset=='all'").iloc[0].mse, 2.25)

    def test_equal_perturbation_baseline(self):
        mean, zero, constant = baselines([[0, 2], [6, 4]], 3)
        np.testing.assert_array_equal(mean, [3, 3])
        np.testing.assert_array_equal(zero, np.zeros((3, 2)))
        np.testing.assert_array_equal(constant, [[3, 3]]*3)

    def test_known_pearson_and_reverse(self):
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 2, 3]), 1)
        self.assertAlmostEqual(pearson([1, 2, 3], [3, 2, 1]), -1)
        self.assertAlmostEqual(pearson([1, 2, 3], [1, 3, 2]), .5)

    def test_small_scale_correlation_is_not_missing(self):
        self.assertAlmostEqual(pearson([1e-100, 2e-100, 3e-100], [2e-100, 4e-100, 6e-100]), 1)

    def test_constant_pearson_missing(self):
        self.assertTrue(np.isnan(pearson([0, 0, 0], [1, 2, 3])))
        self.assertTrue(np.isnan(pearson([1, 2, 3], [2, 2, 2])))

    def test_spearman_average_ties(self):
        np.testing.assert_equal(average_ranks([5, 2, 2, 1]), [4, 2.5, 2.5, 1])
        rows = self.score([[1, 1, 3]], [[1, 2, 3]])
        self.assertAlmostEqual(rows[0]['spearman'], np.sqrt(3)/2)

    def test_top20_ties_use_gene_id(self):
        genes = np.asarray([f"g{i:02d}" for i in range(25)][::-1])
        index = top_indices(np.ones(25), genes)
        self.assertEqual(genes[index].tolist(), sorted(genes)[:20])

    def test_top20_overlap_direction_known(self):
        genes = np.asarray([f"g{i:02d}" for i in range(25)])
        p, t = np.arange(1, 26, dtype=float), -np.arange(1, 26, dtype=float)
        row = self.score([p], [t], genes=genes.tolist())[0]
        self.assertEqual(row['top20_overlap'], 1)
        self.assertEqual(row['top20_direction'], 0)
        p = np.r_[np.arange(100, 120), np.zeros(5)]
        t = np.r_[np.zeros(5), np.arange(100, 120)]
        self.assertEqual(self.score([p], [t], genes=genes.tolist())[0]['top20_overlap'], .75)

    def test_zero_topk_has_no_direction(self):
        row = self.score([[0, 0, 0]], [[1, 2, 3]])[0]
        self.assertTrue(row['zero_topk_is_no_signal_tie'])
        self.assertTrue(np.isnan(row['top20_direction']))

    def test_residual_correlation_and_mse_invariance(self):
        p, t, mean = np.array([[2, 4, 8.]]), np.array([[3, 5, 7.]]), np.array([1, 2, 3.])
        row = self.score(p, t, mean)[0]
        self.assertAlmostEqual(row['residual_pearson'], pearson(p[0]-mean, t[0]-mean))
        self.assertEqual(np.mean((p-t)**2), np.mean(((p-mean)-(t-mean))**2))
        constant = self.score([mean], t, mean)[0]
        self.assertTrue(np.isnan(constant['residual_pearson']))

    def test_self_coordinate_exclusion(self):
        row = self.score([[9, 1, 2]], [[0, 1, 2]], targets=['t'], genes=['t', 'a', 'b'])[0]
        self.assertEqual(row['mse'], 27)
        self.assertEqual(row['mse_without_self'], 0)

    def test_missing_truth_never_scored_as_zero(self):
        with self.assertRaisesRegex(ValueError, 'ground truth missing'):
            evaluate([[0, 0]], None, [0, 0], np.asarray(['a']), np.asarray(['x', 'y']), [True], [False], 'm', 42)

    def test_nonfinite_and_shape_rejected(self):
        with self.assertRaises(ValueError):
            self.score([[1, np.nan]], [[1, 2]])
        with self.assertRaises(ValueError):
            self.score([[1, 2]], [[1, 2, 3]])


class ContractTests(unittest.TestCase):
    def package(self):
        return dict(target_ids=np.asarray(['a', 'b']), gene_ids=np.asarray(['x', 'y']),
                    delta_pred=np.zeros((2, 2)), feature_available=np.asarray([True, False]),
                    used_fallback=np.asarray([False, True]))

    def test_prediction_contract_success(self):
        validate_prediction(self.package(), ['a', 'b'], ['x', 'y'])

    def test_reordered_targets_and_genes_rejected(self):
        for targets, genes in ((['b', 'a'], ['x', 'y']), (['a', 'b'], ['y', 'x'])):
            with self.assertRaises(ValueError):
                validate_prediction(self.package(), targets, genes)

    def test_duplicate_and_integer_flags_rejected(self):
        p = self.package(); p['gene_ids'] = np.asarray(['x', 'x'])
        with self.assertRaises(ValueError):
            validate_prediction(p, ['a', 'b'], ['x', 'x'])
        p = self.package(); p['feature_available'] = np.asarray([1, 0])
        with self.assertRaises(ValueError):
            validate_prediction(p, ['a', 'b'], ['x', 'y'])

    def test_split_target_and_control_leakage(self):
        spec = dict(targets=dict(train=['a'], val=['b'], test=['c']),
                    control_cell_ids=dict(train=['x'], val=['y'], test=['z']))
        validate_split(spec)
        spec['targets']['test'] = ['a']
        with self.assertRaises(ValueError): validate_split(spec)
        spec['targets']['test'] = ['c']; spec['control_cell_ids']['test'] = ['x']
        with self.assertRaises(ValueError): validate_split(spec)

    def test_training_label_test_contamination(self):
        p = dict(target_ids=np.asarray(['a']), gene_ids=np.asarray(['x']),
                 delta_true=np.array([[1.]]), split=np.asarray(['test']))
        with self.assertRaises(ValueError): validate_labels(p, ['a'], ['x'], ['train'])


class StatisticsTests(unittest.TestCase):
    def fixture(self):
        reference = pd.DataFrame(dict(seed=[42]*4, split=['test']*4,
                                     target_gene=list('abcd'), mse=[2, 4, 6, 8]))
        current = reference.copy(); current['mse'] -= 1
        return reference, current

    def test_paired_bootstrap_known_interval_and_alignment(self):
        reference, current = self.fixture()
        result = paired_bootstrap(reference, current.iloc[::-1], repeats=1000)
        self.assertEqual(result['mean_mse_improvement_over_train_mean'], 1)
        self.assertEqual(result['ci95_low'], 1)
        self.assertEqual(result['ci95_high'], 1)

    def test_bootstrap_reproducible(self):
        reference, current = self.fixture(); current.loc[0, 'mse'] = 6
        self.assertEqual(paired_bootstrap(reference, current), paired_bootstrap(reference, current))

    def test_bootstrap_rejects_pooled_seeds_and_missing_pairs(self):
        reference, current = self.fixture()
        current.loc[0, 'seed'] = 123
        with self.assertRaises(ValueError): paired_bootstrap(reference, current)
        reference, current = self.fixture()
        with self.assertRaises(ValueError): paired_bootstrap(reference, current.iloc[:-1])

    def test_valid_counts_and_sample_std(self):
        rows = MetricsTests().score([[0, 0], [1, 2]], [[1, 2], [1, 2]])
        frame = pd.DataFrame(rows)
        s = summarize(frame).query("subset=='all'")
        self.assertEqual(s.iloc[0].pearson_n_valid, 1)
        pieces = []
        for seed, value in ((42, 1), (123, 2), (2026, 3)):
            c = s.copy(); c['seed'] = seed; c['mse'] = value; pieces.append(c)
        aggregate = across_seeds(pd.concat(pieces)).iloc[0]
        self.assertEqual(aggregate.mse_mean, 2)
        self.assertEqual(aggregate.mse_std, 1)

    def test_strength_strata_shared_across_models(self):
        rows = []
        for model in ('zero', 'model'):
            for i, value in enumerate([1, 4, 9, 16, 25, 36]):
                rows.append(dict(seed=42, split='test', model=model, target_gene=str(i), mse=value))
        frame, thresholds = add_strength_strata(pd.DataFrame(rows))
        self.assertEqual(frame.groupby('target_gene').strength_stratum.nunique().max(), 1)
        self.assertEqual(len(thresholds), 1)
        np.testing.assert_array_equal(frame.query("model=='zero'").truth_rms, [1, 2, 3, 4, 5, 6])


class IndependentIntegrationTests(unittest.TestCase):
    def build_fixture(self, root):
        base, inputs, data = root/'experiment', root/'data/processed/c_inputs/seed_42', root/'data/processed/seed_42'
        inputs.mkdir(parents=True); data.mkdir(parents=True)
        genes = np.asarray(['g0', 'g1', 'g2'])
        spec = dict(seed=42, targets=dict(train=['tr0', 'tr1'], val=['va'], test=['te0', 'te1']),
                    control_cell_ids=dict(train=['c0'], val=['c1'], test=['c2']))
        write_json(inputs/'split.json', spec)
        (inputs/'output_genes.txt').write_text('g0\ng1\ng2\n', encoding='utf-8')
        trainval = np.array([[0, 0, 0], [2, 4, 6], [2, 2, 4]], float)
        truth = np.array([[1, 4, 7], [2, 2, 2]], float)
        np.savez_compressed(inputs/'labels_train_val.npz', target_ids=np.asarray(['tr0', 'tr1', 'va']),
                            gene_ids=genes, split=np.asarray(['train', 'train', 'val']), delta_true=trainval)
        np.savez_compressed(data/'D_test.npz', target_ids=np.asarray(['te0', 'te1']), gene_ids=genes,
                            split=np.asarray(['test', 'test']), delta_true=truth)
        mean = trainval[:2].mean(axis=0)
        models = ['zero', 'train_mean', 'ridge_fixture']
        for model in models:
            folder = base/model/'seed_42'; folder.mkdir(parents=True)
            write_json(folder/'run.json', dict(train_target_ids=spec['targets']['train'], val_target_ids=['va']))
            np.savez_compressed(folder/'model_arrays.npz', train_mean=mean)
            for split, targets, values, filename in [('test', ['te0', 'te1'], truth, 'predictions.npz'),
                                                    ('val', ['va'], trainval[2:], 'validation_predictions.npz')]:
                pred = np.zeros_like(values) if model == 'zero' else np.tile(mean, (len(targets), 1)) if model == 'train_mean' else values
                np.savez_compressed(folder/filename, target_ids=np.asarray(targets), gene_ids=genes, delta_pred=pred,
                                    feature_available=np.ones(len(targets), bool), used_fallback=np.zeros(len(targets), bool))
        frozen = dict(seeds=[42], models=models, external_input_hashes={
            f'data/processed/c_inputs/seed_42/{file}': sha256(inputs/file)
            for file in ['split.json', 'labels_train_val.npz', 'output_genes.txt']},
            test_label_expected_sha256={'42': sha256(data/'D_test.npz')})
        return base, frozen, data

    def test_independent_end_to_end_known_values(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); base, frozen, data = self.build_fixture(root)
            result, evidence = score_independently(base, root, frozen)
            self.assertEqual(len(result), 9)
            self.assertTrue((result.query("model=='ridge_fixture'").mse == 0).all())
            expected = np.mean((np.array([[1,4,7], [2,2,2]])-np.array([1,2,3]))**2, axis=1)
            np.testing.assert_allclose(result.query("model=='train_mean' and split=='test'").mse, expected)
            self.assertEqual(len(evidence), 5)

    def test_missing_B_input_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); base, frozen, data = self.build_fixture(root)
            (data/'D_test.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'B input missing'):
                score_independently(base, root, frozen)

    def test_B_version_tampering_rejected(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); base, frozen, data = self.build_fixture(root)
            with (data/'D_test.npz').open('ab') as stream: stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                score_independently(base, root, frozen)

    def test_A_reference_contract_and_hash(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); base, frozen, data = self.build_fixture(root)
            path = base/'ridge_fixture/seed_42/predictions.npz'
            manifest = root/'A.json'
            a = dict(model='knn_fixture', test_labels_accessed=False, seeds={'42': dict(
                predictions=str(path), sha256=sha256(path))})
            write_json(manifest, a)
            result, evidence = score_independently(base, root, frozen, manifest)
            self.assertEqual(len(result.query("model=='knn_fixture'")), 2)
            a['seeds']['42']['sha256'] = 'incorrect'; write_json(manifest, a)
            with self.assertRaisesRegex(ValueError, 'A prediction hash mismatch'):
                score_independently(base, root, frozen, manifest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
