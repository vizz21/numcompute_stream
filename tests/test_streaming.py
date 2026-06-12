import unittest
import numpy as np
from numcompute_stream import (
    StreamingDecisionTreeClassifier,
    RandomForestClassifier,
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StreamingAccuracy,
    StreamingConfusionMatrix,
    StreamingMeanVariance,
    Pipeline,
    StreamTrainer,
    load_csv
)

class TestStreaming(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]])
        self.y = np.array([0, 0, 1, 1, 1])
        self.classes = np.array([0, 1])

    def test_tree_fit_predict(self):
        tree = StreamingDecisionTreeClassifier(max_depth=3)
        tree.fit(self.X, self.y)
        preds = tree.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)
        self.assertTrue(np.all(np.isin(preds, self.classes)))

    def test_tree_partial_fit(self):
        tree = StreamingDecisionTreeClassifier(max_depth=3)
        tree.partial_fit(self.X[:2], self.y[:2], classes=self.classes)
        tree.partial_fit(self.X[2:], self.y[2:])
        preds = tree.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_rf_fit_predict(self):
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.fit(self.X, self.y)
        preds = rf.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_rf_partial_fit(self):
        rf = RandomForestClassifier(n_estimators=5, max_depth=3)
        rf.partial_fit(self.X[:2], self.y[:2], classes=self.classes)
        rf.partial_fit(self.X[2:], self.y[2:])
        preds = rf.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_scaler_partial_fit(self):
        scaler = StandardScaler()
        scaler.partial_fit(self.X[:3])
        scaler.partial_fit(self.X[3:])
        
        scaler_full = StandardScaler()
        scaler_full.fit(self.X)
        
        np.testing.assert_array_almost_equal(scaler.mean_, scaler_full.mean_)
        np.testing.assert_array_almost_equal(scaler.std_, scaler_full.std_)

    def test_minmax_partial_fit(self):
        scaler = MinMaxScaler()
        scaler.partial_fit(self.X[:3])
        scaler.partial_fit(self.X[3:])
        
        scaler_full = MinMaxScaler()
        scaler_full.fit(self.X)
        
        np.testing.assert_array_almost_equal(scaler.min_, scaler_full.min_)
        np.testing.assert_array_almost_equal(scaler.max_, scaler_full.max_)

    def test_onehot_partial_fit(self):
        encoder = OneHotEncoder()
        encoder.partial_fit(np.array([0, 1]))
        encoder.partial_fit(np.array([1, 2]))
        
        self.assertEqual(len(encoder.categories_), 3)
        encoded = encoder.transform(np.array([0, 2]))
        self.assertEqual(encoded.shape, (2, 3))

    def test_imputer_partial_fit(self):
        X_nan = np.array([[1.0, np.nan], [np.nan, 3.0], [5.0, 6.0]])
        imputer = SimpleImputer(strategy="mean")
        imputer.partial_fit(X_nan[:2])
        imputer.partial_fit(X_nan[2:])
        
        expected_stats = np.array([3.0, 4.5])
        np.testing.assert_array_almost_equal(imputer.statistics_, expected_stats)

    def test_streaming_accuracy(self):
        acc = StreamingAccuracy()
        acc.update(np.array([0, 1]), np.array([0, 0])) 
        acc.update(np.array([1, 1]), np.array([1, 1])) 
        self.assertEqual(acc.result(), 0.75)

    def test_streaming_confusion_matrix(self):
        cm = StreamingConfusionMatrix(labels=[0, 1])
        cm.update(np.array([0, 1]), np.array([0, 0]))
        cm.update(np.array([1, 1]), np.array([1, 1]))
        expected = np.array([[1, 0], [1, 2]])
        np.testing.assert_array_equal(cm.result(), expected)

    def test_streaming_mean_variance(self):
        smv = StreamingMeanVariance()
        smv.update(self.X[:, 0])
        np.testing.assert_almost_equal(smv.mean(), np.mean(self.X[:, 0]))
        np.testing.assert_almost_equal(smv.variance(), np.var(self.X[:, 0]))

    def test_pipeline_partial_fit(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', StreamingDecisionTreeClassifier(max_depth=3))
        ])
        pipe.partial_fit(self.X[:3], self.y[:3])
        pipe.partial_fit(self.X[3:], self.y[3:])
        
        preds = pipe.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_stream_trainer(self):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', StreamingDecisionTreeClassifier(max_depth=3))
        ])
        trainer = StreamTrainer(pipe)
        
        for i in range(0, len(self.X), 2):
            X_chunk = self.X[i:i+2]
            y_chunk = self.y[i:i+2]
            trainer.update(X_chunk, y_chunk)
        
        metrics = trainer.get_metrics()
        self.assertIn('accuracy', metrics)
        self.assertIn('confusion_matrix', metrics)

    def test_empty_input(self):
        tree = StreamingDecisionTreeClassifier()
        with self.assertRaises(ValueError):
            tree.partial_fit(np.array([]).reshape(0, 2), np.array([]))

    def test_nan_handling(self):
        X_nan = np.array([[1.0, np.nan], [2.0, 3.0]])
        scaler = StandardScaler()

        scaler.partial_fit(X_nan)
        self.assertFalse(np.isnan(scaler.mean_).any())

    def test_zero_variance(self):
        X_const = np.array([[1.0, 1.0], [1.0, 1.0]])
        scaler = StandardScaler()
        scaler.fit(X_const)
        np.testing.assert_array_equal(scaler.std_, np.array([1.0, 1.0]))

    def test_tree_max_depth(self):
        tree = StreamingDecisionTreeClassifier(max_depth=1)
        tree.fit(self.X, self.y)
        self.assertLessEqual(self._count_leaves(tree.root), 2)

    def _count_leaves(self, node):
        if node.is_leaf: return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def test_rf_n_estimators(self):
        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(self.X, self.y)
        self.assertEqual(len(rf.trees), 10)

    def test_imputer_constant(self):
        imputer = SimpleImputer(strategy="constant", fill_value=99.0)
        X_nan = np.array([[np.nan, 1.0], [2.0, np.nan]])
        imputer.fit(X_nan)
        X_imputed = imputer.transform(X_nan)
        self.assertEqual(X_imputed[0, 0], 99.0)
        self.assertEqual(X_imputed[1, 1], 99.0)

    def test_imputer_median(self):
        imputer = SimpleImputer(strategy="median")
        X = np.array([[1.0], [3.0], [10.0]])
        imputer.partial_fit(X[:2]) # 1, 3 -> median 2
        imputer.partial_fit(X[2:]) # 1, 3, 10 -> median 3
        self.assertEqual(imputer.statistics_[0], 3.0)

    def test_pipeline_transform(self):
        pipe = Pipeline([('scaler', StandardScaler())])
        pipe.fit(self.X)
        X_trans = pipe.transform(self.X)
        self.assertEqual(X_trans.shape, self.X.shape)
        np.testing.assert_almost_equal(np.mean(X_trans, axis=0), 0.0)

    def test_streaming_accuracy_reset(self):
        acc = StreamingAccuracy()
        acc.update(np.array([0]), np.array([0]))
        acc.reset()
        self.assertEqual(acc.result(), 0.0)

    def test_streaming_mean_variance_reset(self):
        smv = StreamingMeanVariance()
        smv.update(np.array([1, 2, 3]))
        smv.reset()
        self.assertTrue(np.isnan(smv.mean()))

    def test_minmax_range(self):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(self.X)
        X_trans = scaler.transform(self.X)
        self.assertEqual(np.min(X_trans), -1.0)
        self.assertEqual(np.max(X_trans), 1.0)

    def test_onehot_transform_unseen(self):
        encoder = OneHotEncoder()
        encoder.fit(np.array([0, 1]))
        X_unseen = np.array([2])
        encoded = encoder.transform(X_unseen)
        np.testing.assert_array_equal(encoded, [[0, 0]])

    def test_stream_trainer_score_chunk(self):
        pipe = Pipeline([('tree', StreamingDecisionTreeClassifier())])
        trainer = StreamTrainer(pipe)
        trainer.update(self.X, self.y)
        scores = trainer.score_chunk(self.X, self.y)
        self.assertIn('accuracy', scores)
        self.assertGreaterEqual(scores['accuracy'], 0.0)

    def test_tree_criterion_entropy(self):
        tree = StreamingDecisionTreeClassifier(criterion='entropy')
        tree.fit(self.X, self.y)
        preds = tree.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_stats_welford_function(self):
        from numcompute_stream.stats import welford
        m, v = welford(self.X[:, 0])
        np.testing.assert_almost_equal(m, np.mean(self.X[:, 0]))
        np.testing.assert_almost_equal(v, np.var(self.X[:, 0], ddof=1))

    def test_metrics_f1(self):
        from numcompute_stream.metrics import f1
        score = f1(np.array([0, 1, 1]), np.array([0, 1, 0]))
        self.assertAlmostEqual(score, 2/3)

    def test_io_load_csv(self):
        import os
        csv_content = "feat1,feat2,label\n1.0,2.0,0\n3.0,4.0,1"
        with open("test_io.csv", "w") as f:
            f.write(csv_content)
        data = load_csv("test_io.csv")
        self.assertEqual(data.shape, (2, 3))
        os.remove("test_io.csv")

if __name__ == '__main__':
    unittest.main()
