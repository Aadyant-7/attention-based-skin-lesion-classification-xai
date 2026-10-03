"""CPU-only synthetic tests for saved-prediction alignment safeguards."""
import unittest
import pandas as pd
import numpy as np
from research.common import CLASSES
from research.fuse import aligned_probabilities, validate_candidate, CANDIDATES


class FusionChecks(unittest.TestCase):
    def setUp(self):
        self.validation = pd.DataFrame({'image_id':['a','b'], 'diagnosis':['akiec','bcc']}).set_index('image_id')
        self.frame = pd.DataFrame([dict(image_id=i,true_class=c,predicted_class=c,
            **{f'p_{k}':float(k==c) for k in CLASSES}) for i,c in [('a','akiec'),('b','bcc')]])

    def test_alignment_uses_identity_not_csv_order(self):
        values = aligned_probabilities(self.frame.iloc[::-1], self.validation)
        np.testing.assert_array_equal(values, np.eye(7)[:2])

    def test_duplicate_missing_or_wrong_labels_rejected(self):
        for broken in (self.frame.iloc[:1], pd.concat([self.frame,self.frame.iloc[:1]]),
                       self.frame.assign(true_class=['bcc','akiec'])):
            with self.assertRaises(ValueError): aligned_probabilities(broken,self.validation)

    def test_invalid_or_inconsistent_probabilities_rejected(self):
        for broken in (self.frame.assign(p_akiec=[np.nan,0]), self.frame.assign(p_akiec=[.2,0]),
                       self.frame.assign(predicted_class=['bcc','bcc'])):
            with self.assertRaises(ValueError): aligned_probabilities(broken,self.validation)

    def test_registered_pairs_and_triple_only_no_weight_or_subset_search(self):
        for rid, parents in CANDIDATES.items():
            config = dict(experiment_id=rid, parent_run_ids=parents, weights=[1/len(parents)]*len(parents),
                recipe_version='equal_probability_v1', protocol='exploratory_image_level', class_order=list(CLASSES))
            validate_candidate(config)
            for field, value in (('weights', [.8,.2]), ('parent_run_ids', parents[::-1]),
                                 ('experiment_id', 'unregistered'), ('protocol','strict_lesion_disjoint')):
                with self.assertRaises(ValueError): validate_candidate(dict(config, **{field:value}))


if __name__=='__main__': unittest.main()
