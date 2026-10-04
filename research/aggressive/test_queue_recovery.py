import unittest
from unittest.mock import patch
from .core import retry_registry_upsert
from research import run_aggressive_enhanced_pipeline as queue

class RecoveryTests(unittest.TestCase):
    def test_registry_retries_windows_permission_errors(self):
        row={'experiment_id':'fixture'}
        with patch('research.registry.upsert',side_effect=[PermissionError('locked'),None]) as call,patch('research.aggressive.core.time.sleep'):
            retry_registry_upsert(row)
            self.assertEqual(call.call_count,2)
    def test_registry_retry_is_bounded(self):
        with patch('research.registry.upsert',side_effect=PermissionError('locked')) as call,patch('research.aggressive.core.time.sleep'):
            with self.assertRaises(PermissionError):retry_registry_upsert({})
            self.assertEqual(call.call_count,16)
    def test_registry_other_errors_not_hidden(self):
        with patch('research.registry.upsert',side_effect=ValueError('bad row')):
            with self.assertRaises(ValueError):retry_registry_upsert({})
    def test_planned_handoff_is_not_general_failure_override(self):
        with patch.object(queue.Path,'exists',return_value=True),patch.object(queue.Path,'read_text',return_value='{"handoff_complete":true,"decision":{"stop":true}}'):
            self.assertTrue(queue.planned_handoff_pending('s32_efficientnet_b3_cbam_enhanced_seed42'))
            self.assertFalse(queue.planned_handoff_pending('s33_densenet201_enhanced_seed42'))
        with patch.object(queue.Path,'exists',return_value=True),patch.object(queue.Path,'read_text',return_value='{"handoff_complete":false,"decision":{"stop":true}}'):
            self.assertFalse(queue.planned_handoff_pending('s32_efficientnet_b3_cbam_enhanced_seed42'))

if __name__=='__main__':unittest.main()
