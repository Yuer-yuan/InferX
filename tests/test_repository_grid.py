import unittest
from scripts.run_repository_pilot import job_grid


class GridTests(unittest.TestCase):
    def test_complete_paired_grid_and_split_isolation(self):
        tasks = [dict(task_id=f'test/{r*6+i}', split='dev' if i < 2 else 'eval',
                      repository_index=r) for r in range(4) for i in range(6)]
        dev = job_grid(tasks, 'dev'); evaluation = job_grid(tasks, 'eval')
        self.assertEqual(len(dev), 48)
        self.assertEqual(len(evaluation), 64)
        self.assertTrue(all(t['split'] == 'dev' for j in dev for t in j['tasks']))
        self.assertTrue(all(t['split'] == 'eval' for j in evaluation for t in j['tasks']))
        keys = [(j['episode_id'], j['mode'], j['policy'], j['round_id']) for j in evaluation]
        self.assertEqual(len(set(keys)), 64)
        for episode in range(4):
            paired = [j for j in evaluation if j['episode_id'] == f'episode/{episode}']
            self.assertEqual(len({tuple(t['task_id'] for t in j['tasks']) for j in paired}), 1)
            self.assertEqual({t['repository_index'] for t in paired[0]['tasks']}, set(range(4)))
        self.assertEqual(job_grid(tasks, 'eval'), evaluation)
