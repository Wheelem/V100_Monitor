import unittest
from unittest.mock import patch, MagicMock

from monitor.ssh_client import V100SSHClient, NvidiaSMIError


class TestV100SSHClient(unittest.TestCase):
    def setUp(self):
        # Patch paramiko to avoid real SSH connections
        patcher = patch('monitor.ssh_client.paramiko.SSHClient')
        self.addCleanup(patcher.stop)
        self.mock_client_class = patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_class.return_value = self.mock_client

    def _setup_exec(self, stdout_data: str, stderr_data: str = ""):
        stdout = MagicMock()
        stdout.read.return_value = stdout_data.encode('utf-8')
        stderr = MagicMock()
        stderr.read.return_value = stderr_data.encode('utf-8')
        self.mock_client.exec_command.return_value = (None, stdout, stderr)

    def test_successful_parse(self):
        output = "10, 200, 70, 100, 1024, 16384\n"
        self._setup_exec(output)
        client = V100SSHClient('host', 'user')
        data = client.run_nvidia_smi()
        self.assertEqual(data.strip(), output.strip())
        gpus = V100SSHClient.parse_nvidia_smi(data)
        self.assertEqual(len(gpus), 1)
        self.assertAlmostEqual(gpus[0].gpu_util, 10)

    def test_missing_column(self):
        output = "10, 200, 70, 100\n"
        self._setup_exec(output)
        client = V100SSHClient('host', 'user')
        data = client.run_nvidia_smi()
        with self.assertRaises(NvidiaSMIError):
            V100SSHClient.parse_nvidia_smi(data)

    def test_ssh_failure(self):
        self.mock_client.exec_command.side_effect = Exception('ssh error')
        client = V100SSHClient('host', 'user')
        with self.assertRaises(NvidiaSMIError):
            client.run_nvidia_smi()

    def test_no_output(self):
        self._setup_exec('')
        client = V100SSHClient('host', 'user')
        with self.assertRaises(NvidiaSMIError):
            client.run_nvidia_smi()


if __name__ == "__main__":
    unittest.main()