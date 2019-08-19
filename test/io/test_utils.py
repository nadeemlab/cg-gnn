"""Unit test for complex_module.core."""
import unittest
from histocartography.io.utils import download_file_to_local
from histocartography.io.utils import get_s3





class CoreTestCase(unittest.TestCase):
    """CoreTestCase class."""

    def setUp(self):
        """Setting up the test."""

        self.s3 = get_s3()
        pass

    def test_download_file_to_local(self):
        """Test download_file_to_local()."""

        desired_name = 'tmp.svs'
        saved_name = download_file_to_local(s3=self.s3, local_name=desired_name)


        self.assertEqual(saved_name, desired_name)

    def tearDown(self):
        """Tear down the tests."""
        pass