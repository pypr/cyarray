"""Tests for the carray module.

Only the LongArray is tested. As the code in carray.pyx is auto-generated,
tests for one class hould suffice.

"""

# standard imports
import unittest
import numpy
import pytest

# local imports
from cyarray.carray import LongArray, py_aligned


class TestAligned(unittest.TestCase):

    def test_aligned_to_64_bits(self):
        self.assertEqual(py_aligned(12, 1), 64)
        self.assertEqual(py_aligned(1, 1), 64)
        self.assertEqual(py_aligned(64, 1), 64)
        self.assertEqual(py_aligned(120, 1), 128)

        self.assertEqual(py_aligned(1, 2), 32)
        self.assertEqual(py_aligned(12, 2), 32)
        self.assertEqual(py_aligned(32, 2), 32)
        self.assertEqual(py_aligned(33, 2), 64)

        self.assertEqual(py_aligned(1, 3), 64)
        self.assertEqual(py_aligned(65, 3), 256)

        self.assertEqual(py_aligned(1, 4), 16)
        self.assertEqual(py_aligned(16, 4), 16)
        self.assertEqual(py_aligned(21, 4), 32)

        self.assertEqual(py_aligned(1, 5), 64)
        self.assertEqual(py_aligned(13, 5), 128)

        self.assertEqual(py_aligned(1, 8), 8)
        self.assertEqual(py_aligned(8, 8), 8)
        self.assertEqual(py_aligned(11, 8), 16)


class TestLongArray(unittest.TestCase):
    """
    Tests for the LongArray class.
    """

    def test_constructor(self):
        """
        Test the constructor.
        """
        la = LongArray(10)

        self.assertEqual(la.length, 10)
        self.assertEqual(la.alloc, 10)
        self.assertEqual(len(la.get_npy_array()), 10)

        la = LongArray()

        self.assertEqual(la.length, 0)
        self.assertEqual(la.alloc, 16)
        self.assertEqual(len(la.get_npy_array()), 0)

    def test_get_set_indexing(self):
        """
        Test get/set and [] operator.
        """
        la = LongArray(10)
        la.set(0, 10)
        la.set(9, 1)

        self.assertEqual(la.get(0), 10)
        self.assertEqual(la.get(9), 1)

        la[9] = 2
        self.assertEqual(la[9], 2)

    def test_append(self):
        """
        Test the append function.
        """
        la = LongArray(0)
        la.append(1)
        la.append(2)
        la.append(3)

        self.assertEqual(la.length, 3)
        self.assertEqual(la[0], 1)
        self.assertEqual(la[1], 2)
        self.assertEqual(la[2], 3)

    def test_reserve(self):
        """
        Tests the reserve function.
        """
        la = LongArray(0)
        la.reserve(10)

        self.assertEqual(la.alloc, 16)
        self.assertEqual(la.length, 0)
        self.assertEqual(len(la.get_npy_array()), 0)

        la.reserve(20)
        self.assertEqual(la.alloc, 20)
        self.assertEqual(la.length, 0)
        self.assertEqual(len(la.get_npy_array()), 0)

    def test_resize(self):
        """
        Tests the resize function.
        """
        la = LongArray(0)

        la.resize(20)
        self.assertEqual(la.length, 20)
        self.assertEqual(len(la.get_npy_array()), 20)
        self.assertEqual(la.alloc >= la.length, True)

    def test_get_npy_array(self):
        """
        Tests the get_npy_array array.
        """
        la = LongArray(3)
        la[0] = 1
        la[1] = 2
        la[2] = 3

        nparray = la.get_npy_array()
        self.assertEqual(len(nparray), 3)

        for i in range(3):
            self.assertEqual(nparray[0], la[0])

    def test_set_data(self):
        """
        Tests the set_data function.
        """
        la = LongArray(5)
        np = numpy.arange(5)
        la.set_data(np)

        for i in range(5):
            self.assertEqual(la[i], np[i])

        self.assertRaises(ValueError, la.set_data, numpy.arange(10))

    def test_squeeze(self):
        la = LongArray(5)
        la.append(4)

        self.assertEqual(la.alloc > la.length, True)

        la.squeeze()

        self.assertEqual(la.length, 6)
        self.assertEqual(la.alloc >= la.length, True)
        self.assertEqual(len(la.get_npy_array()), 6)

    def test_squeeze_for_zero_length_array(self):
        # Given.
        la = LongArray()

        # When
        la.squeeze()

        # Then
        self.assertEqual(la.length, 0)
        self.assertEqual(len(la.get_npy_array()), 0)
        self.assertEqual(la.alloc >= la.length, True)
        del la  # This should work and not segfault.

    def test_squeeze_large_array_should_not_segfault(self):
        # Given
        la = LongArray(10)
        la.set_data(numpy.zeros(10, dtype=int))
        la.reserve(100000)

        # When
        la.squeeze()
        la.reserve(1000)

        # Then
        self.assertEqual(la.length, 10)
        numpy.testing.assert_array_almost_equal(la.get_npy_array(), 0)
        self.assertEqual(la.alloc >= la.length, True)

    def test_reset(self):
        """
        Tests the reset function.
        """
        la = LongArray(5)
        la.reset()

        self.assertEqual(la.length, 0)
        self.assertEqual(la.alloc, 5)
        self.assertEqual(len(la.get_npy_array()), 0)

    def test_extend(self):
        """
        Tests the extend function.
        """
        l1 = LongArray(5)

        for i in range(5):
            l1[i] = i

        l2 = LongArray(5)

        for i in range(5):
            l2[i] = 5 + i

        l1.extend(l2.get_npy_array())

        self.assertEqual(l1.length, 10)
        self.assertEqual(
            numpy.allclose(
                l1.get_npy_array(),
                numpy.arange(10)),
            True)

    def test_remove(self):
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))
        rem = [0, 4, 3]
        l1.remove(numpy.array(rem, dtype=int))
        self.assertEqual(l1.length, 7)
        self.assertEqual(numpy.allclose([7, 1, 2, 8, 9, 5, 6],
                                        l1.get_npy_array()), True)

        l1.remove(numpy.array(rem, dtype=int))
        self.assertEqual(l1.length, 4)
        self.assertEqual(numpy.allclose(
            [6, 1, 2, 5], l1.get_npy_array()), True)

        rem = [0, 1, 3]
        l1.remove(numpy.array(rem, dtype=int))
        self.assertEqual(l1.length, 1)
        self.assertEqual(numpy.allclose([2], l1.get_npy_array()), True)

        l1.remove(numpy.array([0], dtype=int))
        self.assertEqual(l1.length, 0)
        self.assertEqual(len(l1.get_npy_array()), 0)

    def test_remove_with_strides(self):
        # Given
        l1 = LongArray(12)
        l1.set_data(numpy.arange(12))

        # When
        rem = [3, 1]
        l1.remove(numpy.array(rem, dtype=int), stride=3)

        # Then
        self.assertEqual(l1.length, 6)
        self.assertEqual(numpy.allclose([0, 1, 2, 6, 7, 8],
                                        l1.get_npy_array()), True)

        # Given
        l1 = LongArray(12)
        l1.set_data(numpy.arange(12))

        # When
        rem = [0, 2]
        l1.remove(numpy.array(rem, dtype=int), stride=3)

        # Then
        self.assertEqual(l1.length, 6)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.assertEqual(numpy.allclose([9, 10, 11, 3, 4, 5],
                                        l1.get_npy_array()), True)

    def test_align_array(self):
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        new_indices = LongArray(10)
        new_indices.set_data(numpy.asarray([1, 5, 3, 2, 4, 7, 8, 6, 9, 0]))

        l1.align_array(new_indices)
        self.assertEqual(numpy.allclose([1, 5, 3, 2, 4, 7, 8, 6, 9, 0],
                                        l1.get_npy_array()), True)

        # Test case with strides.
        l1 = LongArray(6)
        l1.set_data(numpy.arange(6))

        new_indices = LongArray(3)
        new_indices.set_data(numpy.asarray([2, 1, 0]))
        l1.align_array(new_indices, 2)
        self.assertEqual(numpy.allclose([4, 5, 2, 3, 0, 1],
                                        l1.get_npy_array()), True)

    def test_copy_subset(self):
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        l2 = LongArray(4)
        l2[0] = 4
        l2[1] = 3
        l2[2] = 2
        l2[3] = 1

        # a valid copy.
        l1.copy_subset(l2, 5, 9)
        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 4, 3, 2, 1, 9],
                                        l1.get_npy_array()), True)

        # try to copy different sized arrays without any index specification.
        l1.set_data(numpy.arange(10))
        # copy to the last k values of source array.
        l1.copy_subset(l2, start_index=6)
        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 5, 4, 3, 2, 1],
                                        l1.get_npy_array()), True)

        l1.set_data(numpy.arange(10))
        l1.copy_subset(l2, start_index=7)
        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 5, 6, 4, 3, 2],
                                        l1.get_npy_array()), True)

        # some invalid operations.
        l1.set_data(numpy.arange(10))
        self.assertRaises(ValueError, l1.copy_subset, l2, -1, 1)
        self.assertRaises(ValueError, l1.copy_subset, l2, 3, 2)
        self.assertRaises(ValueError, l1.copy_subset, l2, 0, 11)
        self.assertRaises(ValueError, l1.copy_subset, l2, 10, 20)
        self.assertRaises(ValueError, l1.copy_subset, l2, -1, -1)

    def test_copy_subset_works_with_strides(self):
        # Given
        l1 = LongArray(8)
        l1.set_data(numpy.arange(8))

        l2 = LongArray(4)
        l2.set_data(numpy.arange(10, 14))

        # When
        l1.copy_subset(l2, 2, 3, stride=2)

        # Then
        numpy.testing.assert_array_equal(
            l1.get_npy_array(),
            [0, 1, 2, 3, 10, 11, 6, 7]
        )

        # When
        l1.copy_subset(l2, 2, 4, stride=2)

        # Then
        numpy.testing.assert_array_equal(
            l1.get_npy_array(),
            [0, 1, 2, 3, 10, 11, 12, 13]
        )

    def test_copy_values(self):
        # Given
        l1 = LongArray(8)
        l1.set_data(numpy.arange(8))
        l2 = LongArray(8)
        l2.set_data(numpy.zeros(8, dtype=int))

        # When
        indices = LongArray(3)
        indices.set_data(numpy.array([2, 4, 6]))
        l1.copy_values(indices, l2)

        # Then
        numpy.testing.assert_array_equal(
            l2.get_npy_array(),
            [2, 4, 6] + [0] * 5
        )

        # When
        l2.set_data(numpy.zeros(8, dtype=int))
        indices.set_data(numpy.array([1, 2, 3]))

        l1.copy_values(indices, l2, stride=2)

        # Then
        numpy.testing.assert_array_equal(
            l2.get_npy_array(),
            [2, 3, 4, 5, 6, 7, 0, 0]
        )

    def test_copy_values_with_start_index(self):
        # Given
        l1 = LongArray(8)
        l1.set_data(numpy.arange(8))
        l2 = LongArray(8)
        l2.set_data(numpy.zeros(8, dtype=int))

        # When
        indices = LongArray(3)
        indices.set_data(numpy.array([2, 4, 6]))
        l1.copy_values(indices, l2, start=5)

        # Then
        numpy.testing.assert_array_equal(
            l2.get_npy_array(),
            [0] * 5 + [2, 4, 6]
        )

        # When
        l2.set_data(numpy.zeros(8, dtype=int))
        indices.set_data(numpy.array([1, 2, 3]))

        l1.copy_values(indices, l2, stride=2, start=2)

        # Then
        numpy.testing.assert_array_equal(
            l2.get_npy_array(),
            [0, 0, 2, 3, 4, 5, 6, 7]
        )

    def test_update_min_max(self):
        """
        Tests the update_min_max function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        l1.update_min_max()

        self.assertEqual(l1.minimum, 0)
        self.assertEqual(l1.maximum, 9)

        l1[9] = -1
        l1[0] = -20
        l1[4] = 200
        l1.update_min_max()

        self.assertEqual(l1.minimum, -20)
        self.assertEqual(l1.maximum, 200)

    def test_pickling(self):
        """
        Tests the __reduce__ and __setstate__ functions.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        import pickle

        l1_dump = pickle.dumps(l1)

        l1_load = pickle.loads(l1_dump)
        self.assertEqual(
            (l1_load.get_npy_array() == l1.get_npy_array()).all(), True)

    def test_set_view(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When.
        view = LongArray()
        view.set_view(src, 1, 4)

        # Then.
        self.assertEqual(view.length, 3)
        expect = list(range(1, 4))
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_set_view_for_empty_array(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When.
        view = LongArray()
        view.set_view(src, 1, 1)

        # Then.
        self.assertEqual(view.length, 0)
        expect = []
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_set_view_stores_reference_to_parent(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When
        view = LongArray()
        view.set_view(src, 1, 4)
        del src

        # Then.
        self.assertEqual(view.length, 3)
        expect = list(range(1, 4))
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_reset_works_after_set_view(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))
        view = LongArray()
        view.set_view(src, 1, 3)

        # When.
        view.reset()
        view.extend(numpy.arange(3) * 10)

        # Then.
        self.assertEqual(view.length, 3)
        expect = (numpy.arange(3) * 10).tolist()
        self.assertListEqual(view.get_npy_array().tolist(), expect)


class BenchmarkLongArray(unittest.TestCase):
    """
    Tests for the LongArray class.
    """

    @pytest.fixture(autouse=True)
    def setupBenchmark(self, benchmark):
        self.benchmark = benchmark

    def test_constructor(self):
        """
        Test the constructor.
        """
        n = numpy.random.randint(low=10, high=100)
        la = self.benchmark(LongArray, n)

        self.assertEqual(la.length, n)
        self.assertEqual(la.alloc, n)
        self.assertEqual(len(la.get_npy_array()), n)

    def test_set_indexing(self):
        n = 100
        lab = LongArray(n)
        self.benchmark.pedantic(lab.set, args=(9, n))
        self.assertEqual(lab[9], n)

    def test_get_indexing(self):
        la = LongArray(100)
        la[98] = 15
        res = self.benchmark(la.get, 98)
        self.assertEqual(res, 15)

    def test_append(self):
        lab = LongArray(0)
        n = 100
        self.benchmark(lab.append, n)
        self.assertEqual(lab[0], n)

    def test_reserve(self):
        """
        Tests the reserve function.
        """

        def breserve(n):
            la = LongArray(0)
            la.reserve(n)
            return la

        n = 100
        la = self.benchmark(breserve, n)
        self.assertEqual(la.alloc, n)

    def test_resize(self):
        """
        Tests the resize function.
        """

        def bresize(lab):
            n = numpy.random.randint(low=10, high=20)
            lab.resize(n)
            return lab, n

        la = LongArray(10)
        la, n = self.benchmark(bresize, la)
        self.assertEqual(la.length, n)

    def test_get_npy_array(self):
        la = LongArray(100)
        la[0] = 1
        la[1] = 2
        la[2] = 3

        nparray = self.benchmark(la.get_npy_array)
        for i in range(3):
            self.assertEqual(nparray[0], la[0])

    def test_set_data(self):
        """
        Tests the set_data function.
        """
        n = 50
        la = LongArray(n)
        np = numpy.arange(n)
        self.benchmark(la.set_data, np)

        for i in range(n):
            self.assertEqual(la[i], np[i])

        self.assertRaises(ValueError, la.set_data, numpy.arange(55))

    def test_squeeze(self):

        def bsqueeze():
            lab = LongArray(5)
            lab.append(4)
            lab.squeeze()
            return lab

        la = self.benchmark(bsqueeze)

        self.assertEqual(la.length, 6)
        self.assertEqual(la.alloc >= la.length, True)
        self.assertEqual(len(la.get_npy_array()), 6)

    def test_reset(self):
        def breset():
            lab = LongArray(5)
            lab.reset()
            return lab

        la = self.benchmark(breset)

        self.assertEqual(la.length, 0)
        self.assertEqual(la.alloc, 5)
        self.assertEqual(len(la.get_npy_array()), 0)

    def test_extend(self):
        l2 = LongArray(5)

        for i in range(5):
            l2[i] = 5 + i

        def bextend(l2n):
            l1b = LongArray(0)
            l1b.extend(l2n.get_npy_array())
            return l1b

        l1 = self.benchmark(bextend, l2)

        self.assertEqual(l1.length, 5)
        self.assertEqual(
            numpy.allclose(
                l1.get_npy_array(),
                numpy.arange(5, 10)),
            True)

    def test_remove(self):

        def bremove(rem):
            l1b = LongArray(10)
            l1b.set_data(numpy.arange(10))
            l1b.remove(rem)
            return l1b

        rem = [0, 4, 3]
        l1 = self.benchmark(bremove, numpy.array(rem, dtype=int))

        self.assertEqual(l1.length, 7)
        self.assertEqual(numpy.allclose([7, 1, 2, 8, 9, 5, 6],
                                        l1.get_npy_array()), True)

    def test_remove_with_strides(self):

        def bremove(rem):
            l1b = LongArray(12)
            l1b.set_data(numpy.arange(12))
            l1b.remove(rem, stride=3)
            return l1b

        rem = [3, 1]
        l1 = self.benchmark(bremove, numpy.array(rem, dtype=int))

        # Then
        self.assertEqual(l1.length, 6)
        self.assertEqual(numpy.allclose([0, 1, 2, 6, 7, 8],
                                        l1.get_npy_array()), True)

        # Given
        l1 = LongArray(12)
        l1.set_data(numpy.arange(12))

        # When
        rem = [0, 2]
        l1.remove(numpy.array(rem, dtype=int), stride=3)

        # Then
        self.assertEqual(l1.length, 6)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.assertEqual(numpy.allclose([9, 10, 11, 3, 4, 5],
                                        l1.get_npy_array()), True)

    def test_align_array(self):
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))

        new_indices = LongArray(10)
        new_indices.set_data(numpy.asarray([1, 5, 3, 2, 4, 7, 8, 6, 9, 0]))

        l1.align_array(new_indices)
        self.assertEqual(numpy.allclose([1, 5, 3, 2, 4, 7, 8, 6, 9, 0],
                                        l1.get_npy_array()), True)

        # Test case with strides.

        def balign_array():
            l1b = LongArray(6)
            l1b.set_data(numpy.arange(6))

            new_indices = LongArray(3)
            new_indices.set_data(numpy.asarray([2, 1, 0]))
            l1b.align_array(new_indices, 2)
            return l1b

        l1 = self.benchmark(balign_array)
        self.assertEqual(numpy.allclose([4, 5, 2, 3, 0, 1],
                                        l1.get_npy_array()), True)

    def test_copy_subset(self):

        def bcopy_subset(l2b):
            l1b = LongArray(10)
            l1b.set_data(numpy.arange(10))

            # a valid copy.
            l1b.copy_subset(l2b, 5, 9)
            return l1b

        l2 = LongArray(4)
        l2[0] = 4
        l2[1] = 3
        l2[2] = 2
        l2[3] = 1

        l1 = self.benchmark(bcopy_subset, l2)

        self.assertEqual(numpy.allclose([0, 1, 2, 3, 4, 4, 3, 2, 1, 9],
                                        l1.get_npy_array()), True)

    def test_copy_subset_works_with_strides(self):
        def bcopy_subset(l2b):
            l1b = LongArray(8)
            l1b.set_data(numpy.arange(8))
            l1b.copy_subset(l2b, 2, 3, stride=2)
            return l1b

        # Given
        l2 = LongArray(4)
        l2.set_data(numpy.arange(10, 14))

        # When
        l1 = self.benchmark(bcopy_subset, l2)

        # Then
        numpy.testing.assert_array_equal(
            l1.get_npy_array(),
            [0, 1, 2, 3, 10, 11, 6, 7]
        )

    def test_copy_values(self):
        def bcopy_values(l2b, indices):
            l1b = LongArray(8)
            l1b.set_data(numpy.arange(8))
            l1b.copy_values(indices, l2b)
            return l1b

        # Given
        l1 = LongArray(8)
        l1.set_data(numpy.arange(8))
        l2 = LongArray(8)
        l2.set_data(numpy.zeros(8, dtype=int))

        # When
        indices = LongArray(3)
        indices.set_data(numpy.array([2, 4, 6]))
        l1 = self.benchmark.pedantic(bcopy_values, args=(l2, indices))

        # Then
        numpy.testing.assert_array_equal(
            l2.get_npy_array(),
            [2, 4, 6] + [0] * 5
        )

    def test_update_min_max(self):
        """
        Tests the update_min_max function.
        """

        def bupdate_min_max():
            l1b = LongArray(10)
            l1b.set_data(numpy.arange(10))
            l1b.update_min_max()
            return l1b

        l1 = self.benchmark(bupdate_min_max)

        self.assertEqual(l1.minimum, 0)
        self.assertEqual(l1.maximum, 9)

    def test_pickling(self):
        """
        Tests the __reduce__ and __setstate__ functions.
        """
        import pickle

        def bpickle(l1b):
            l1_dump = pickle.dumps(l1b)
            return pickle.loads(l1_dump)

        l1 = LongArray(3)
        l1.set_data(numpy.arange(3))

        l1_load = self.benchmark(bpickle, l1)
        self.assertEqual(
            (l1_load.get_npy_array() == l1.get_npy_array()).all(), True)

    def test_set_view(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When.
        def bset_view(bsrc):
            bview = LongArray()
            bview.set_view(bsrc, 1, 4)
            return bview

        view = self.benchmark(bset_view, src)

        # Then.
        self.assertEqual(view.length, 3)
        expect = list(range(1, 4))
        self.assertListEqual(view.get_npy_array().tolist(), expect)

    def test_set_view_for_empty_array(self):
        # Given
        src = LongArray()
        src.extend(numpy.arange(5))

        # When.

        def bset_view(bsrc):
            view = LongArray()
            view.set_view(bsrc, 1, 1)
            return view

        view = self.benchmark(bset_view, src)

        # Then.
        self.assertEqual(view.length, 0)
        expect = []
        self.assertListEqual(view.get_npy_array().tolist(), expect)


if __name__ == '__main__':
    unittest.main()
