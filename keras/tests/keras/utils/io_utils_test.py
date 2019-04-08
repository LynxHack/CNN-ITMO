'''Tests for functions in io_utils.py.
'''
import os
import io
import pytest

from contextlib import contextmanager

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.io_utils import HDF5Matrix
from keras.utils.io_utils import H5Dict
from keras.utils.io_utils import ask_to_proceed_with_overwrite
from keras.utils.io_utils import save_to_binary_h5py
from keras.utils.io_utils import load_from_binary_h5py
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
import numpy as np
import six
import h5py
import tempfile
try:
    from pathlib import Path
    from unittest.mock import patch
except:
    from pathlib2 import Path
    from mock import patch


@pytest.fixture
def in_tmpdir(tmpdir):
    """Runs a function in a temporary directory.

    Checks that the directory is empty afterwards.
    """
    with tmpdir.as_cwd():
        yield None
    assert not tmpdir.listdir()


def create_dataset(h5_path='test.h5'):
    X = np.random.randn(200, 10).astype('float32')
    y = np.random.randint(0, 2, size=(200, 1))
    with h5py.File(h5_path, 'w') as f:
        # Creating dataset to store features
        X_dset = f.create_dataset('my_data', (200, 10), dtype='f')
        X_dset[:] = X
        # Creating dataset to store labels
        y_dset = f.create_dataset('my_labels', (200, 1), dtype='i')
        y_dset[:] = y


def test_io_utils(in_tmpdir):
    '''Tests the HDF5Matrix code using the sample from @jfsantos at
    https://gist.github.com/jfsantos/e2ef822c744357a4ed16ec0c885100a3
    '''
    _, h5_path = tempfile.mkstemp('.h5')
    create_dataset(h5_path)

    # Instantiating HDF5Matrix for the training set,
    # which is a slice of the first 150 elements
    X_train = HDF5Matrix(h5_path, 'my_data', start=0, end=150)
    y_train = HDF5Matrix(h5_path, 'my_labels', start=0, end=150)

    # Likewise for the test set
    X_test = HDF5Matrix(h5_path, 'my_data', start=150, end=200)
    y_test = HDF5Matrix(h5_path, 'my_labels', start=150, end=200)

    # HDF5Matrix behave more or less like Numpy matrices with regards to indexing
    assert y_train.shape == (150, 1), 'HDF5Matrix shape should match input array'
    # But they do not support negative indices, so don't try print(X_train[-1])

    assert y_train.dtype == np.dtype('i'), (
        'HDF5Matrix dtype should match input array')
    assert y_train.ndim == 2, 'HDF5Matrix ndim should match input array'
    assert y_train.size == 150, 'HDF5Matrix ndim should match input array'

    model = Sequential()
    model.add(Dense(64, input_shape=(10,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd')

    # Note: you have to use shuffle='batch' or False with HDF5Matrix
    model.fit(X_train, y_train, batch_size=32, shuffle='batch', verbose=False)
    # test that evalutation and prediction don't crash and
    # return reasonable results
    out_pred = model.predict(X_test, batch_size=32, verbose=False)
    out_eval = model.evaluate(X_test, y_test, batch_size=32, verbose=False)

    assert out_pred.shape == (50, 1), 'Prediction shape does not match'
    assert out_eval.shape == (), 'Shape of evaluation does not match'
    assert out_eval > 0, (
        'Evaluation value does not meet criteria: {}'.format(out_eval))

    # test slicing for shortened array
    assert len(X_train[0:]) == len(X_train), 'Incorrect shape for sliced data'

    # test __getitem__
    with pytest.raises(IndexError):
        X_train[1000]
    with pytest.raises(IndexError):
        X_train[1000:1001]
    with pytest.raises(IndexError):
        X_train[[1000, 1001]]
    with pytest.raises(IndexError):
        X_train[six.moves.range(1000, 1001)]
    with pytest.raises(IndexError):
        X_train[np.array([1000])]
    with pytest.raises(TypeError):
        X_train[None]
    assert (X_train[0] == X_train[:1][0]).all()
    assert (X_train[[0, 1]] == X_train[:2]).all()
    assert (X_train[np.array([0, 1])] == X_train[:2]).all()

    # test normalizer
    normalizer = lambda x: x + 1
    normalized_X_train = HDF5Matrix(h5_path, 'my_data', start=0, end=150,
                                    normalizer=normalizer)
    assert np.isclose(normalized_X_train[0][0], X_train[0][0] + 1)

    # test resizing normalizer
    normalizer_rs = lambda x: x[:, ::2]
    normalized_rs_X_train = HDF5Matrix(h5_path, 'my_data', start=0, end=150,
                                       normalizer=normalizer_rs)
    assert (normalized_rs_X_train.shape[1] == 5)

    # test dtype changing normalizer
    normalizer_dtype = lambda x: x.astype(np.uint8)
    normalized_dtype_X_train = HDF5Matrix(h5_path, 'my_data', start=0, end=150,
                                          normalizer=normalizer_dtype)
    assert (normalized_dtype_X_train.dtype == np.uint8)

    os.remove(h5_path)


def test_ask_to_proceed_with_overwrite():
    with patch('six.moves.input') as mock:
        mock.return_value = 'y'
        assert ask_to_proceed_with_overwrite('/tmp/not_exists')

        mock.return_value = 'n'
        assert not ask_to_proceed_with_overwrite('/tmp/not_exists')


def test_H5Dict_attrs():
    _, h5_path = tempfile.mkstemp('.h5')

    # test both HDF5 and dict implementations
    paths = [h5_path, dict()]

    for path in paths:
        f = H5Dict(path, mode='w')

        # str
        f['x'] = 'abcd'
        f['x2'] = u'abcd'

        # list<bytes>
        f['y'] = [b'efg', b'hij', b'klmn']
        f['y2'] = (b'asd', b'sdf', b'dfg')

        # ndarray
        array = np.random.random((4, 5, 512))
        f['z'] = array

        f.close()
        del f

        f = H5Dict(path, mode='r')

        assert f['x'] == 'abcd'
        assert f['x2'] == u'abcd'
        assert f['y'] == [b'efg', b'hij', b'klmn']
        assert list(f['y2']) == [b'asd', b'sdf', b'dfg']
        assert_allclose(f['z'], array)

        f.close()
    os.remove(h5_path)


def test_H5Dict_groups():
    _, h5_path = tempfile.mkstemp('.h5')

    # test both HDF5 and dict implementations
    paths = [h5_path, dict()]

    for path in paths:
        f = H5Dict(path, mode='w')

        group1 = f['group1']
        group2 = group1['group2']

        group2['x'] = 'abcd'

        group3 = group2['group3']
        group3['y'] = [b'efg', b'hij', b'klmn']

        group4 = group3['group4']
        array = np.random.random((4, 5, 512))
        group4['z'] = array

        f.close()

        f = H5Dict(path, mode='r')

        assert 'group1' in f
        group1 = f['group1']

        assert 'group2' in group1
        group2 = group1['group2']
        assert group2['x'] == 'abcd'

        assert 'group3' in group2
        group3 = group2['group3']
        assert group3['y'] == [b'efg', b'hij', b'klmn']

        assert 'group4' in group3
        group4 = group3['group4']
        assert_allclose(group4['z'], array)

        f.close()
    os.remove(h5_path)


def test_H5Dict_accepts_pathlib_Path():
    """GitHub issue: 11459"""
    _, h5_path = tempfile.mkstemp('.h5')

    f = H5Dict(Path(h5_path), mode='w')
    f['x'] = 'abcd'
    f.close()
    del f

    f = H5Dict(Path(h5_path), mode='r')
    assert f['x'] == 'abcd'
    f.close()

    os.remove(h5_path)


@contextmanager
def temp_filename(suffix):
    """Context that returns a temporary filename and deletes the file on exit if
    it still exists (so that this is not forgotten).
    """
    _, temp_fname = tempfile.mkstemp(suffix=suffix)
    yield temp_fname
    if os.path.exists(temp_fname):
        os.remove(temp_fname)


def test_save_to_binary_h5py_direct_to_file():
    data = np.random.random((3, 5))

    def save_function(h5file_):
        h5file_['data'] = data

    with temp_filename('.h5') as fname:
        with open(fname, 'wb') as f:
            save_to_binary_h5py(save_function, f)

        with h5py.File(fname) as h5file:
            data_rec = h5file['data'][:]

    assert_array_equal(data_rec, data)


def test_save_to_binary_h5py_to_bytes_io():
    data = np.random.random((3, 5))

    def save_function(h5file_):
        h5file_['data'] = data

    file_like = io.BytesIO()
    save_to_binary_h5py(save_function, file_like)

    file_like.seek(0)

    with temp_filename('.h5') as fname:
        with open(fname, 'wb') as f:
            f.write(file_like.read())

        with h5py.File(fname) as h5file:
            data_rec = h5file['data'][:]

    assert_array_equal(data_rec, data)


def test_load_from_binary_h5py_direct_from_file():
    data = np.random.random((3, 5))

    def load_function(h5file_):
        return h5file_['data'][:]

    with temp_filename('.h5') as fname:
        with h5py.File(fname, 'w') as h5file:
            h5file['data'] = data

        with open(fname, 'rb') as f:
            data_rec = load_from_binary_h5py(load_function, f)

    assert_array_equal(data_rec, data)


def test_load_from_binary_h5py_from_bytes_io():
    data = np.random.random((3, 5))

    def load_function(h5file_):
        return h5file_['data'][:]

    with temp_filename('.h5') as fname:
        with h5py.File(fname, 'w') as h5file:
            h5file['data'] = data

        file_like = io.BytesIO()
        with open(fname, 'rb') as f:
            file_like.write(f.read())

    file_like.seek(0)
    data_rec = load_from_binary_h5py(load_function, file_like)

    assert_array_equal(data_rec, data)


def test_save_load_binary_h5py():

    data1 = np.random.random((3, 5))
    data2 = np.random.random((2, 3, 5))
    attr = 1
    datas = [data1, data2, attr]

    def save_function(h5file_):
        h5file_['data1'] = data1
        h5file_['subgroup/data2'] = data2
        h5file_['data1'].attrs['attr'] = attr

    def load_function(h5file_):
        d1 = h5file_['data1'][:]
        d2 = h5file_['subgroup/data2'][:]
        a = h5file_['data1'].attrs['attr']
        return d1, d2, a

    file_like = io.BytesIO()
    save_to_binary_h5py(save_function, file_like)
    file_like.seek(0)
    datas_rec = load_from_binary_h5py(load_function, file_like)
    for d_rec, d in zip(datas_rec, datas):
        assert_array_equal(d_rec, d)


if __name__ == '__main__':
    pytest.main([__file__])
