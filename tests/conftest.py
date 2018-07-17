
import pytest
import joblib
from os import path


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="run slow tests")

    parser.addoption("--rundisplay", action="store_true",
                     help="run display tests")


def pytest_runtest_setup(item):
    if 'slow' in item.keywords and not item.config.getvalue("runslow"):
        pytest.skip("need --runslow option to run")

    if 'display' in item.keywords and not item.config.getvalue("rundisplay"):
        pytest.skip("need --rundisplay option to run")


@pytest.fixture(scope='session')
def root_path():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '..'))


@pytest.fixture(scope='session')
def titanic(root_path):
    file = path.join(root_path, 'pdpbox', 'datasets', 'test_titanic.pkl')
    return joblib.load(file)


@pytest.fixture(scope='module')
def titanic_data(titanic):
    return titanic['data']


@pytest.fixture(scope='module')
def titanic_features(titanic):
    return titanic['features']


@pytest.fixture(scope='module')
def titanic_target(titanic):
    return titanic['target']


@pytest.fixture(scope='module')
def titanic_model(titanic):
    return titanic['xgb_model']


@pytest.fixture(scope='session')
def ross(root_path):
    file = path.join(root_path, 'pdpbox', 'datasets', 'test_ross.pkl')
    return joblib.load(file)


@pytest.fixture(scope='module')
def ross_data(ross):
    return ross['data']


@pytest.fixture(scope='module')
def ross_features(ross):
    return ross['features']


@pytest.fixture(scope='module')
def ross_target(ross):
    return ross['target']


@pytest.fixture(scope='module')
def ross_model(ross):
    return ross['rf_model']


@pytest.fixture(scope='session')
def otto(root_path):
    file = path.join(root_path, 'pdpbox', 'datasets', 'test_otto.pkl')
    return joblib.load(file)


@pytest.fixture(scope='module')
def otto_data(otto):
    return otto['data']


@pytest.fixture(scope='module')
def otto_features(otto):
    return otto['features']


@pytest.fixture(scope='module')
def otto_target(otto):
    return otto['target']


@pytest.fixture(scope='module')
def otto_model(otto):
    return otto['rf_model']