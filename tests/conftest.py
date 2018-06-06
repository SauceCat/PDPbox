
import pytest
import joblib
from pathlib import Path


@pytest.fixture(scope='session')
def titanic():
    root_path = Path(__file__).resolve().parents[1]
    file = root_path / 'pdpbox' / 'datasets' / 'test_titanic.pkl'
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
def ross():
    root_path = Path(__file__).resolve().parents[1]
    file = root_path / 'pdpbox' / 'datasets' / 'test_ross.pkl'
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
def otto():
    root_path = Path(__file__).resolve().parents[1]
    file = root_path / 'pdpbox' / 'datasets' / 'test_otto.pkl'
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