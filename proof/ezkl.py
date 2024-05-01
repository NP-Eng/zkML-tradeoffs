import json
import torch    
import ezkl
import time
import sklearn as sk
from typing import Union

RESOURCES_DIR = "proof/resources/ezkl/"
INPUT = RESOURCES_DIR + "input.json"
SETTINGS = RESOURCES_DIR + "settings.json"
CALIBRATION = RESOURCES_DIR + "calibration.json"
WITNESS = RESOURCES_DIR + "witness.json"
COMPILED_MODEL = RESOURCES_DIR + "compiled_model.json"
VK = RESOURCES_DIR + "vk.json"
PK = RESOURCES_DIR + "pk.json"
PROOF = RESOURCES_DIR + "proof.pf"

def create_files():
    open(INPUT, 'w').close()
    open(SETTINGS, 'w').close()
    open(CALIBRATION, 'w').close()
    open(WITNESS, 'w').close()
    open(COMPILED_MODEL, 'w').close()
    open(VK, 'w').close()
    open(PK, 'w').close()
    open(PROOF, 'w').close()

def save_input(model: Union[torch.nn.Module, sk.base.BaseEstimator], sample: torch.Tensor, shape: tuple):
    if isinstance(model, sk.base.BaseEstimator):
        torch_out = model.predict(sample.detach().numpy())
        out = [o.reshape([-1]).tolist() for o in torch_out]
    else:
        model.eval()
        torch_out = model(sample)
        out = [o.detach().numpy().reshape([-1]).tolist() for o in torch_out]
    in_data = sample.detach().numpy().reshape([-1]).tolist()

    data = dict(
        input_shapes=[shape],
        input_data=[in_data],
        output_data=out
    )

    # Serialize data into file:
    json.dump(data, open(INPUT, 'w'))

def gen_settings(model_onnx_file: str):
    assert ezkl.gen_settings(
        model_onnx_file,
        SETTINGS
    ) == True

def gen_calibration(shape, model_onnx_file: str):
    data_array = (torch.randn(20, *shape).detach().numpy()).reshape([-1]).tolist()
    data = dict(input_data = [data_array])

    # Serialize data into file:
    json.dump(data, open(CALIBRATION, 'w'))

    assert ezkl.calibrate_settings(INPUT, model_onnx_file, SETTINGS, "resources") == True

def compile_model(model_onnx_file: str):
    assert ezkl.compile_circuit(model_onnx_file, COMPILED_MODEL, SETTINGS) == True
    assert ezkl.get_srs(SETTINGS) == True
    ezkl.gen_witness(INPUT, COMPILED_MODEL, WITNESS)
    
def setup(model: Union[torch.nn.Module, sk.base.BaseEstimator], model_onnx_file: str, sample: torch.Tensor, shape: tuple):

    # EZKL setup helper functions
    create_files()
    save_input(model, sample, shape)
    gen_settings(model_onnx_file)
    gen_calibration(shape, model_onnx_file)
    compile_model(model_onnx_file)

    assert ezkl.setup(
        COMPILED_MODEL,
        VK,
        PK,
    ) == True

def prove():
    _= ezkl.prove(
        WITNESS,
        COMPILED_MODEL,
        PK,
        PROOF,
        "single",
    )

def verify():
    assert ezkl.verify(
        PROOF,
        SETTINGS,
        VK,
    ) == True

def bench_ezkl_single_round(model: Union[torch.nn.Module, sk.base.BaseEstimator], model_onnx_file: str, sample: torch.Tensor, shape: tuple):
    setup_time = -time.time()
    setup(model, model_onnx_file, sample, shape)
    setup_time += time.time()
    
    prove_time = -time.time()
    prove()
    prove_time += time.time()

    verify_time = -time.time()
    verify()
    verify_time += time.time()

    return setup_time, prove_time, verify_time

import contextlib

def bench_ezkl(model: Union[torch.nn.Module, sk.base.BaseEstimator], model_onnx_file: str, sample: torch.Tensor, shape: tuple, rounds: int = 1):
    setup_time, prove_time, verify_time = 0, 0, 0
    for _ in range(rounds):
        with contextlib.redirect_stderr(None):
            s, p, v = bench_ezkl_single_round(model, model_onnx_file, sample, shape)
        setup_time += s
        prove_time += p
        verify_time += v

    setup_time /= rounds
    prove_time /= rounds
    verify_time /= rounds

    print(f"Setup time: {str(setup_time)[:5]} [s]")
    print(f"Prover time: {str(prove_time)[:5]} [s]")
    print(f"Verifier time: {str(verify_time)[:5]} [s]")
    
    return setup_time, prove_time, verify_time