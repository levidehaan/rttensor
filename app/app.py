import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import tensorrt_llm
from termcolor import cprint
from tensorrt_llm._utils import mpi_barrier, mpi_rank, mpi_world_size
from tensorrt_llm.executor import SamplingParams
from mpi4py import MPI
import uvicorn
import torch
from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import JSONResponse, Response, StreamingResponse
from tensorrt_llm.bindings.executor import ExecutorConfig, PromptTuningConfig, ModelType, OutputConfig
from tensorrt_llm.executor import ExecutorBindingsWorker, GenerationResult
from fastapi.responses import JSONResponse, StreamingResponse, UJSONResponse
from tensorrt_llm.hlapi import tokenizer
from transformers import AutoTokenizer
from typing_extensions import AsyncGenerator
from uvicorn.config import LOGGING_CONFIG
from custom_logging.custom_logging import CustomizeLogger
from pydantic import BaseModel

tensorrt_llm.logger.set_level('info')
import logging

logger = logging.getLogger(__name__)

config_path = Path(__file__).with_name("logging_config.json")

TP_SIZE = 2
PP_SIZE = 1  # Assuming no pipeline parallelism, update if necessary
TIMEOUT_KEEP_ALIVE = 5  # seconds
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds
app = FastAPI()
logger = CustomizeLogger.make_logger(config_path)
app.logger = logger

executor: Optional[ExecutorBindingsWorker] = None


class StreamingResponseWithTime(StreamingResponse):
    def __init__(self, *args, **kwargs):
        self.start_time = kwargs.pop('start_time', None)
        self.logger = kwargs.pop('logger', None)
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.start_time and self.logger:
                self.logger.info(
                    "Time took to process the request and return response is {} sec".format(
                        time.time() - self.start_time)
                )


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    if isinstance(response, StreamingResponse):
        return StreamingResponseWithTime(
            response.body_iterator,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
            start_time=start_time,
            logger=request.app.logger
        )

    request.app.logger.info(
        "Time took to process the request and return response is {} sec".format(time.time() - start_time)
    )

    return response


@app.get("/stats")
async def stats() -> Response:
    assert executor is not None
    return JSONResponse(json.loads(await executor.aget_stats()))


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


def sanitize_json_string(json_str: str) -> str:
    # Remove non-ASCII characters
    json_str = re.sub(r'[^\x20-\x7E]+', '', json_str)
    # Replace single quotes with double quotes to form valid JSON
    # json_str = json_str.replace("'", '"')
    # Ensure proper escaping of quotes inside the JSON
    # json_str = re.sub(r'(?<!\\)"', r'\\"', json_str)
    return json_str


@app.post("/generate")
async def generate(request: Request) -> Response:
    if executor is None:
        return JSONResponse({"error": "Executor is not initialized"}, status_code=503)
    executor.block_subordinates()

    try:
        request_body = await request.body()
        # cprint(f"Request body: {request_body}", "green")
        sanitized_json_str = sanitize_json_string(request_body.decode('utf-8'))
        # cprint(f"Sanitized JSON: {sanitized_json_str}", "green")
        request_dict = json.loads(sanitized_json_str)
    except Exception as e:
        cprint(f"Error: {e}", "red")
        return JSONResponse({"error": "Error in request"}, status_code=400)

    # cprint(f"Request: {request_dict}", "green")
    # cprint(f"request dir: {dir(request)}", "blue")

    stop_words_list = request_dict.get("stop")

    # if not set join the default stop words
    if stop_words_list:
        stop_words_list = [["<|eot_id|>"] + stop_words_list]
    else:
        stop_words_list = [["<|eot_id|>"]]
    # init tokenizer class tokenizer

    cprint(f"Stop words list: {stop_words_list}", "green")

    tkn = AutoTokenizer.from_pretrained(args.hf_model_dir)
    stop_words = tensorrt_llm.runtime.decode_words_list(stop_words_list, tkn)

    # cprint(f"\n\nstopWords: {stop_words[0]}\n\n", "green")

    sampling_params = SamplingParams(
        random_seed=request_dict.get("random_seed", 501),
        max_new_tokens=request_dict.get("max_tokens", 16000),
        temperature=request_dict.get("temperature", 0.5),
        top_p=request_dict.get("top_p", 0.89),
        top_k=request_dict.get("top_k", 3),
        repetition_penalty=request_dict.get("repetition_penalty", 1.05),
        frequency_penalty=request_dict.get("frequency_penalty", 0.3),
        presence_penalty=request_dict.get("presence_penalty", 1.01),
        early_stopping=True,
        stop_words=stop_words[0]
    )

    prompt = request_dict.pop("prompt", "")

    messages = [
        {"role": "system", "content": "You are a super intelligent AI who always answers to the best of his ability."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tkn.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(torch.int32).numpy()

    # cprint(f"Prompt INT: {input_ids}", "green")
    with torch.no_grad():
        output = executor.generate(input_ids, sampling_params=sampling_params)
    # cprint(f"Output: {output}", "green")
    cprint(f"Prompt: {prompt}\n\n", "blue", "on_grey")

    # cprint(f"Output DIR: {dir(output)}", "blue")

    output_decoded = tkn.decode(output[0].token_ids)

    # strip "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" from the output
    output_decoded = output_decoded.replace("<|eot_id|>", "")

    cprint(f"Output: {output_decoded}", "cyan", "on_grey")

    return JSONResponse({"texts": output_decoded})


async def timeout_checker(last_chunk_time, INACTIVITY_TIMEOUT=600):
    while True:
        await asyncio.sleep(1)  # Check every second
        if time.time() - last_chunk_time > INACTIVITY_TIMEOUT:
            cprint("Timeout reached, breaking the loop", "red", "on_grey", attrs=['bold'])
            break


def sequence_exists(long_list, target_sequence):
    # Convert the list and target sequence to numpy arrays
    long_list_array = np.array(long_list)
    target_sequence_array = np.array(target_sequence)

    # Check if the long list is long enough
    if long_list_array.size < target_sequence_array.size:
        return False

    # Create a rolling window view of the long list
    shape = (long_list_array.size - target_sequence_array.size + 1, target_sequence_array.size)
    strides = (long_list_array.strides[0], long_list_array.strides[0])
    rolling_view = np.lib.stride_tricks.as_strided(long_list_array, shape=shape, strides=strides)

    # Check if any row in the rolling view matches the target sequence
    matches = np.all(rolling_view == target_sequence_array, axis=1)

    return np.any(matches)


@app.post("/v2/chat/completions")
async def generate(request: Request) -> Response:
    assert executor is not None
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    cprint(f"Request: {request_dict}", "green")
    prompt = request_dict['messages'][0]['content']
    streaming = request_dict.pop("streaming", True)

    stop_words_list = request_dict.get("stop")

    # if not set join the default stop words
    if not stop_words_list:
        stop_words_list = ["\n\n\n\n\n"]

    stop_words_list = [["\nObservation: "] + stop_words_list]
    # init tokenizer class tokenizer
    cprint(f"Stop words list: {stop_words_list}", "magenta", "on_red", attrs=['bold'])

    tkn = executor.tokenizer
    end_token = tkn.encode("<|eot_id|>", add_special_tokens=False)[0]
    tkn.eos_token_id = end_token
    # iterate over stop words and encode them
    stop_words = []
    for stop in stop_words_list[0]:
        stop_words.append(tkn.encode(stop, add_special_tokens=False))

    stop_words.append([633, 38863, 367])
    stop_words.append([633])

    cprint(f"Stop Word Tokens : {stop_words} \n end_token: {end_token}", "green", "on_grey", attrs=['bold'])

    for stop in stop_words:
        cprint(f"the actual words decoded: {tkn.decode(stop)}", "green", "on_grey", attrs=['bold'])

    sampling_params = SamplingParams(
        random_seed=request_dict.get("random_seed", 501),
        max_new_tokens=request_dict.get("max_tokens", 16000),
        temperature=request_dict.get("temperature", 0.5),
        top_p=request_dict.get("top_p", 1),
        top_k=request_dict.get("top_k", 3),
        repetition_penalty=request_dict.get("repetition_penalty", 1.05),
        frequency_penalty=request_dict.get("frequency_penalty", 0.3),
        presence_penalty=request_dict.get("presence_penalty", 1.01),
        early_stopping=True,
        stop_words=stop_words,
        end_id=end_token
    )

    messages = [
        {"role": "system", "content": "You are a super intelligent AI that follows directions perfectly."},
        {"role": "user", "content": prompt},
    ]

    input_ids = tkn.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(torch.int32).numpy()



    # cprint(f"Prompt INT: {input_ids}", "green")
    with torch.no_grad():
        promise = executor.generate_async(input_ids, streaming, sampling_params=sampling_params)

    # assert isinstance(promise, GenerationResult)
    # cprint(f"type of promise: {type(promise)}", "green")

    async def stream_results() -> AsyncGenerator[bytes, None]:
        last_text = ""
        diff = ""
        # Handle each GenerationResult in the list
        for generation_result in promise:
            # cprint(f"Processing GenerationResult: {generation_result}", "green")
            while not generation_result.done():
                # cprint(f"Waiting for {generation_result}\n calling step", "green")
                await generation_result.aresult_step()


                # Decode the new text
                new_txt = tkn.decode(generation_result.token_ids, skip_special_tokens=False)

                if not new_txt:
                    continue
                if last_text != "":
                    # Check if the stop token is in the result
                    diff = new_txt[len(last_text):]
                else:
                    diff = new_txt

                last_text = new_txt

                new_chunk = {
                    "id": "chatcmpl-41c505e4-5233-4cd0-8be4-5ef5aa0691b6",
                    "model": "UBER",
                    "created": time.time(),
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {
                                "content": diff
                            },
                            "index": 0,
                            "finish_reason": "none"
                        }
                    ]
                }

                # cprint(f"Yielding new chunk {new_chunk} of {generation_result.token_ids}", "green")
                yield "data: " + json.dumps(new_chunk) + "\n\n"

        cprint("Done streaming", "green", "on_grey", attrs=['bold'])
        yield "data: [DONE]\n\n"

        # if streaming:
        cprint(f"Prompt: {prompt}\n\n", "blue", "on_white", attrs=['dark'])
        cprint(f"Output: {last_text}", "magenta", "on_grey", attrs=['bold', 'underline'])

    return StreamingResponse(stream_results())

    # # Non-streaming case
    # await promise.aresult()
    # return JSONResponse({"text": promise.text})


# a GET version of the /v2/chat/completions endpoint
@app.get("/v1/completions")
async def generate(
        prompt: str,
        model: Optional[str] = Query("UBER"),
        stop_words: Optional[list] = Query(None),
        temperature: Optional[float] = Query(0.5),
        max_tokens: Optional[int] = Query(300),
        top_p: Optional[float] = Query(0.89),
        top_k: Optional[int] = Query(50),
        repetition_penalty: Optional[float] = Query(1.05),
        frequency_penalty: Optional[float] = Query(0.2),
        presence_penalty: Optional[float] = Query(0.6),
        streaming: Optional[bool] = Query(True)
) -> Response:
    stop_words_list = stop_words or [["<|end|>", "\n\n\n\n\n\n"]]
    tknz = tensorrt_llm.hlapi.tokenizer
    tkn = tknz.TransformersTokenizer.from_pretrained(args.hf_model_dir)

    stop_words = tensorrt_llm.runtime.decode_words_list(stop_words_list, tkn)

    cprint(
        f"\n\nstopWords: {stop_words[0]}\n\n", "green")

    sampling_params = SamplingParams(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        early_stopping=True,
        stop_words=stop_words[0]
    )

    cprint(
        f"Temp: {temperature}\n top_p {top_p}\n top_k: {top_k}\n repition_penalty: {repetition_penalty}, prompt: {prompt}, model: {model}",
        "green")

    # Simulated executor and promise for demonstration purposes

    output = executor.generate(prompt, sampling_params=sampling_params)

    cprint(f"Prompt: {prompt}\n\n", "blue", "on_white")
    cprint(f"Output: {output.text}", "green", "on_dark_grey")

    return JSONResponse({"texts": output.text})


def initialize_executor(args, mpi_rank):
    global executor
    cprint(f"Initializing executor for rank {mpi_rank}", "green")
    executor_config = ExecutorConfig(max_beam_width=args.max_beam_width)
    torch.cuda.set_device(mpi_rank)

    hf_model_dir = args.hf_model_dir
    engine_dir = args.engine_dir
    tokenizer_dir = hf_model_dir

    mpi_barrier()
    tensorrt_llm.logger.warning(f"Build finished for rank {mpi_rank}")
    executor = ExecutorBindingsWorker(engine_dir, tokenizer_dir, executor_config)
    cprint("Executor created", "green")


async def start_server(args):
    # add timings to logs to see how long a request took
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "AI SERVER - %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="debug",
                            timeout_keep_alive=TIMEOUT_KEEP_ALIVE, log_config=LOGGING_CONFIG)
    server = uvicorn.Server(config)
    await server.serve()


async def main(args):
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    # cprint(f"MPI rank: {mpi_rank}, world size: {world_size}", "green")
    if world_size != TP_SIZE * PP_SIZE:
        raise ValueError("MPI world size must be equal to TP_SIZE * PP_SIZE")

    if mpi_rank < TP_SIZE:
        initialize_executor(args, mpi_rank)

    if mpi_rank == 0:
        await start_server(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir", type=str, required=True,
                        help="Read the model data and tokenizer from this directory")
    parser.add_argument("--engine_dir", type=str, required=True, help="Directory to save and load the engine.")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_beam_width", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()
    cprint(f"""
------------------------------------------------------------------------------------------
|||          STARTING AI INFERENCE SERVER: AUTOMATON 501  - ON GPU {MPI.COMM_WORLD.Get_rank()}    |||
------------------------------------------------------------------------------------------""", "magenta", "on_red",
           attrs=['bold', 'blink'])
    # cprint(f"MPI COMM WORLD SIZE: {dir(MPI.COMM_WORLD)}", "green")
    # cprint(f"MPI COMM WORLD SIZE: {MPI.COMM_WORLD.Get_size()}", "green")
    asyncio.run(main(args))
