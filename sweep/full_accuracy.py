"""Full-corpus 2B baseline reproduction and gated distillation confirmation.

Run with python -m sweep.full_accuracy. --smoke uses tiny data and never permits promotion.
All subprocesses run sequentially; only the current experiment's files and jobs are controlled.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

from lars.train.accuracy import capture, distill, load_capture, record_key, state_key
from lars.train.data import read_records, write_records
from sweep import analyze as scoring
from sweep.confirmation import (ROOT, HOLD, POLICY, prepare, pilot_analysis, canonical_sdk,
    apply_calibration, summary, certificates, paired_interval, promotion, save_dataset, row_keys)
from sweep.accuracy_ladder import digest
from sweep.experiment import Runner, atomic_json, fingerprint

TEMPLATE = ROOT / "sweep/templates/Qwen3.5-2B.json"
DEV = ROOT / "target/release/lars-dev"
TOOLS = ROOT / "target/llama-tools/bin"
LLAMA = ROOT / "crates/lars-llama-sys/llama.cpp"
CONVERT = ROOT / "sweep/.venv-convert/bin/python"
REFERENCE = ROOT / "models/reference-2b/packs/lars-2b-dev/0.0.1"


def worker(task):
    op = task["operation"]
    if op == "export":
        adapter, output = Path(task["adapter"]), Path(task["output"])
        history = json.loads(adapter.with_name(adapter.name+".history.json").read_text())
        final = adapter.with_name(adapter.name+".steps") / f"step-{history[-1]['step']}"
        output.mkdir(parents=True, exist_ok=True)
        for filename in ("adapters.safetensors", "adapter_config.json"):
            shutil.copyfile(final / filename, output / filename)
        atomic_json(output / "selection.json", {"rule": POLICY["checkpoint"], "step": history[-1]["step"], "source": str(final)})
    elif op == "fuse":
        output = Path(task["output"])
        partial = output.with_name(output.name+".part")
        if partial.exists():
            shutil.rmtree(partial)  # only this worker's incomplete output
        subprocess.run([sys.executable,"-m","mlx_lm","fuse","--model",task["model"],"--adapter-path",task["adapter"],
                        "--save-path",str(partial)], check=True)
        base = Path(task["model"])
        for filename in ("tokenizer.json","tokenizer_config.json","vocab.json","merges.txt","special_tokens_map.json","chat_template.jinja"):
            if (base / filename).exists():
                shutil.copyfile(base / filename, partial / filename)
        from sweep.unshift_norms import main as unshift
        unshift(partial)
        if output.exists():
            output.rename(output.with_name(output.name+f".previous-{time.time_ns()}"))
        partial.rename(output)
    elif op == "imatrix-text":
        from transformers import AutoTokenizer
        from lars.labels import assign_labels
        from lars.render import compose_prompt, render_content
        from lars.train.features import _render
        from sweep.imatrix_text import template_fn
        tokenizer = AutoTokenizer.from_pretrained(task["model"], local_files_only=True)
        labels = assign_labels(160, lambda x: len(tokenizer.encode(" "+x, add_special_tokens=False)) == 1)
        records = list(read_records(task["records"]))
        random.Random(1729).shuffle(records)
        fn = template_fn(str(TEMPLATE))
        prompts = []
        for r in records[:task["limit"]]:
            body, _, _ = _render(r, labels, list(range(len(r.options))))
            prefix, suffix = compose_prompt(fn, render_content(r.state), body)
            prompts.append(prefix+suffix)
        Path(task["output"]).write_text("\n".join(prompts)+"\n")
    elif op == "canonical":
        canonical_sdk(Path(task["raw"]), Path(task["data"]), task["dataset"], Path(task["output"]))
    elif op == "mlx-capture":
        from lars.backends.mlx import MLXBackend
        output = Path(task["output"])
        records = list(read_records(task["records"]))
        capture(MLXBackend(task["model"], adapter=task.get("adapter")), records, output,
                max_tokens=8192 if task.get("wire") else 2048)
        rows = load_capture(output)
        if task.get("wire"):
            metadata = [json.loads(l) for l in Path(task["wire"]).read_text().split("\n") if l.strip()]
            for meta in metadata:
                rows[meta["record_key"]]["hard_index"] = meta["hard_index"]
                if "p" not in rows[meta["record_key"]]:
                    raise ValueError("long evaluation row cannot be silently skipped")
        for r in records:
            rows[record_key(r)]["keys"] = row_keys(r)
        output.write_text("".join(json.dumps(row)+"\n" for row in rows.values()))
    elif op == "fit-calibration":
        rows = load_capture(Path(task["capture"]))
        fitted = scoring.fit([scoring.Row(r["source"],r["kind"],r["keys"],np.asarray(r["logits"]),r["hard_index"],
                                          np.asarray(r["target"])) for r in rows.values()])
        atomic_json(task["output"], {"temperatures": fitted["temperature"], "platt": fitted["platt"],
                                     "fitted_on": task["capture"]})
    elif op == "distill":
        teacher = apply_calibration(Path(task["teacher_dev"]), json.loads(Path(task["teacher_calibration"]).read_text()))
        student = apply_calibration(Path(task["student_dev"]), json.loads(Path(task["student_calibration"]).read_text()))
        a, b = summary(teacher)["per_source"], summary(student)["per_source"]
        allowed = {s for s in a if a[s]["accuracy"] > b[s]["accuracy"]}
        records = list(read_records(task["records"]))
        if task.get("pool_only"):
            selected = [r for r in records if r.source in allowed]
            write_records(selected,task["output"])
            atomic_json(Path(task["output"]).with_suffix(".manifest.json"), {"allowed_sources":sorted(allowed),"rows":len(selected)})
        else:
            teacher_train = apply_calibration(Path(task["teacher_train"]), json.loads(Path(task["teacher_calibration"]).read_text()), allow_skipped=True)
            # The targets remain gold for every other source. The baseline's own probabilities
            # determine the priority among difficult examples; no test result enters this step.
            student_train = load_capture(Path(task["student_train"]))
            output, manifest = distill(records, teacher_train, student_train, [], mix=.5, per_source=500, allowed_sources=allowed)
            if [record_key(r) for r in output] != [record_key(r) for r in records]:
                raise ValueError("distillation changed the training population")
            write_records(output, task["output"])
            atomic_json(Path(task["output"]).with_suffix(".manifest.json"), manifest)
    elif op == "mlx-prompts":
        from lars.backends.mlx import MLXBackend
        import mlx.core as mx
        backend = MLXBackend(task["model"], adapter=task["adapter"])
        rows = [json.loads(l) for l in Path(task["prompts"]).read_text().split("\n") if l.strip()]
        output = []
        for row in rows:
            ids = [backend._label_id(label) for label in row["labels"]]
            logits = backend.model(mx.array([backend._encode(row["prompt"])]))[0,-1,mx.array(ids)].astype(mx.float32)
            z = np.asarray(logits,float); p=np.exp(z-z.max()); p/=p.sum()
            output.append({"p":p.tolist(),"target":row["target"]})
        atomic_json(task["output"], output)
    elif op == "emit-pack":
        output = Path(task["output"]); output.mkdir(parents=True,exist_ok=True)
        model = Path(task["model"])
        link = output / "model.gguf"
        if not link.exists():
            os.link(model,link)
        identity = {"name":task["name"],"version":"0.0.1","model_sha256":digest(model)}
        cal = json.loads(Path(task["calibration"]).read_text()) | {"pack":identity}
        atomic_json(output / "calibration.json",cal)
        manifest = {"format":1,"name":identity["name"],"version":identity["version"],"backend":"llamacpp",
            "model":{"file":"model.gguf","bytes":model.stat().st_size,"sha256":identity["model_sha256"],
                     "base":"Qwen/Qwen3.5-2B","license":"Apache-2.0","quantization":"Q4_K_M"},
            "template":json.loads(TEMPLATE.read_text()),"layout":"v1",
            "runtime":{"flash_attention":"off","llama_cpp":task["runtime"]},
            "calibration":{"file":"calibration.json","sha256":digest(output/"calibration.json"),
                           "fitted_on":"independent full-confirmation calibration split"},
            "evaluation":{"status":"pending confirmation; experimental pack"}}
        atomic_json(output/"pack.json",manifest)
    else:
        raise ValueError(f"unknown worker operation {op}")


class GateFailed(RuntimeError):
    pass


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--task",type=Path)
    parser.add_argument("--out",type=Path,default=ROOT/"sweep/out/accuracy-full")
    parser.add_argument("--model")
    parser.add_argument("--teacher")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    if args.task:
        worker(json.loads(args.task.read_text()));return
    if not args.model or not args.teacher:
        parser.error("--model and --teacher are required")
    os.chdir(ROOT);out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/"run.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config={"model":args.model,"teacher":args.teacher,"smoke":args.smoke,"out":str(out)}
        path=out/"config.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("output directory has a different configuration")
        atomic_json(path,config)
        if (out/"protocol.json").exists() and json.loads((out/"protocol.json").read_text()) != POLICY|{"smoke":args.smoke}:
            raise ValueError("protocol changed; use a new experiment directory")
        manifest=prepare(out,args.smoke)
        pilot_analysis(out/"pilot-breakdown.json")
        if args.prepare_only:
            print(json.dumps({k:v for k,v in manifest.items() if k not in {"inputs","outputs"}}));return
        try:
            run(args,out)
        except GateFailed as exc:
            atomic_json(out/"status.json",{"state":"stopped_at_gate","reason":str(exc),"finished":time.time()})
            print(f"Stopped at evidence gate: {exc}",flush=True)
        except Exception as exc:
            status=out/"status.json"
            previous=json.loads(status.read_text()) if status.exists() else {}
            atomic_json(status,previous|{"state":"failed","error":str(exc),"finished":time.time()})
            raise


def run(args,out):
    data=out/"data"
    env=dict(os.environ,HF_HOME=str(ROOT/".hf"),HF_HUB_OFFLINE="1",LARS_MLX_CACHE_GB="8",TOKENIZERS_PARALLELISM="false")
    runner=Runner(out,env)
    code=[Path(__file__),ROOT/"sweep/confirmation.py",ROOT/"sweep/experiment.py",
          *sorted((ROOT/"training/src/lars").rglob("*.py"))]
    training_code=sorted((ROOT/"training/src/lars").rglob("*.py"))
    atomic_json(out/"environment.json",{"python":sys.version,"packages":{d.metadata["Name"]:d.version for d in distributions()},
        "code":{str(p):digest(p) for p in code},"model":args.model,"teacher":args.teacher})
    py=[sys.executable,"-m","sweep.full_accuracy"]
    def job(stage_name,op,inputs,outputs,**kwargs):
        task=out/f"{stage_name}.task.json";atomic_json(task,{"operation":op,**{k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()}})
        runner.stage(stage_name,command=[*py,"--task",task],inputs=[task,*code,*inputs],outputs=outputs)
    def sdk(name,model,dataset):
        raw=out/f"{name}.raw.jsonl";wire=data/f"{dataset}.wire.jsonl"
        runner.stage(name,command=[DEV,"capture","--model",model,"--template",TEMPLATE,"--data",wire,"--out",raw],
                     inputs=[DEV,model,TEMPLATE,wire],outputs=[raw])
        target=out/f"{name}.jsonl"
        job(name+"-canonical","canonical",[raw,wire,data/f"{dataset}.records.jsonl"],[target],
            raw=raw,data=data,dataset=dataset,output=target)
        return target
    def mlx(name,model,dataset=None,records=None,adapter=None):
        target=out/f"{name}.jsonl"
        records=records or data/f"{dataset}.records.jsonl"
        params={"model":model,"records":records,"output":target}
        inputs=[records]
        if dataset:
            params["wire"]=data/f"{dataset}.wire.jsonl";inputs.append(params["wire"])
        if adapter:
            params["adapter"]=adapter;inputs.append(adapter)
        job(name,"mlx-capture",inputs,[target],**params)
        return target
    def train(name,records,seed):
        adapter=out/"training"/name
        options=[sys.executable,"-m","lars.train.lora","--model",args.model,"--records",records,data/"heldout.jsonl",
                 "--holdout-sources",",".join(HOLD),"--limit","100000","--epochs","1","--seed",str(seed),
                 "--batch-tokens","4096","--max-tokens","2048","--eval-every","500","--out",adapter]
        if args.smoke:
            options += ["--max-steps","2"]
        runner.stage(name,command=options,inputs=[*training_code,records,data/"heldout.jsonl",out/"protocol.json"],
                     outputs=[adapter.with_name(adapter.name+".history.json"),adapter.with_name(adapter.name+".steps")])
        exported=out/"adapters"/name
        job(name+"-export","export",[adapter.with_name(adapter.name+".history.json"),adapter.with_name(adapter.name+".steps")],
            [exported],adapter=adapter,output=exported)
        return exported
    def build(name,adapter):
        fused=out/"fused"/name
        job(name+"-fuse","fuse",[adapter],[fused],model=args.model,adapter=adapter,output=fused)
        bf16=out/f"{name}-BF16.gguf"
        runner.stage(name+"-convert",command=[CONVERT,LLAMA/"convert_hf_to_gguf.py",fused,"--outtype","bf16","--no-mtp","--outfile",bf16],
            inputs=[fused,LLAMA/"convert_hf_to_gguf.py",ROOT/"sweep/unshift_norms.py"],outputs=[bf16])
        text=out/"imatrix.txt"
        job("imatrix-text","imatrix-text",[data/"train.jsonl",TEMPLATE],[text],model=args.model,records=data/"train.jsonl",
            output=text,limit=20 if args.smoke else 1500)
        matrix=out/f"{name}.imatrix"
        runner.stage(name+"-imatrix",command=[TOOLS/"llama-imatrix","-m",bf16,"-f",text,"--parse-special","--chunks",
            "2" if args.smoke else "400","-ngl","99","-o",matrix],inputs=[TOOLS/"llama-imatrix",bf16,text],outputs=[matrix])
        quant=out/f"{name}-Q4_K_M.gguf"
        runner.stage(name+"-quantize",command=[TOOLS/"llama-quantize","--imatrix",matrix,bf16,quant,"Q4_K_M"],
            inputs=[TOOLS/"llama-quantize",matrix,bf16],outputs=[quant])
        parity=sdk(name+"-parity",quant,"parity")
        prompts=out/f"{name}-prompts.jsonl"
        runner.stage(name+"-prompts",command=[DEV,"prompts","--model",quant,"--template",TEMPLATE,"--data",data/"parity.wire.jsonl","--out",prompts],
                     inputs=[DEV,quant,TEMPLATE,data/"parity.wire.jsonl"],outputs=[prompts])
        mlx_parity=out/f"{name}-mlx-parity.json"
        job(name+"-mlx-parity","mlx-prompts",[adapter,prompts],[mlx_parity],model=args.model,adapter=adapter,prompts=prompts,output=mlx_parity)
        a=list(load_capture(parity).values());b=json.loads(mlx_parity.read_text())
        agreement=float(np.mean([np.argmax(x["p"])==np.argmax(y["p"]) for x,y in zip(a,b,strict=True)]))
        acc_a=np.mean([np.argmax(x["p"])==x["hard_index"] for x in a]);acc_b=np.mean([np.argmax(y["p"])==x["hard_index"] for x,y in zip(a,b,strict=True)])
        check={"rows":len(a),"label_agreement":agreement,"gguf_accuracy":float(acc_a),"mlx_accuracy":float(acc_b)}
        atomic_json(out/f"{name}-conversion-check.json",check)
        if not args.smoke and (abs(acc_a-acc_b)>POLICY["conversion"]["max_accuracy_difference"] or agreement<POLICY["conversion"]["min_label_agreement"]):
            raise GateFailed(f"{name}: GGUF conversion does not preserve MLX answers")
        return quant

    # The restored reference is evaluated with its recorded calibration, before any training.
    reference=sdk("reference-legacy",REFERENCE/"model.gguf","legacy")
    refcal=json.loads((REFERENCE/"calibration.json").read_text())
    refmetrics=summary(apply_calibration(reference,refcal))
    expected=json.loads((REFERENCE/"pack.json").read_text())["evaluation"]
    atomic_json(out/"reference-parity.json",{"actual":refmetrics,"expected":expected})
    if not args.smoke and (abs(refmetrics["macro_accuracy"]-expected["macro_accuracy"])>POLICY["reference_parity"]["max_accuracy_difference"] or
        abs(refmetrics["macro_brier"]-expected["macro_brier"])>POLICY["reference_parity"]["max_brier_difference"]):
        raise GateFailed("restored SDK reference does not reproduce its recorded metrics")

    captures={};calibrators={};certified={}
    for seed in ([0] if args.smoke else POLICY["seeds"]):
        base=train(f"baseline-s{seed}",data/"train.jsonl",seed)
        model=build(f"baseline-s{seed}",base)
        legacy=sdk(f"baseline-s{seed}-legacy",model,"legacy")
        # Reproduction-only calibration uses the historical first 200 validation rows.
        # Confirmation calibration and certificates use their separate frozen splits below.
        historical=out/"historical.validation.jsonl"
        historical_text=(data/"select.wire.jsonl").read_text() if args.smoke else \
            "".join(p.read_text() for p in sorted((ROOT/"data/evals").glob("*.validation.jsonl")))
        if not historical.exists() or historical.read_text()!=historical_text:
            historical.write_text(historical_text)
        legacy_cal=out/f"baseline-s{seed}-legacy-calibration.json"
        runner.stage(f"baseline-s{seed}-legacy-calibrate",command=[DEV,"calibrate","--model",model,"--template",TEMPLATE,"--data",historical,"--out",legacy_cal],
                     inputs=[DEV,model,TEMPLATE,historical],outputs=[legacy_cal])
        actual=summary(apply_calibration(legacy,json.loads(legacy_cal.read_text())))
        atomic_json(out/f"baseline-s{seed}-reproduction.json",{"actual":actual,"reference":refmetrics,"policy":POLICY["reproduction"]})
        if not args.smoke and (actual["macro_accuracy"]<refmetrics["macro_accuracy"]-.01 or actual["macro_brier"]>refmetrics["macro_brier"]+.015):
            raise GateFailed(f"baseline seed {seed} misses reference reproduction tolerance")
        baseline_cal=out/f"baseline-s{seed}-calibration.json"
        runner.stage(f"baseline-s{seed}-calibrate",command=[DEV,"calibrate","--model",model,"--template",TEMPLATE,"--data",data/"calibration.wire.jsonl","--out",baseline_cal],
                     inputs=[DEV,model,TEMPLATE,data/"calibration.wire.jsonl"],outputs=[baseline_cal])
        baseline_dev=sdk(f"baseline-s{seed}-dev",model,"dev")
        teacher_fit=mlx("teacher-calibration",args.teacher,dataset="calibration")
        teacher_cal=out/"teacher-calibration.json"
        job("teacher-fit","fit-calibration",[teacher_fit],[teacher_cal],capture=teacher_fit,output=teacher_cal)
        teacher_dev=mlx("teacher-dev",args.teacher,dataset="dev")
        params={"records":data/"train.jsonl","teacher_dev":teacher_dev,"student_dev":baseline_dev,
                "teacher_calibration":teacher_cal,"student_calibration":baseline_cal}
        pool=data/f"candidates-s{seed}.jsonl"
        job(f"pool-s{seed}","distill",list(params.values()),[pool,pool.with_suffix(".manifest.json")],pool_only=True,output=pool,**params)
        if not pool.stat().st_size:
            raise GateFailed(f"teacher has no development advantage over full baseline seed {seed}")
        teacher_train=mlx(f"teacher-candidates-s{seed}",args.teacher,records=pool)
        student_train=mlx(f"student-candidates-s{seed}",args.model,records=pool,adapter=base)
        distilled=data/f"distilled-s{seed}.jsonl"
        job(f"distill-s{seed}","distill",[*params.values(),teacher_train,student_train],
            [distilled,distilled.with_suffix(".manifest.json")],output=distilled,teacher_train=teacher_train,student_train=student_train,**params)
        if not json.loads(distilled.with_suffix(".manifest.json").read_text())["distilled"]:
            raise GateFailed("no training targets qualified for distillation")
        candidate=train(f"distilled-s{seed}",distilled,seed)
        candidate_model=build(f"distilled-s{seed}",candidate)
        candidate_cal=out/f"distilled-s{seed}-calibration.json"
        runner.stage(f"distilled-s{seed}-calibrate",command=[DEV,"calibrate","--model",candidate_model,"--template",TEMPLATE,"--data",data/"calibration.wire.jsonl","--out",candidate_cal],
                     inputs=[DEV,candidate_model,TEMPLATE,data/"calibration.wire.jsonl"],outputs=[candidate_cal])
        for recipe,weights,cal in (("baseline",model,baseline_cal),("distilled",candidate_model,candidate_cal)):
            name=f"{recipe}-s{seed}";calibrators[name]=json.loads(cal.read_text())
            captures[name]={}
            for dataset in ("certificate","confirmation","typed-regression","ood","open-confirmation"):
                path=sdk(name+"-"+dataset,weights,dataset)
                captures[name][dataset]=apply_calibration(path,calibrators[name])
            certified[name]=certificates(captures[name]["certificate"],captures[name]["confirmation"])
            pack=out/"packs"/name
            job(name+"-pack","emit-pack",[weights,cal,TEMPLATE],[pack],model=weights,calibration=cal,output=pack,
                name="full-"+name,runtime="fb4b273")
            expected=out/f"{name}-expected-sdk.jsonl"
            runner.stage(name+"-sdk-expected",command=[DEV,"eval","--model",weights,"--template",TEMPLATE,"--calibration",cal,
                "--data",data/"parity.wire.jsonl","--dump",expected],
                inputs=[DEV,weights,TEMPLATE,cal,data/"parity.wire.jsonl"],outputs=[expected])
            runner.stage(name+"-sdk-pack",command=["node",ROOT/"sweep/check_pack.mjs",pack,data/"parity.wire.jsonl",expected,out/(name+"-sdk-pack.json")],
                inputs=[pack,expected,ROOT/"sweep/check_pack.mjs",ROOT/"ts/native/lars.darwin-arm64.node",data/"parity.wire.jsonl"],
                outputs=[out/(name+"-sdk-pack.json")])
        write_results(out,captures,certified,smoke=args.smoke)
    atomic_json(out/"status.json",{"state":"complete","finished":time.time(),"promotion":json.loads((out/"decision.json").read_text())})
    print("Full confirmation complete; decision recorded, no pack automatically promoted",flush=True)


def write_results(out,captures,certified,smoke=False):
    seeds=sorted(int(name.rsplit("s",1)[-1]) for name in captures if name.startswith("baseline-"))
    suites={}
    for dataset in ("confirmation","typed-regression","ood","open-confirmation"):
        values={}
        for recipe in ("baseline","distilled"):
            ms=[summary(captures[f"{recipe}-s{s}"][dataset]) for s in seeds]
            values[recipe]={"accuracy":float(np.mean([m["macro_accuracy"] for m in ms])),
                "brier":float(np.mean([m["macro_brier"] for m in ms])),
                "score_accuracy":float(np.mean([m["scores"].get("accuracy",0) for m in ms])),"seeds":ms}
        values["accuracy_deltas"]=[b["macro_accuracy"]-a["macro_accuracy"] for a,b in zip(values["baseline"]["seeds"],values["distilled"]["seeds"],strict=True)]
        values["score_accuracy_deltas"]=[b["scores"].get("accuracy",0)-a["scores"].get("accuracy",0) for a,b in zip(values["baseline"]["seeds"],values["distilled"]["seeds"],strict=True)]
        suites[dataset]=values
    intervals={"accuracy":paired_interval([captures[f"baseline-s{s}"]["confirmation"] for s in seeds],[captures[f"distilled-s{s}"]["confirmation"] for s in seeds]),
               "score_accuracy":paired_interval([captures[f"baseline-s{s}"]["confirmation"] for s in seeds],[captures[f"distilled-s{s}"]["confirmation"] for s in seeds],score_only=True)}
    coverage={recipe:{"macro_coverage":float(np.mean([certified[f"{recipe}-s{s}"]["macro_coverage"] for s in seeds])),
        "accepted":sum(certified[f"{recipe}-s{s}"]["accepted"] for s in seeds),
        "errors":sum(sum(v["errors"] for v in certified[f"{recipe}-s{s}"]["per_source"].values()) for s in seeds)} for recipe in ("baseline","distilled")}
    for value in coverage.values():
        value["accepted_error"]=value["errors"]/value["accepted"] if value["accepted"] else None
    for recipe,value in coverage.items():
        value["worst_seed_error_upper95"]=max(certified[f"{recipe}-s{s}"]["accepted_error_95ci"][1] for s in seeds)
    decision=promotion(suites,intervals,coverage)
    if smoke or len(seeds)<len(POLICY["seeds"]):
        decision.update(eligible=False,reason="smoke run" if smoke else "remaining seeds pending")
    atomic_json(out/"results.json",{"completed_seeds":seeds,"suites":suites,"intervals":intervals,"coverage":coverage,"certificates":certified})
    atomic_json(out/"decision.json",decision)


if __name__=="__main__":
    main()
