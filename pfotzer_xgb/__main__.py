import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description="Radiation Profiles & Pfotzer Peak Estimation (XGBoost only)")
    sub=parser.add_subparsers(dest="command",required=True)
    demo=sub.add_parser("demo-data",help="Create explicitly synthetic software fixtures")
    demo.add_argument("--output",required=True)
    prep=sub.add_parser("prepare",help="Parse explicitly verified raw files")
    prep.add_argument("--manifest",required=True)
    prep.add_argument("--output",required=True)
    for cmd in ["run","fit-final"]:
        p=sub.add_parser(cmd)
        p.add_argument("--data",required=True)
        p.add_argument("--config",required=True)
        p.add_argument("--output",required=True)
        p.add_argument("--metadata")
    p=sub.add_parser("predict")
    p.add_argument("--model",required=True)
    p.add_argument("--data",required=True)
    p.add_argument("--output",required=True)
    args=parser.parse_args()
    if args.command=="demo-data":
        from .synthetic import make_demo
        result=make_demo(args.output)
    elif args.command=="prepare":
        from .preprocessing import prepare
        result=prepare(args.manifest,args.output)
    elif args.command in ["run","fit-final"]:
        from .experiments import run,fit_final
        config=json.loads(Path(args.config).read_text())
        result=(run if args.command=="run" else fit_final)(args.data,args.output,config,args.metadata)
    else:
        from .experiments import predict_new
        result=predict_new(args.model,args.data,args.output)
    print(result)


if __name__=="__main__":
    main()
